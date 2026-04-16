#!/usr/bin/env python3
"""
Compile EdgeFM Horizon artifacts from a generated compile spec.

This script is intentionally backend-toolchain-friendly rather than tightly
bound to one Horizon SDK API. It performs three tasks:

1. Load and validate the EdgeFM-generated compile spec.
2. Import the generated Python module and instantiate the model factory.
3. Optionally invoke an external Horizon compiler command template.

If no compiler command is provided, the script still emits a preparation
manifest and example inputs so the generated module can be consumed manually
by Horizon tooling.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected object JSON at {path}, got {type(data).__name__}")
    return data


def _import_module_from_path(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_seq_len(compile_spec: dict[str, Any]) -> int:
    engine_cfg = compile_spec.get("engine_config", {})
    kvcache = engine_cfg.get("kvcache", {})
    requests = kvcache.get("requests", [])
    if isinstance(requests, list) and requests:
        req0 = requests[0]
        if isinstance(req0, dict):
            prefix = req0.get("prefix_token_ids", [])
            max_tokens = int(req0.get("max_tokens", 1))
            prefix_len = len(prefix) if isinstance(prefix, list) else 0
            return max(1, min(max_tokens, max(prefix_len, 1)))
    return 1


def _build_example_inputs(compile_spec: dict[str, Any], stage: str) -> dict[str, Any]:
    model_cfg = compile_spec.get("model_config", {})
    hidden_size = int(model_cfg.get("hidden_size", 0))
    num_layers = int(model_cfg.get("num_hidden_layers", 0))
    num_heads = int(model_cfg.get("num_attention_heads", 1))
    num_kv_heads = int(model_cfg.get("num_key_value_heads", num_heads))
    head_dim = hidden_size // max(num_heads, 1) if hidden_size else 0
    seq_len = _default_seq_len(compile_spec) if stage == "prefill" else 1
    example = {
        "stage": stage,
        "token_ids_shape": [1, seq_len],
        "position_ids_shape": [1, seq_len],
        "embeddings_shape": [1, seq_len, hidden_size] if hidden_size else [],
        "embed_token_id": -1,
    }
    if num_layers > 0 and head_dim > 0:
        example["kv_cache_shapes"] = [
            {
                "layer_id": layer_id,
                "k": [1, max(seq_len - 1, 1), num_kv_heads, head_dim],
                "v": [1, max(seq_len - 1, 1), num_kv_heads, head_dim],
            }
            for layer_id in range(num_layers)
        ]
    return example


def _write_preparation_manifest(
    prep_dir: Path,
    compile_spec_path: Path,
    compile_spec: dict[str, Any],
    module_path: Path,
    stage: str,
    artifact_path: Path,
    compiler_command: str | None,
) -> Path:
    manifest = {
        "schema": "edgefm_horizon_compile_prep_v1",
        "compile_spec_path": str(compile_spec_path),
        "module_path": str(module_path),
        "factory_function": compile_spec["compile_entry"]["factory_function"],
        "stage": stage,
        "artifact_path": str(artifact_path),
        "generated_module": compile_spec.get("generated_module", {}),
        "example_inputs": _build_example_inputs(compile_spec, stage),
        "compiler_command_template": compiler_command or "",
    }
    prep_path = prep_dir / "compile_prep.json"
    prep_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return prep_path


def _instantiate_model(module_path: Path, factory_name: str, stage: str):
    module = _import_module_from_path(module_path)
    if not hasattr(module, factory_name):
        raise AttributeError(f"Factory '{factory_name}' not found in {module_path}")
    factory = getattr(module, factory_name)
    model = factory(stage=stage)
    return model


def _resolve_compiler_command(
    cli_command: str | None,
    env_command: str | None,
) -> str | None:
    return cli_command or env_command


def _format_compiler_command(
    template: str,
    *,
    compile_spec_path: Path,
    module_path: Path,
    factory_function: str,
    stage: str,
    artifact_path: Path,
    prep_manifest_path: Path,
) -> str:
    return template.format(
        compile_spec=shlex.quote(str(compile_spec_path)),
        module_path=shlex.quote(str(module_path)),
        factory_function=shlex.quote(factory_function),
        stage=shlex.quote(stage),
        artifact_path=shlex.quote(str(artifact_path)),
        prep_manifest=shlex.quote(str(prep_manifest_path)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile EdgeFM Horizon artifacts from compile spec")
    parser.add_argument("compile_spec", help="Path to EdgeFM-generated compile_spec.json")
    parser.add_argument("--stage", default="prefill", choices=["prefill", "decode"], help="Model stage to instantiate")
    parser.add_argument(
        "--compiler-command",
        default=None,
        help=(
            "External compiler command template. Placeholders: "
            "{compile_spec}, {module_path}, {factory_function}, {stage}, "
            "{artifact_path}, {prep_manifest}"
        ),
    )
    parser.add_argument(
        "--artifact-path",
        default=None,
        help="Override artifact output path. Defaults to compile_spec artifact.artifact_path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare module and manifest but do not invoke external compiler",
    )
    args = parser.parse_args()

    compile_spec_path = Path(args.compile_spec).resolve()
    compile_spec = _load_json(compile_spec_path)
    if compile_spec.get("backend") != "horizon":
        raise ValueError(f"Unsupported backend in compile spec: {compile_spec.get('backend')}")

    generated_module = compile_spec.get("generated_module", {})
    module_path = Path(generated_module["module_path"]).resolve()
    factory_function = compile_spec["compile_entry"]["factory_function"]
    artifact_path = Path(
        args.artifact_path or compile_spec["artifact"]["artifact_path"]
    ).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    model = _instantiate_model(module_path, factory_function, args.stage)
    prep_dir = artifact_path.parent
    prep_manifest_path = _write_preparation_manifest(
        prep_dir=prep_dir,
        compile_spec_path=compile_spec_path,
        compile_spec=compile_spec,
        module_path=module_path,
        stage=args.stage,
        artifact_path=artifact_path,
        compiler_command=_resolve_compiler_command(
            args.compiler_command,
            os.environ.get("EDGE_FM_HORIZON_COMPILER_CMD"),
        ),
    )

    model_summary = {
        "class": model.__class__.__name__,
        "module": model.__class__.__module__,
        "stage": args.stage,
        "artifact_path": str(artifact_path),
        "prep_manifest_path": str(prep_manifest_path),
    }
    (prep_dir / "model_summary.json").write_text(
        json.dumps(model_summary, indent=2), encoding="utf-8"
    )

    compiler_command = _resolve_compiler_command(
        args.compiler_command,
        os.environ.get("EDGE_FM_HORIZON_COMPILER_CMD"),
    )
    if args.dry_run or not compiler_command:
        print(json.dumps(
            {
                "status": "prepared",
                "compile_spec": str(compile_spec_path),
                "module_path": str(module_path),
                "prep_manifest": str(prep_manifest_path),
                "artifact_path": str(artifact_path),
                "compiler_command_configured": bool(compiler_command),
            },
            indent=2,
        ))
        return 0

    command = _format_compiler_command(
        compiler_command,
        compile_spec_path=compile_spec_path,
        module_path=module_path,
        factory_function=factory_function,
        stage=args.stage,
        artifact_path=artifact_path,
        prep_manifest_path=prep_manifest_path,
    )
    subprocess.run(command, shell=True, check=True)

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Compiler command finished but artifact was not found at {artifact_path}"
        )

    print(json.dumps(
        {
            "status": "compiled",
            "compile_spec": str(compile_spec_path),
            "artifact_path": str(artifact_path),
            "prep_manifest": str(prep_manifest_path),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
