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
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.horizon.j6m_rewrite import (
    prepare_j6m_rewrite_artifacts,
    should_enable_j6m_rewrite,
)


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
    for entry in compile_spec.get("stages", []) or []:
        if isinstance(entry, dict) and entry.get("name") == stage:
            return {
                "stage": stage,
                "inputs": {
                    item.get("name", ""): item.get("shape", [])
                    for item in entry.get("inputs", [])
                    if isinstance(item, dict) and item.get("name")
                },
                "outputs": {
                    item.get("name", ""): item.get("shape", [])
                    for item in entry.get("outputs", [])
                    if isinstance(item, dict) and item.get("name")
                },
                "kv_layout": entry.get("kv_layout", ""),
            }

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


def _stage_entry(compile_spec: dict[str, Any], stage: str) -> dict[str, Any] | None:
    for entry in compile_spec.get("stages", []) or []:
        if isinstance(entry, dict) and entry.get("name") == stage:
            return entry
    return None


def _require_stage_entry(compile_spec: dict[str, Any], stage: str) -> dict[str, Any]:
    entry = _stage_entry(compile_spec, stage)
    if entry is None:
        raise ValueError(f"Stage '{stage}' was not found in compile spec")
    return entry


def _artifact_path_for_stage(
    compile_spec: dict[str, Any],
    stage: str,
    cli_artifact_path: str | None,
) -> Path:
    if cli_artifact_path:
        return Path(cli_artifact_path).resolve()
    entry = _stage_entry(compile_spec, stage)
    if entry and entry.get("artifact_path"):
        return Path(str(entry["artifact_path"])).resolve()
    return Path(compile_spec["artifact"]["artifact_path"]).resolve()


def _write_preparation_manifest(
    prep_dir: Path,
    compile_spec_path: Path,
    compile_spec: dict[str, Any],
    module_path: Path,
    stage: str,
    artifact_path: Path,
    compiler_command: str | None,
    horizon_rewrite: dict[str, Any] | None,
) -> Path:
    manifest = {
        "schema": "edgefm_horizon_compile_prep_v1",
        "compile_spec_path": str(compile_spec_path),
        "module_path": str(module_path),
        "factory_function": compile_spec["compile_entry"]["factory_function"],
        "stage": stage,
        "stages": compile_spec.get("stages", []),
        "artifact_path": str(artifact_path),
        "generated_module": compile_spec.get("generated_module", {}),
        "example_inputs": _build_example_inputs(compile_spec, stage),
        "compiler_command_template": compiler_command or "",
        "horizon_rewrite": horizon_rewrite or {"enabled": False},
    }
    prep_path = prep_dir / "compile_prep.json"
    prep_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return prep_path


def _instantiate_model(module_path: Path, factory_name: str, stage: str, skip_model_init: bool = False):
    if skip_model_init:
        return None
    module = _import_module_from_path(module_path)
    if not hasattr(module, factory_name):
        raise AttributeError(f"Factory '{factory_name}' not found in {module_path}")
    factory = getattr(module, factory_name)
    model = factory(stage=stage)
    return model


def _tensor_shape(tensor_desc: dict[str, Any]) -> list[int]:
    shape = tensor_desc.get("shape", [])
    if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
        name = tensor_desc.get("name", "<unnamed>")
        raise ValueError(f"Tensor '{name}' must declare a static integer shape")
    return [int(dim) for dim in shape]


def _torch_dtype(dtype: str):
    import torch

    normalized = dtype.lower()
    if normalized in {"float", "float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"int32", "i32"}:
        return torch.int32
    if normalized in {"int64", "i64", "long"}:
        return torch.int64
    if normalized in {"uint8", "u8"}:
        return torch.uint8
    if normalized in {"bool", "boolean"}:
        return torch.bool
    raise ValueError(f"Unsupported tensor dtype for ONNX export dummy input: {dtype}")


def _dummy_tensor_for_input(tensor_desc: dict[str, Any]):
    import torch

    name = str(tensor_desc.get("name", ""))
    shape = _tensor_shape(tensor_desc)
    dtype = _torch_dtype(str(tensor_desc.get("dtype", "float32")))
    if dtype.is_floating_point:
        return torch.zeros(shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.ones(shape, dtype=dtype)
    if dtype == torch.uint8 and "mask" in name:
        return torch.ones(shape, dtype=dtype)
    if "position" in name and len(shape) >= 2:
        seq_len = shape[-1]
        base = torch.arange(seq_len, dtype=dtype).reshape([1] * (len(shape) - 1) + [seq_len])
        return base.expand(shape).contiguous()
    return torch.zeros(shape, dtype=dtype)


def _build_dummy_inputs(stage_entry: dict[str, Any]) -> tuple[list[Any], list[str], list[str]]:
    inputs = [
        item
        for item in stage_entry.get("inputs", []) or []
        if isinstance(item, dict) and item.get("name")
    ]
    outputs = [
        item
        for item in stage_entry.get("outputs", []) or []
        if isinstance(item, dict) and item.get("name")
    ]
    return (
        [_dummy_tensor_for_input(item) for item in inputs],
        [str(item["name"]) for item in inputs],
        [str(item["name"]) for item in outputs],
    )


def _normalize_onnx_ir_version(onnx_path: Path, ir_version: int) -> None:
    if ir_version <= 0:
        return
    import onnx

    model = onnx.load(str(onnx_path))
    if model.ir_version > ir_version:
        model.ir_version = ir_version
        onnx.save(model, str(onnx_path))


def _export_stage_to_onnx(
    model: Any,
    compile_spec: dict[str, Any],
    stage: str,
    onnx_path: Path,
    opset_version: int,
    onnx_ir_version: int,
) -> Path:
    if model is None:
        raise ValueError("--export-onnx requires model initialization; remove --skip-model-init")

    import torch

    stage_entry = _require_stage_entry(compile_spec, stage)
    example_inputs, input_names, output_names = _build_dummy_inputs(stage_entry)
    if not input_names:
        raise ValueError(f"Stage '{stage}' does not declare inputs")
    if not output_names:
        raise ValueError(f"Stage '{stage}' does not declare outputs")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(example_inputs),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            do_constant_folding=True,
        )
    _normalize_onnx_ir_version(onnx_path, onnx_ir_version)
    return onnx_path


def _shape_to_horizon(shape: list[int]) -> str:
    return "x".join(str(dim) for dim in shape)


def _write_hb_compile_config(
    stage_entry: dict[str, Any],
    *,
    onnx_path: Path,
    config_path: Path,
    working_dir: Path,
    output_prefix: str,
    march: str,
    jobs: int,
    optimize_level: str,
    compile_mode: str,
    core_num: int,
) -> Path:
    inputs = [
        item
        for item in stage_entry.get("inputs", []) or []
        if isinstance(item, dict) and item.get("name")
    ]
    if not inputs:
        raise ValueError(f"Stage '{stage_entry.get('name', '<unknown>')}' does not declare inputs")
    input_names = [str(item["name"]) for item in inputs]
    input_shapes = [_shape_to_horizon(_tensor_shape(item)) for item in inputs]
    featuremaps = ";".join("featuremap" for _ in inputs)
    no_preprocess = ";".join("no_preprocess" for _ in inputs)

    config = {
        "model_parameters": {
            "onnx_model": str(onnx_path),
            "march": march,
            "output_model_file_prefix": output_prefix,
            "working_dir": str(working_dir),
            "layer_out_dump": False,
        },
        "input_parameters": {
            "input_name": ";".join(input_names),
            "input_shape": ";".join(input_shapes),
            "input_type_rt": featuremaps,
            "input_type_train": featuremaps,
            "norm_type": no_preprocess,
            "input_space_and_range": "",
        },
        "compiler_parameters": {
            "compile_mode": compile_mode,
            "core_num": core_num,
            "jobs": jobs,
            "max_time_per_fc": 0,
            "optimize_level": optimize_level,
        },
        "calibration_parameters": {},
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
    except Exception:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def _run_hb_compile(config_path: Path, artifact_path: Path, working_dir: Path, output_prefix: str) -> Path:
    working_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["hb_compile", "-c", str(config_path)], check=True)
    compiled_hbm = working_dir / f"{output_prefix}.hbm"
    if not compiled_hbm.exists():
        raise FileNotFoundError(f"hb_compile finished but HBM was not found at {compiled_hbm}")
    if compiled_hbm.resolve() != artifact_path.resolve():
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(compiled_hbm, artifact_path)
    return artifact_path


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
    parser.add_argument(
        "--stage",
        default="prefill",
        choices=["prefill", "decode"],
        help="Model stage to instantiate",
    )
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
    parser.add_argument(
        "--horizon-rewrite",
        default="auto",
        choices=["auto", "on", "off"],
        help="Prepare J6M-safe rewrite diagnostics before invoking the compiler",
    )
    parser.add_argument(
        "--lerobot-root",
        default=None,
        help="Override LeRobot checkout root used by SmolVLA J6M rewrite diagnostics",
    )
    parser.add_argument(
        "--skip-model-init",
        action="store_true",
        help="Do not instantiate the generated model; useful for rewrite dry-runs without model weights",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the selected stage to ONNX using dummy inputs from compile_spec.stages",
    )
    parser.add_argument(
        "--onnx-path",
        default=None,
        help="Override ONNX output path. Defaults to <artifact-dir>/<stage>.onnx",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=17,
        help="ONNX opset used by torch.onnx.export when --export-onnx is enabled",
    )
    parser.add_argument(
        "--onnx-ir-version",
        type=int,
        default=9,
        help="Clamp ONNX IR version after export; use 0 to keep exporter default",
    )
    parser.add_argument(
        "--reuse-onnx",
        action="store_true",
        help="Use an existing --onnx-path for --hb-compile instead of exporting again",
    )
    parser.add_argument(
        "--hb-compile",
        action="store_true",
        help="Run hb_compile after ONNX export and copy the HBM to the stage artifact path",
    )
    parser.add_argument(
        "--hb-config-path",
        default=None,
        help="Override generated hb_compile YAML path. Defaults to <artifact-dir>/<stage>_hb_compile.yaml",
    )
    parser.add_argument(
        "--hb-working-dir",
        default=None,
        help="Override hb_compile working_dir. Defaults to <artifact-dir>/.hb_compile_<stage>",
    )
    parser.add_argument(
        "--hb-march",
        default="nash-m",
        help="Horizon march passed to hb_compile YAML, e.g. nash-m for J6M",
    )
    parser.add_argument(
        "--hb-jobs",
        type=int,
        default=32,
        help="hb_compile compiler_parameters.jobs",
    )
    parser.add_argument(
        "--hb-optimize-level",
        default="O0",
        help="hb_compile compiler_parameters.optimize_level",
    )
    parser.add_argument(
        "--hb-compile-mode",
        default="latency",
        help="hb_compile compiler_parameters.compile_mode",
    )
    parser.add_argument(
        "--hb-core-num",
        type=int,
        default=1,
        help="hb_compile compiler_parameters.core_num",
    )
    args = parser.parse_args()

    compile_spec_path = Path(args.compile_spec).resolve()
    compile_spec = _load_json(compile_spec_path)
    if compile_spec.get("backend") != "horizon":
        raise ValueError(f"Unsupported backend in compile spec: {compile_spec.get('backend')}")

    generated_module = compile_spec.get("generated_module", {})
    module_path = Path(generated_module["module_path"]).resolve()
    factory_function = compile_spec["compile_entry"]["factory_function"]
    artifact_path = _artifact_path_for_stage(compile_spec, args.stage, args.artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    prep_dir = artifact_path.parent
    rewrite_result: dict[str, Any] | None = None
    if should_enable_j6m_rewrite(compile_spec, args.horizon_rewrite):
        rewrite_result = prepare_j6m_rewrite_artifacts(
            compile_spec_path=compile_spec_path,
            compile_spec=compile_spec,
            output_dir=prep_dir,
            stage=args.stage,
            lerobot_root=args.lerobot_root,
        )

    model = _instantiate_model(
        module_path,
        factory_function,
        args.stage,
        skip_model_init=args.skip_model_init,
    )
    if model is not None and rewrite_result is not None:
        rewrite_result = prepare_j6m_rewrite_artifacts(
            compile_spec_path=compile_spec_path,
            compile_spec=compile_spec,
            output_dir=prep_dir,
            stage=args.stage,
            lerobot_root=args.lerobot_root,
            model=model,
        )

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
        horizon_rewrite=rewrite_result,
    )

    model_summary = {
        "class": model.__class__.__name__ if model is not None else "",
        "module": model.__class__.__module__ if model is not None else "",
        "stage": args.stage,
        "artifact_path": str(artifact_path),
        "prep_manifest_path": str(prep_manifest_path),
        "model_init_skipped": model is None,
    }
    (prep_dir / "model_summary.json").write_text(
        json.dumps(model_summary, indent=2), encoding="utf-8"
    )

    onnx_path: Path | None = None
    hb_config_path: Path | None = None
    hb_working_dir: Path | None = None
    if args.reuse_onnx:
        if not args.hb_compile:
            raise ValueError("--reuse-onnx is only meaningful with --hb-compile")
        if not args.onnx_path:
            raise ValueError("--reuse-onnx requires --onnx-path")
        onnx_path = Path(args.onnx_path).resolve()
        if not onnx_path.exists():
            raise FileNotFoundError(f"--reuse-onnx path does not exist: {onnx_path}")
    elif args.export_onnx or args.hb_compile:
        onnx_path = Path(args.onnx_path).resolve() if args.onnx_path else prep_dir / f"{args.stage}.onnx"
        _export_stage_to_onnx(
            model,
            compile_spec,
            args.stage,
            onnx_path,
            opset_version=args.onnx_opset,
            onnx_ir_version=args.onnx_ir_version,
        )

    if args.hb_compile:
        if onnx_path is None:
            raise RuntimeError("Internal error: ONNX path is missing for hb_compile")
        stage_entry = _require_stage_entry(compile_spec, args.stage)
        hb_config_path = (
            Path(args.hb_config_path).resolve()
            if args.hb_config_path
            else prep_dir / f"{args.stage}_hb_compile.yaml"
        )
        hb_working_dir = (
            Path(args.hb_working_dir).resolve()
            if args.hb_working_dir
            else prep_dir / f".hb_compile_{args.stage}"
        )
        _write_hb_compile_config(
            stage_entry,
            onnx_path=onnx_path,
            config_path=hb_config_path,
            working_dir=hb_working_dir,
            output_prefix=artifact_path.stem,
            march=args.hb_march,
            jobs=args.hb_jobs,
            optimize_level=args.hb_optimize_level,
            compile_mode=args.hb_compile_mode,
            core_num=args.hb_core_num,
        )
        _run_hb_compile(
            hb_config_path,
            artifact_path=artifact_path,
            working_dir=hb_working_dir,
            output_prefix=artifact_path.stem,
        )

    compiler_command = _resolve_compiler_command(
        args.compiler_command,
        os.environ.get("EDGE_FM_HORIZON_COMPILER_CMD"),
    )
    if args.dry_run or not compiler_command:
        status = "compiled" if args.hb_compile else "prepared"
        print(json.dumps(
            {
                "status": status,
                "compile_spec": str(compile_spec_path),
                "module_path": str(module_path),
                "prep_manifest": str(prep_manifest_path),
                "artifact_path": str(artifact_path),
                "onnx_path": str(onnx_path) if onnx_path is not None else "",
                "hb_config_path": str(hb_config_path) if hb_config_path is not None else "",
                "hb_working_dir": str(hb_working_dir) if hb_working_dir is not None else "",
                "compiler_command_configured": bool(compiler_command),
                "horizon_rewrite": rewrite_result or {"enabled": False},
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
            "horizon_rewrite": rewrite_result or {"enabled": False},
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
