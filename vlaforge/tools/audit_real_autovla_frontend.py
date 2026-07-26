#!/usr/bin/env python3
"""Audit the released AutoVLA checkpoint through a bounded real frontend.

The evidence is deliberately partitioned: it starts from deterministic
post-attention hidden vectors and covers the real final Qwen MLP, tied action
vocabulary projection, action codebook rollout, Semantic IR, and verified Plan.
It does not claim camera/prompt/VLM-prefill or full autoregressive generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.adapters.autovla_real import (  # noqa: E402
    AUTOVLA_ACTION_START_ID,
    AUTOVLA_ACTION_VOCAB_SIZE,
    AUTOVLA_CHECKPOINT_REVISION,
    AUTOVLA_CHECKPOINT_SIZE,
    AUTOVLA_QWEN_REVISION,
    AUTOVLA_SOURCE_SHA256,
    AUTOVLA_UPSTREAM_REVISION,
    RealAutoVLAConfig,
    build_real_autovla_program,
    capture_real_autovla_regions,
    finite_trajectory,
    load_real_autovla_regions,
    run_real_autovla_chain,
)
from vlaforge.analysis import verify  # noqa: E402
from vlaforge.compiler import CompilerProfile, compile_module  # noqa: E402
from vlaforge.interpreter import (  # noqa: E402
    InputBinding,
    InputStamp,
    Interpreter,
    TensorView,
)
from vlaforge.ir.serializer import io_schema_digest  # noqa: E402
from vlaforge.plan import PlanExecutor  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(actual: Any, expected: Any) -> dict[str, object]:
    import torch

    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    exact = bool(torch.equal(actual_cpu, expected_cpu))
    if not (actual_cpu.is_floating_point() or expected_cpu.is_floating_point()):
        return {
            "shape": list(actual_cpu.shape),
            "dtype": str(actual.dtype),
            "exact": exact,
            "maximum_absolute_error": 0.0 if exact else float("inf"),
            "mean_absolute_error": 0.0 if exact else float("inf"),
            "normalized_root_mean_square_error": 0.0 if exact else float("inf"),
        }
    actual_f64 = actual_cpu.to(torch.float64)
    expected_f64 = expected_cpu.to(torch.float64)
    difference = (actual_f64 - expected_f64).abs()
    numerator = torch.linalg.vector_norm(actual_f64 - expected_f64)
    denominator = torch.linalg.vector_norm(expected_f64)
    nrmse = float(numerator / denominator if denominator != 0 else numerator)
    return {
        "shape": list(actual_cpu.shape),
        "dtype": str(actual.dtype),
        "exact": exact,
        "maximum_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "normalized_root_mean_square_error": nrmse,
    }


def _assert_metrics(
    name: str,
    record: Mapping[str, object],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if (
        float(record["maximum_absolute_error"]) > absolute_tolerance
        and float(record["normalized_root_mean_square_error"])
        > relative_tolerance
    ):
        raise ValueError(f"AutoVLA parity failed for {name}: {record}")


def _repository_state() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "source_dirty": dirty}


def _bindings(hidden: Any, revision: int) -> dict[str, InputBinding]:
    return {
        "post_attention_hidden": InputBinding(
            TensorView(
                hidden,
                tuple(int(item) for item in hidden.shape),
                "bf16",
                device=str(hidden.device),
                alignment=64,
            ),
            InputStamp(revision=revision),
        )
    }


def _run_session(
    runtime: Any,
    hidden: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    outputs = []
    for revision in (100, 100, 101):
        result = runtime.run(
            "plan",
            inputs=_bindings(hidden, revision),
        )
        outputs.append(
            {
                "trajectory": result.committed_outputs.output("trajectory"),
                "action_tokens": result.committed_outputs.output(
                    "action_tokens"
                ),
            }
        )
    cache_events = [
        event
        for event in runtime.trace.events
        if event.kind == "cache"
        and event.data.get("region") == "autovla_decoder_mlp"
    ]
    return outputs, {
        "events": len(cache_events),
        "hits": sum(bool(event.data["hit"]) for event in cache_events),
        "misses": sum(not bool(event.data["hit"]) for event in cache_events),
    }


def _capture_records(captures: tuple[Any, ...], export_dir: Path) -> list[dict]:
    records = []
    for capture in captures:
        assert capture.evidence is not None
        program = export_dir / f"{capture.region.name}.pt2e"
        evidence = export_dir / f"{capture.region.name}.capture.json"
        records.append(
            {
                "name": capture.region.name,
                "program": str(program.resolve()),
                "program_sha256": _sha256(program),
                "program_size_bytes": program.stat().st_size,
                "evidence": str(evidence.resolve()),
                "graph_sha256": capture.evidence.graph_digest,
                "export_seconds": capture.evidence.export_seconds,
                "maximum_export_error": (
                    capture.evidence.maximum_absolute_error
                ),
                "effect_audit": capture.evidence.effect_audit.to_dict(),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--qwen-config", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--upstream-revision",
        default=AUTOVLA_UPSTREAM_REVISION,
    )
    parser.add_argument(
        "--checkpoint-revision",
        default=AUTOVLA_CHECKPOINT_REVISION,
    )
    parser.add_argument(
        "--qwen-revision",
        default=AUTOVLA_QWEN_REVISION,
    )
    parser.add_argument("--absolute-tolerance", type=float, default=1e-5)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("AutoVLA real frontend audit requires CUDA")
    if args.device != "cuda:0":
        torch.cuda.set_device(args.device)

    module = build_real_autovla_program(device=args.device)
    diagnostics = verify(module, raise_on_error=False)
    if diagnostics:
        raise ValueError(f"AutoVLA Semantic IR invalid: {diagnostics}")
    compilation = compile_module(
        module,
        profile=CompilerProfile.VERIFIED,
        default_device=args.device,
        state_device=args.device,
    )
    cache_certificate = compilation.certificate.caches[0]

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    regions = load_real_autovla_regions(
        RealAutoVLAConfig(
            source_root=args.source_root.resolve(),
            checkpoint=args.checkpoint.resolve(),
            codebook=args.codebook.resolve(),
            qwen_config=args.qwen_config.resolve(),
            device=args.device,
            upstream_revision=args.upstream_revision,
            checkpoint_revision=args.checkpoint_revision,
            qwen_revision=args.qwen_revision,
        )
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    eager_started = time.perf_counter()
    eager = run_real_autovla_chain(regions)
    torch.cuda.synchronize()
    eager_seconds = time.perf_counter() - eager_started

    capture_started = time.perf_counter()
    captures = capture_real_autovla_regions(
        regions,
        args.export_dir,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - capture_started
    capture_map = {
        capture.region.name: capture.exported_program.module()
        for capture in captures
    }

    with torch.no_grad():
        exported_hidden = capture_map["autovla_decoder_mlp"](
            regions.example_hidden
        )
        exported_logits = capture_map["autovla_action_projection"](
            exported_hidden
        )
        exported_trajectory, exported_tokens = capture_map[
            "autovla_trajectory_decode"
        ](exported_logits)
    exported = {
        "decoded_hidden": exported_hidden,
        "action_logits": exported_logits,
        "trajectory": exported_trajectory,
        "action_tokens": exported_tokens,
    }
    eager_vs_exported = {
        name: _metrics(exported[name], eager[name])
        for name in eager
    }
    for name, record in eager_vs_exported.items():
        _assert_metrics(
            name,
            record,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )

    validators = {"finite_trajectory": finite_trajectory}
    semantic = Interpreter(
        compilation.module,
        regions=capture_map,
        validators=validators,
    )
    planned = PlanExecutor(
        compilation.plan,
        compilation.module,
        regions=capture_map,
        validators=validators,
    )
    semantic_outputs, semantic_cache = _run_session(
        semantic, regions.example_hidden
    )
    plan_outputs, plan_cache = _run_session(planned, regions.example_hidden)
    if semantic_cache != {"events": 3, "hits": 1, "misses": 2}:
        raise ValueError(f"unexpected Semantic cache behavior: {semantic_cache}")
    if plan_cache != semantic_cache:
        raise ValueError(
            f"Semantic/Plan cache mismatch: {semantic_cache} vs {plan_cache}"
        )
    if semantic.trace.to_data() != planned.trace.to_data():
        raise ValueError("AutoVLA Semantic/Plan trace mismatch")

    eager_vs_semantic = {}
    semantic_vs_plan = {}
    for name in ("trajectory", "action_tokens"):
        eager_vs_semantic[name] = _metrics(
            semantic_outputs[0][name], eager[name]
        )
        semantic_vs_plan[name] = _metrics(
            plan_outputs[0][name], semantic_outputs[0][name]
        )
        _assert_metrics(
            f"eager_vs_semantic/{name}",
            eager_vs_semantic[name],
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        _assert_metrics(
            f"semantic_vs_plan/{name}",
            semantic_vs_plan[name],
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )

    if not torch.equal(
        semantic_outputs[0]["action_tokens"],
        semantic_outputs[1]["action_tokens"],
    ):
        raise ValueError("same-revision AutoVLA action tokens changed")
    if not all(
        AUTOVLA_ACTION_START_ID
        <= int(token)
        < AUTOVLA_ACTION_START_ID + AUTOVLA_ACTION_VOCAB_SIZE
        for token in eager["action_tokens"]
    ):
        raise ValueError("AutoVLA action tokens fall outside pinned vocabulary")

    source_files = []
    for relative, expected_sha256 in AUTOVLA_SOURCE_SHA256.items():
        path = args.source_root / relative
        actual_sha256 = _sha256(path)
        source_files.append(
            {
                "relative_path": relative,
                "path": str(path.resolve()),
                "sha256": actual_sha256,
                "expected_sha256": expected_sha256,
                "matches_pinned_revision": actual_sha256 == expected_sha256,
            }
        )
    peak_cuda = int(torch.cuda.max_memory_allocated())
    report = {
        "schema": "vlaforge.autovla_real_frontend/1",
        "status": "passed",
        "passed": True,
        "model": "AutoVLA PDMS 89",
        "evidence_level": "L2-partitioned-real-checkpoint-frontend",
        "evidence_kind": "real-checkpoint-weight-backed-partition-capture",
        "scope": {
            "captured": [
                "final Qwen post-attention RMSNorm+MLP residual block",
                "final RMSNorm and real action-vocabulary projection rows",
                "released 2048-entry vehicle codebook and ten-step rollout",
                "InputRevision exact reuse",
                "transactional trajectory and action-token outputs",
            ],
            "excluded": [
                "camera decoding and synchronization",
                "prompt/processor construction",
                "vision encoder and language-model prefill/attention",
                "full autoregressive token-generation loop",
                "fast/slow natural-language CoT routing",
            ],
            "claim": (
                "This is a real-weight frontend partition, not full end-to-end "
                "AutoVLA support and not a generated C++ artifact result."
            ),
        },
        "upstream": {
            "repository": "https://github.com/ucla-mobility/AutoVLA",
            "revision": args.upstream_revision,
            "code_license": "UCLA Academic Software License",
            "source_files": source_files,
        },
        "checkpoint": {
            "repository": "https://huggingface.co/Zewei-Zhou/AutoVLA",
            "revision": args.checkpoint_revision,
            "filename": args.checkpoint.name,
            "path": str(args.checkpoint.resolve()),
            "sha256": regions.checkpoint_sha256,
            "size_bytes": args.checkpoint.stat().st_size,
            "expected_size_bytes": AUTOVLA_CHECKPOINT_SIZE,
            "license": "UCLA Academic Software License; research use",
            "resolved_keys": dict(regions.resolved_keys),
            "selected_tensor_shapes": {
                name: list(shape)
                for name, shape in regions.tensor_shapes.items()
            },
            "last_decoder_layer": regions.layer_index,
        },
        "qwen": {
            "repository": "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct",
            "revision": args.qwen_revision,
            "config_path": str(args.qwen_config.resolve()),
            "config_sha256": regions.qwen_config_sha256,
        },
        "codebook": {
            "path": str(args.codebook.resolve()),
            "sha256": regions.codebook_sha256,
            "shape": [AUTOVLA_ACTION_VOCAB_SIZE, 6, 4, 2],
            "action_start_id": AUTOVLA_ACTION_START_ID,
        },
        "input_contract": {
            "post_attention_hidden": {
                "shape": list(regions.example_hidden.shape),
                "dtype": str(regions.example_hidden.dtype),
                "device": str(regions.example_hidden.device),
                "ownership": "external borrowed-until-Run-returns",
                "identity": "InputRevision; no timing or synchronization semantics",
            }
        },
        "output_contract": {
            "group": "planning",
            "trajectory": [10, 3],
            "action_tokens": [10],
            "transactional": True,
            "external_publish": False,
        },
        "semantic_ir": {
            "io_schema_digest": io_schema_digest(module),
            "regions": [region.name for region in module.regions],
            "core_op_delta": 0,
            "bounded_decode_steps": 10,
            "cache_region": cache_certificate.region,
            "cache_input_ids": list(cache_certificate.input_ids),
            "cache_state_ids": list(cache_certificate.state_ids),
            "authoritative_state_bytes": 0,
            "derived_cache": "autovla_decoder_mlp output",
            "static_arena_bytes": compilation.certificate.arena.compiled_bytes,
        },
        "correctness": {
            "eager_vs_strict_export": eager_vs_exported,
            "eager_vs_semantic_ir": eager_vs_semantic,
            "semantic_ir_vs_plan": semantic_vs_plan,
            "semantic_plan_trace_exact": True,
            "trajectory_finite": finite_trajectory(eager["trajectory"]),
            "action_tokens_in_range": True,
            "same_revision_outputs_stable": True,
        },
        "exact_reuse": {
            "run_revisions": [100, 100, 101],
            "semantic": semantic_cache,
            "plan": plan_cache,
            "expected": "one hit and two misses",
        },
        "captures": _capture_records(captures, args.export_dir),
        "timing_and_memory": {
            "checkpoint_load_seconds": load_seconds,
            "eager_partition_seconds": eager_seconds,
            "capture_total_seconds": capture_seconds,
            "peak_cuda_allocated_bytes": peak_cuda,
            "peak_host_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        },
        "environment": {
            "host": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "device": args.device,
        },
        "repository": _repository_state(),
        "reproduction": {
            "command": [
                sys.executable,
                str(Path(__file__).relative_to(_REPOSITORY_ROOT)),
                "--source-root",
                str(args.source_root.resolve()),
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--codebook",
                str(args.codebook.resolve()),
                "--qwen-config",
                str(args.qwen_config.resolve()),
                "--export-dir",
                str(args.export_dir.resolve()),
                "--report",
                "<report.json>",
                "--device",
                args.device,
            ],
            "environment": {
                "PYTHONPATH": "vlaforge/python",
                "CUDA_VISIBLE_DEVICES": os.getenv(
                    "CUDA_VISIBLE_DEVICES", "<unset>"
                ),
            },
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
