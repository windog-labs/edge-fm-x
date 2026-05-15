#!/usr/bin/env python3
"""Owner A Phase 3 default-off decision gates.

This script reports whether candidate optimization routes are ready to become
default runtime paths. It intentionally keeps risky routes default-off.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_PREFILL_M = [512, 1024, 2048]
DEFAULT_DECODE_LENS = [32, 64]
DEFAULT_MODEL_SIZES = ["0.5b", "1.5b", "3b"]
DEFAULT_LINEAR_ROLES = ["fused_qkv", "attention_output", "fused_gate_up", "mlp_down"]
DEEPGEMM_BUILD_REQUIREMENTS = {
    "official_package": "deepseek-ai/DeepGEMM",
    "required_gpus": ["SM90", "SM100"],
    "cuda": {
        "SM90": ">=12.3, >=12.9 recommended",
        "SM100": ">=12.9",
    },
    "python": ">=3.8",
    "pytorch": ">=2.1",
    "compiler": "C++20",
    "extra_deps": ["CUTLASS >=4.0", "{fmt}"],
}
DEEPEFM_FLOAT8_GAP = (
    "edge-fm Tensor/DType currently has no explicit Float8 type and Qwen weights "
    "load as FP16/BF16, so DeepGEMM needs a future FP8/W8A8 artifact contract "
    "with activation/weight scales before runtime binding."
)


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _detect_cuda_capability() -> tuple[int, int] | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return int(major), int(minor)


def _detect_flashinfer_importable() -> bool:
    return importlib.util.find_spec("flashinfer.deep_gemm") is not None


def _find_module_origin(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return None
    return str(Path(spec.origin).resolve())


def _deepgemm_source_present(project_root: Path) -> bool:
    return (
        project_root / "third_party" / "flashinfer" / "flashinfer" / "deep_gemm.py"
    ).exists()


def _deepgemm_source_paths(project_root: Path) -> list[str]:
    candidates = [
        project_root / "third_party" / "flashinfer" / "flashinfer" / "deep_gemm.py",
        project_root
        / "third_party"
        / "flashinfer"
        / "csrc"
        / "nv_internal"
        / "tensorrt_llm"
        / "deep_gemm"
        / "fp8_gemm.cuh",
    ]
    return [str(path.resolve()) for path in candidates if path.exists()]


def _detect_flashinfer_deepgemm_supported_sms(project_root: Path) -> list[int]:
    search_paths = [
        project_root / "third_party" / "flashinfer" / "flashinfer" / "deep_gemm.py",
    ]
    module_origin = _find_module_origin("flashinfer.deep_gemm")
    if module_origin is not None:
        search_paths.append(Path(module_origin))

    supported: set[int] = set()
    for path in search_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in re.finditer(r"supported_compute_capability\(\[([^\]]+)\]\)", text):
            supported.update(int(value) for value in re.findall(r"\d+", match.group(1)))
        if "sm10x" in text or "sm100" in text:
            supported.update({100, 103})
        if "sm90" in text and "fp8_gemm" in text:
            supported.add(90)
    return sorted(supported)


def _capability_to_sm(cuda_capability: tuple[int, int] | None) -> int | None:
    if cuda_capability is None:
        return None
    return cuda_capability[0] * 10 + cuda_capability[1]


def _deepgemm_hardware_supported(cuda_capability: tuple[int, int] | None) -> bool:
    sm = _capability_to_sm(cuda_capability)
    return sm is not None and (sm == 90 or sm >= 100)


def _build_deepgemm_shape_support_matrix(
    *,
    cuda_capability: tuple[int, int] | None,
    flashinfer_supported_sms: list[int],
    source_present: bool,
    flashinfer_importable: bool,
) -> list[dict[str, Any]]:
    sm = _capability_to_sm(cuda_capability)
    adapter_hardware_supported = sm in flashinfer_supported_sms if sm is not None else False
    local_adapter_supported = source_present and flashinfer_importable and adapter_hardware_supported
    common_blockers = []
    if not source_present:
        common_blockers.append("flashinfer DeepGEMM adapter source is missing")
    if not flashinfer_importable:
        common_blockers.append("flashinfer.deep_gemm is not importable in the active Python env")
    if not adapter_hardware_supported:
        common_blockers.append(
            f"current GPU sm{sm if sm is not None else 'unknown'} is not supported by the local adapter"
        )
    common_blockers.append(DEEPEFM_FLOAT8_GAP)

    matrix: list[dict[str, Any]] = []
    for role in DEFAULT_LINEAR_ROLES:
        matrix.append(
            {
                "layer_role": role,
                "stage": "prefill",
                "current_edgefm_contract": "dense FP16/BF16 linear",
                "candidate_deepgemm_contract": "future FP8/W8A8 dense or grouped GEMM with scale tensors",
                "local_flashinfer_adapter_supported": local_adapter_supported,
                "default_runtime_safe": False,
                "blockers": common_blockers,
            }
        )

    matrix.append(
        {
            "layer_role": "lm_head",
            "stage": "decode",
            "current_edgefm_contract": "m=1 full-logits FP16/BF16->Float32 linear",
            "candidate_deepgemm_contract": "not selected for this round",
            "local_flashinfer_adapter_supported": False,
            "default_runtime_safe": False,
            "blockers": [
                "DeepGEMM is not a good m=1 decode main-path candidate in this plan",
                "full-logits correctness and sampler contract remain the default",
            ],
        }
    )
    return matrix


def build_owner_a_phase3_decision_report(
    *,
    project_root: str | Path | None = None,
    cuda_capability: tuple[int, int] | None = None,
    flashinfer_importable: bool | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve() if project_root is not None else _default_project_root()
    capability = cuda_capability if cuda_capability is not None else _detect_cuda_capability()
    flashinfer_available = (
        bool(flashinfer_importable)
        if flashinfer_importable is not None
        else _detect_flashinfer_importable()
    )
    source_present = _deepgemm_source_present(root)
    sm = _capability_to_sm(capability)
    official_deepgemm_origin = _find_module_origin("deep_gemm")
    flashinfer_origin = _find_module_origin("flashinfer.deep_gemm")
    flashinfer_supported_sms = _detect_flashinfer_deepgemm_supported_sms(root)
    hardware_supported = _deepgemm_hardware_supported(capability)
    adapter_hardware_supported = sm in flashinfer_supported_sms if sm is not None else False
    # Even on supported hardware, Owner A should not bind DeepGEMM into runtime
    # until edge-fm has an FP8/W8A8 artifact contract with scale tensors.
    runtime_binding_ready = False

    return {
        "report_contract": "edgefm.owner_a_phase3_decision_gates.v1",
        "project_root": str(root),
        "benchmark_matrix": {
            "model_sizes": DEFAULT_MODEL_SIZES,
            "prefill_lengths": DEFAULT_PREFILL_M,
            "decode_lengths": DEFAULT_DECODE_LENS,
            "cuda_graph": [False, True],
        },
        "lm_head_top1": {
            "integration_status": "experimental_default_off",
            "default_runtime": "full_logits",
            "runtime_flag": "runtime.lm_head_top1.enabled",
            "acceptance_rule": "Only consider making it default after >=1% end-to-end CUDA graph gain and exact token alignment.",
            "fallback_rule": "Unsupported dtype, shape, stage, quant path, or non-greedy sampling must fall back to full logits.",
            "fallback_required": True,
        },
        "deepgemm": {
            "integration_status": "probe_only",
            "runtime_default": "disabled",
            "source_present": source_present,
            "source_paths": _deepgemm_source_paths(root),
            "official_deep_gemm_importable": official_deepgemm_origin is not None,
            "official_deep_gemm_module_path": official_deepgemm_origin,
            "python_importable": flashinfer_available,
            "python_module": "flashinfer.deep_gemm",
            "python_module_path": flashinfer_origin,
            "cuda_capability": list(capability) if capability is not None else None,
            "sm": sm,
            "hardware_supported": hardware_supported,
            "flashinfer_adapter_supported_sms": flashinfer_supported_sms,
            "flashinfer_adapter_hardware_supported": adapter_hardware_supported,
            "eligible_scope": "prefill_dense_linear_or_fp8_w8a8",
            "excluded_scope": "m1_decode_main_path",
            "candidate_roles": DEFAULT_LINEAR_ROLES,
            "runtime_binding_ready": runtime_binding_ready,
            "build_requirements": DEEPGEMM_BUILD_REQUIREMENTS,
            "shape_support_matrix": _build_deepgemm_shape_support_matrix(
                cuda_capability=capability,
                flashinfer_supported_sms=flashinfer_supported_sms,
                source_present=source_present,
                flashinfer_importable=flashinfer_available,
            ),
            "integration_note": (
                "Do not add a default linear impl_id yet. A real binding first needs "
                "SM90/SM100 validation plus an FP8/W8A8 artifact contract carrying "
                "scale tensors and layout transforms."
            ),
            "fallback_required": True,
        },
        "prefix_kv": {
            "implementation_status": "implemented",
            "default_runtime": "enabled_when_kvcache_slot_has_prefix_token_ids",
            "runtime_contract": (
                "Continuous per-request/per-layer KV slot with read pointers at slot base, "
                "write pointers offset by prefix_size, exact request prefix validation, "
                "warmup-time prefix prefill, and request-time prefill over only the non-prefix suffix."
            ),
            "evidence": [
                "KVManager parses kvcache.requests[].prefix_token_ids and offsets write pointers.",
                "Scheduler rejects requests whose token_ids do not start with the configured prefix.",
                "StandardEngine::warmup() materializes prefix KV and can use it for decode graph capture.",
                "StandardEngine::prepare_prefill_tensors() skips prefix tokens after warmup.",
            ],
            "coverage": [
                "tests/engine/test_kvcache.py covers prefix metadata and read/write pointer offsets.",
                "tests/engine/test_qwen2_generate.py covers CUDA graph warmup with prefix.",
            ],
            "limitations": [
                "No paged attention or non-contiguous KV layout.",
                "No semantic or approximate prefix cache lookup beyond the configured request slot.",
                "Continuous INT8 KV remains separate deferred work.",
            ],
        },
        "int8_kv": {
            "implementation_status": "deferred",
            "constraint": "Owner A only preserves future continuous-buffer interface expectations; kernels are Owner C work.",
        },
    }


def format_owner_a_phase3_decision_report(report: dict[str, Any]) -> str:
    deepgemm = report["deepgemm"]
    lm_head = report["lm_head_top1"]
    prefix_kv = report["prefix_kv"]
    int8_kv = report["int8_kv"]
    return "\n".join([
        "Owner A Phase 3 decision gates",
        f"- lm_head_top1: {lm_head['integration_status']}; default={lm_head['default_runtime']}",
        (
            "- DeepGEMM: "
            f"source_present={deepgemm['source_present']}, "
            f"python_importable={deepgemm['python_importable']}, "
            f"sm={deepgemm['sm']}, "
            f"hardware_supported={deepgemm['hardware_supported']}, "
            f"adapter_sms={deepgemm['flashinfer_adapter_supported_sms']}, "
            f"binding_ready={deepgemm['runtime_binding_ready']}, "
            f"default={deepgemm['runtime_default']}"
        ),
        f"- Prefix KV: {prefix_kv['implementation_status']}",
        f"- INT8 KV: {int8_kv['implementation_status']}",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Owner A Phase 3 probe-only decision gates")
    parser.add_argument("--project-root", default=str(_default_project_root()))
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args()

    report = build_owner_a_phase3_decision_report(project_root=args.project_root)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_owner_a_phase3_decision_report(report))


if __name__ == "__main__":
    main()
