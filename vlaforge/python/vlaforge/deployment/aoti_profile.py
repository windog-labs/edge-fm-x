"""Model-independent AOTInductor candidate configurations, not parity certificates."""

from __future__ import annotations

PROFILES = ("default", "conservative", "eager-numerics", "aten-preserving")


def aoti_configs(profile: str) -> dict[str, object]:
    if profile not in PROFILES:
        raise ValueError(f"unknown AOTInductor profile: {profile}")
    configs: dict[str, object] = {
        "aot_inductor.force_mmap_weights": True,
        "aot_inductor.package": True,
    }
    if profile != "default":
        configs.update(
            force_same_precision=True,
            max_autotune_gemm_backends="ATEN",
            mixed_mm_choice="aten",
            epilogue_fusion=False,
        )
    if profile in ("eager-numerics", "aten-preserving"):
        configs.update(emulate_precision_casts=True, emulate_divison_rounding=True)
    if profile == "aten-preserving":
        # PyTorch's selective-decomposition/lite path keeps unmarked ATen calls.
        # Disable optional graph rewrites too; actual output parity remains a gate.
        configs.update(
            fallback_by_default=True,
            selective_decompose=True,
            pattern_matcher=False,
            use_pre_grad_passes=False,
            use_joint_graph_passes=False,
            use_post_grad_passes=True,
            reorder_for_locality=False,
        )
    return configs
