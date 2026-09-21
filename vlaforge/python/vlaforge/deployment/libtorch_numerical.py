"""Explicit projection of newly observed Python flags to the LibTorch provider.

This domain covers the 22 v2 flags, not every possible execution setting.
The Python package suffix remains compile provenance; the native provider only
checks the declared LibTorch release/API domain and these actual getters.
"""

from __future__ import annotations

from vlaforge.deployment.numerical import NumericalContractError, NumericalPolicy

NAMESPACE = "libtorch.python_context_v2/1"
RELEASE = "2.10.0"
REDUCTION_API = "torch-2.10.0/reduction-and-split-k"
_STRINGS = {
    "autocast_cpu_dtype": {"float16", "bfloat16", "float32", "float64"},
    "autocast_cuda_dtype": {"float16", "bfloat16", "float32", "float64"},
    "float32_matmul_precision": {"highest", "high", "medium"},
    "reduction_api": {REDUCTION_API},
    "torch_release": {RELEASE},
}
_BOOLEANS = frozenset(
    (
        "autocast_cache_enabled",
        "autocast_cpu_enabled",
        "autocast_cuda_enabled",
        "cuda_matmul_allow_bf16_reduced_precision_reduction",
        "cuda_matmul_allow_bf16_reduced_precision_reduction_split_k",
        "cuda_matmul_allow_fp16_reduced_precision_reduction",
        "cuda_matmul_allow_fp16_reduced_precision_reduction_split_k",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_enabled",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "sdpa_cudnn_enabled",
        "sdpa_flash_enabled",
        "sdpa_math_allow_fp16_bf16_reduction",
        "sdpa_math_enabled",
        "sdpa_mem_efficient_enabled",
    )
)


def require_libtorch_policy(policy: NumericalPolicy) -> None:
    """Validate the complete supported transport domain without importing Torch."""
    values = dict(policy.values)
    if policy.namespace != NAMESPACE or set(values) != _BOOLEANS | _STRINGS.keys():
        raise NumericalContractError(
            "unsupported or incomplete LibTorch numerical policy"
        )
    for name in _BOOLEANS:
        if type(values[name]) is not bool:
            raise NumericalContractError(
                f"LibTorch numerical field {name} requires bool"
            )
    for name, accepted in _STRINGS.items():
        if type(values[name]) is not str or values[name] not in accepted:
            raise NumericalContractError(f"unsupported LibTorch numerical field {name}")
    if values["cuda_matmul_allow_tf32"] != (
        values["float32_matmul_precision"] != "highest"
    ):
        raise NumericalContractError("inconsistent LibTorch matmul/TF32 aliases")
    for dtype in ("fp16", "bf16"):
        name = f"cuda_matmul_allow_{dtype}_reduced_precision_reduction"
        if values[name] and not values[name + "_split_k"]:
            raise NumericalContractError("unsupported LibTorch reduction/split-K pair")


def policy_from_context(context) -> NumericalPolicy:
    """Convert a newly observed supported v2 context; never upgrade legacy data."""
    from vlaforge.numerical_context import SCHEMA, NumericalContext

    if not isinstance(context, NumericalContext) or context.schema != SCHEMA:
        raise NumericalContractError(
            "LibTorch provider requires a newly observed v2 context"
        )
    if context.reduction_api != REDUCTION_API:
        raise NumericalContractError(
            "LibTorch provider requires the Torch 2.10.0 API domain"
        )
    values = context.to_dict()
    values.pop("schema")
    values.pop("torch_version")
    values["torch_release"] = RELEASE
    policy = NumericalPolicy(NAMESPACE, tuple(sorted(values.items())))
    require_libtorch_policy(policy)
    return policy


def observe_policy() -> NumericalPolicy:
    from vlaforge.numerical_context import snapshot

    return policy_from_context(snapshot())
