from dataclasses import replace

import pytest
from vlaforge.deployment.libtorch_numerical import (
    NAMESPACE,
    REDUCTION_API,
    RELEASE,
    policy_from_context,
    require_libtorch_policy,
)
from vlaforge.deployment.numerical import NumericalContractError, NumericalPolicy
from vlaforge.numerical_context import LEGACY_SCHEMA, SCHEMA, NumericalContext


def context():
    return NumericalContext(
        float32_matmul_precision="highest",
        cuda_matmul_allow_tf32=False,
        cudnn_allow_tf32=True,
        cuda_matmul_allow_fp16_reduced_precision_reduction=False,
        cuda_matmul_allow_bf16_reduced_precision_reduction=False,
        autocast_cpu_enabled=False,
        autocast_cpu_dtype="bfloat16",
        autocast_cuda_enabled=False,
        autocast_cuda_dtype="float16",
        autocast_cache_enabled=True,
        sdpa_flash_enabled=True,
        sdpa_mem_efficient_enabled=True,
        sdpa_math_enabled=True,
        sdpa_cudnn_enabled=True,
        sdpa_math_allow_fp16_bf16_reduction=False,
        deterministic_algorithms_enabled=False,
        deterministic_algorithms_warn_only=False,
        cudnn_enabled=True,
        cudnn_benchmark=False,
        cudnn_deterministic=False,
        schema=SCHEMA,
        torch_version="2.10.0+cu128",
        reduction_api=REDUCTION_API,
        cuda_matmul_allow_fp16_reduced_precision_reduction_split_k=False,
        cuda_matmul_allow_bf16_reduced_precision_reduction_split_k=False,
    )


def changed(policy, **values):
    return NumericalPolicy(
        policy.namespace, tuple(sorted({**dict(policy.values), **values}.items()))
    )


def test_complete_projection_retains_both_split_k_and_separates_package_version():
    observed = context()
    policy = policy_from_context(observed)
    require_libtorch_policy(policy)
    assert policy.namespace == NAMESPACE and len(policy.values) == 24
    assert dict(policy.values)["torch_release"] == RELEASE
    assert "torch_version" not in dict(policy.values)
    assert observed.torch_version == "2.10.0+cu128"
    for dtype in ("fp16", "bf16"):
        assert (
            dict(policy.values)[
                f"cuda_matmul_allow_{dtype}_reduced_precision_reduction_split_k"
            ]
            is False
        )
    assert NumericalPolicy.from_dict(policy.to_dict()) == policy


def test_historical_or_bool_only_context_cannot_be_upgraded():
    legacy = replace(
        context(),
        schema=LEGACY_SCHEMA,
        torch_version=None,
        reduction_api=None,
        cuda_matmul_allow_fp16_reduced_precision_reduction_split_k=None,
        cuda_matmul_allow_bf16_reduced_precision_reduction_split_k=None,
    )
    with pytest.raises(NumericalContractError, match="newly observed v2"):
        policy_from_context(legacy)
    bool_only = replace(
        context(),
        torch_version="2.7.1+cpu",
        reduction_api="torch-2.7.1/bool-reduction",
        cuda_matmul_allow_fp16_reduced_precision_reduction_split_k=None,
        cuda_matmul_allow_bf16_reduced_precision_reduction_split_k=None,
    )
    with pytest.raises(NumericalContractError, match="2.10.0 API"):
        policy_from_context(bool_only)


@pytest.mark.parametrize(
    "change",
    [
        {"cuda_matmul_allow_tf32": 0},
        {"autocast_cpu_dtype": "Half"},
        {"torch_release": "2.10.1"},
        {"reduction_api": "guessed"},
        {"cuda_matmul_allow_tf32": True},
        {"cuda_matmul_allow_fp16_reduced_precision_reduction": True},
        {"cuda_matmul_allow_bf16_reduced_precision_reduction": True},
    ],
)
def test_unknown_typed_values_or_coupled_state_are_rejected(change):
    with pytest.raises(NumericalContractError):
        require_libtorch_policy(changed(policy_from_context(context()), **change))


def test_every_missing_field_and_unknown_field_fail_closed():
    policy = policy_from_context(context())
    for name, _ in policy.values:
        reduced = replace(
            policy, values=tuple(pair for pair in policy.values if pair[0] != name)
        )
        with pytest.raises(NumericalContractError, match="incomplete"):
            require_libtorch_policy(reduced)
    with pytest.raises(NumericalContractError, match="incomplete"):
        require_libtorch_policy(changed(policy, unknown_backend_flag=False))
    with pytest.raises(NumericalContractError):
        require_libtorch_policy(
            replace(policy, namespace="libtorch.python_context_v2/2")
        )


@pytest.mark.parametrize(
    "reduction,split_k", [(True, True), (False, True), (False, False)]
)
def test_three_reduction_states_remain_distinct(reduction, split_k):
    observed = replace(
        context(),
        cuda_matmul_allow_fp16_reduced_precision_reduction=reduction,
        cuda_matmul_allow_fp16_reduced_precision_reduction_split_k=split_k,
    )
    policy = policy_from_context(observed)
    require_libtorch_policy(policy)
    values = dict(policy.values)
    assert (
        values["cuda_matmul_allow_fp16_reduced_precision_reduction"],
        values["cuda_matmul_allow_fp16_reduced_precision_reduction_split_k"],
    ) == (reduction, split_k)
