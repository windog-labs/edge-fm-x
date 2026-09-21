"""Numerical policy contract tests; not pretrained model or CUDA execution evidence."""

from dataclasses import replace
import json

import pytest

from vlaforge.numerical_context import (
    LEGACY_SCHEMA,
    SCHEMA,
    NumericalContext,
    NumericalContextError,
    offline_restore,
    snapshot,
)


@pytest.fixture
def context():
    pytest.importorskip("torch")
    return snapshot()


def test_complete_context_roundtrips_without_aliasing(context):
    value = context.to_dict()
    assert NumericalContext.from_dict(value) == context
    assert NumericalContext.from_json(context.to_json()) == context
    value["cudnn_benchmark"] = not value["cudnn_benchmark"]
    assert value != context.to_dict()
    context.require_current()


def test_v2_records_explicit_framework_and_reduction_domain(context):
    import torch

    assert context.schema == SCHEMA
    assert context.torch_version == str(torch.__version__)
    assert not context.partial
    if context.reduction_api == "torch-2.7.1/bool-reduction":
        assert (
            context.cuda_matmul_allow_fp16_reduced_precision_reduction_split_k is None
        )
        assert (
            context.cuda_matmul_allow_bf16_reduced_precision_reduction_split_k is None
        )
    else:
        assert context.reduction_api == "torch-2.10.0/reduction-and-split-k"
        assert (
            type(context.cuda_matmul_allow_fp16_reduced_precision_reduction_split_k)
            is bool
        )


def test_legacy_records_roundtrip_without_invented_fields_and_cannot_enforce(context):
    data = context.to_dict()
    for name in (
        "torch_version",
        "reduction_api",
        "cuda_matmul_allow_fp16_reduced_precision_reduction_split_k",
        "cuda_matmul_allow_bf16_reduced_precision_reduction_split_k",
    ):
        del data[name]
    data["schema"] = LEGACY_SCHEMA
    legacy = NumericalContext.from_dict(data)
    assert legacy.partial
    assert legacy.to_dict() == data
    assert len(data) == 21
    with pytest.raises(NumericalContextError, match="partial observation"):
        legacy.require_current()
    with pytest.raises(NumericalContextError, match="partial observation"):
        with offline_restore(legacy, acknowledge_process_global=True):
            pytest.fail("legacy records must not enforce uncaptured domains")
    assert snapshot() == context


@pytest.mark.parametrize(
    "version",
    ["2.8.0", "2.10.1", "2.10.0.dev1", "2.10.0.post1", "1!2.10.0", "3.0.0", "nonsense"],
)
def test_unknown_runtime_version_fails_closed(context, monkeypatch, version):
    import torch

    monkeypatch.setattr(torch, "__version__", version)
    with pytest.raises(NumericalContextError, match="version"):
        snapshot()


@pytest.mark.parametrize(
    "field",
    [
        "torch_version",
        "reduction_api",
        "cuda_matmul_allow_fp16_reduced_precision_reduction_split_k",
    ],
)
def test_missing_v2_domain_fields_are_not_defaulted(context, field):
    data = context.to_dict()
    del data[field]
    with pytest.raises(NumericalContextError, match="fields differ"):
        NumericalContext.from_dict(data)


def test_v2_rejects_unrepresentable_reduction_pair(context):
    with pytest.raises(NumericalContextError, match="reduced-precision/split-K"):
        replace(
            context,
            torch_version="2.10.0+cu128",
            reduction_api="torch-2.10.0/reduction-and-split-k",
            cuda_matmul_allow_fp16_reduced_precision_reduction=True,
            cuda_matmul_allow_fp16_reduced_precision_reduction_split_k=False,
            cuda_matmul_allow_bf16_reduced_precision_reduction_split_k=True,
        )


@pytest.mark.parametrize("caller_pair", [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize("failure_stage", ["none", "body", "setup"])
def test_actual_split_k_complete_caller_pairs_restored(
    context, caller_pair, failure_stage, monkeypatch
):
    import torch

    if context.reduction_api != "torch-2.10.0/reduction-and-split-k":
        pytest.skip(
            "requires actual Torch 2.10 split-K API; run in isolated 2.10 environment"
        )
    matmul = torch.backends.cuda.matmul
    fields = (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    )

    def pairs():
        return {
            field: (getattr(matmul, field), getattr(matmul, field + "_split_k"))
            for field in fields
        }

    original = pairs()
    try:
        for field in fields:
            setattr(matmul, field, caller_pair)
        before = snapshot()
        target = replace(
            before,
            cuda_matmul_allow_fp16_reduced_precision_reduction=True,
            cuda_matmul_allow_bf16_reduced_precision_reduction=True,
            cuda_matmul_allow_fp16_reduced_precision_reduction_split_k=True,
            cuda_matmul_allow_bf16_reduced_precision_reduction_split_k=True,
            sdpa_math_enabled=(
                not before.sdpa_math_enabled
                if failure_stage == "setup"
                else before.sdpa_math_enabled
            ),
        )
        if failure_stage == "setup":
            original_math = torch.backends.cuda.enable_math_sdp
            calls = 0

            def fail_once(value):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("setup failed")
                return original_math(value)

            monkeypatch.setattr(torch.backends.cuda, "enable_math_sdp", fail_once)

        def execute():
            with offline_restore(target, acknowledge_process_global=True):
                assert pairs() == {field: (True, True) for field in fields}
                if failure_stage == "body":
                    raise RuntimeError("body failed")

        if failure_stage != "none":
            with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
                execute()
        else:
            execute()
        assert pairs() == {field: caller_pair for field in fields}
        assert snapshot() == before
    finally:
        for field, pair in original.items():
            setattr(matmul, field, pair)


def test_cross_build_policy_restore_is_rejected_before_mutation(context):
    other = replace(
        context, torch_version=context.torch_version.split("+")[0] + "+otherbuild"
    )
    with pytest.raises(NumericalContextError, match="different PyTorch"):
        with offline_restore(other, acknowledge_process_global=True):
            pytest.fail("must reject mismatching recorded runtime")
    assert snapshot() == context


def test_split_k_change_with_unchanged_legacy_bool_is_detected(context):
    import torch

    if context.reduction_api != "torch-2.10.0/reduction-and-split-k":
        pytest.skip("requires actual Torch 2.10 split-K API")
    target = replace(
        context,
        cuda_matmul_allow_fp16_reduced_precision_reduction=False,
        cuda_matmul_allow_fp16_reduced_precision_reduction_split_k=False,
    )
    with pytest.raises(NumericalContextError, match="split_k"):
        with offline_restore(target, acknowledge_process_global=True):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
                False,
                True,
            )
    assert snapshot() == context


@pytest.mark.parametrize(
    "change", ["missing", "unknown", "schema", "boolean", "dtype", "precision", "tf32"]
)
def test_incomplete_or_unsupported_policy_fails_closed(context, change):
    value = context.to_dict()
    if change == "missing":
        del value["autocast_cuda_enabled"]
    elif change == "unknown":
        value["automatic_defaults"] = True
    elif change == "schema":
        value["schema"] = "future/2"
    elif change == "boolean":
        value["cudnn_benchmark"] = 1
    elif change == "dtype":
        value["autocast_cpu_dtype"] = "float8_unknown"
    elif change == "precision":
        value["float32_matmul_precision"] = "best"
    else:
        value["cuda_matmul_allow_tf32"] = not value["cuda_matmul_allow_tf32"]
    with pytest.raises(NumericalContextError):
        NumericalContext.from_dict(value)


def test_duplicate_serialized_fields_rejected(context):
    text = context.to_json().rstrip().removesuffix("}") + ', "cudnn_enabled": true}'
    with pytest.raises(NumericalContextError, match="duplicate"):
        NumericalContext.from_json(text)
    assert isinstance(json.loads(context.to_json()), dict)


def test_require_current_reports_mismatch_without_mutating(context):
    other = replace(context, cudnn_benchmark=not context.cudnn_benchmark)
    with pytest.raises(NumericalContextError, match="cudnn_benchmark"):
        other.require_current()
    assert snapshot() == context


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
def test_guard_restores_all_flags_and_coupled_matmul_policy(context, precision):
    changed = replace(
        context,
        float32_matmul_precision=precision,
        cuda_matmul_allow_tf32=precision != "highest",
        cudnn_allow_tf32=not context.cudnn_allow_tf32,
        cuda_matmul_allow_fp16_reduced_precision_reduction=False,
        cuda_matmul_allow_bf16_reduced_precision_reduction=False,
        autocast_cpu_enabled=True,
        autocast_cpu_dtype="bfloat16",
        autocast_cuda_enabled=True,
        autocast_cuda_dtype="bfloat16",
        autocast_cache_enabled=False,
        sdpa_flash_enabled=False,
        sdpa_mem_efficient_enabled=False,
        sdpa_math_enabled=True,
        sdpa_cudnn_enabled=False,
        sdpa_math_allow_fp16_bf16_reduction=False,
        deterministic_algorithms_enabled=True,
        deterministic_algorithms_warn_only=True,
        cudnn_enabled=False,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
    )
    with offline_restore(changed, acknowledge_process_global=True):
        assert snapshot() == changed
    assert snapshot() == context


def test_guard_restores_after_body_error(context):
    changed = replace(context, cudnn_benchmark=not context.cudnn_benchmark)
    with pytest.raises(RuntimeError, match="body failed"):
        with offline_restore(changed, acknowledge_process_global=True):
            raise RuntimeError("body failed")
    assert snapshot() == context


def test_guard_requires_explicit_offline_acknowledgement(context):
    with pytest.raises(NumericalContextError, match="acknowledgement"):
        with offline_restore(context, acknowledge_process_global=False):
            pytest.fail("must not enter")


def test_nested_guards_rejected_and_outer_restored(context):
    with offline_restore(context, acknowledge_process_global=True):
        with pytest.raises(NumericalContextError, match="nested"):
            with offline_restore(context, acknowledge_process_global=True):
                pytest.fail("must not enter")
    assert snapshot() == context


def test_external_policy_mutation_is_reported_and_restored(context):
    import torch

    with pytest.raises(NumericalContextError, match="mismatch"):
        with offline_restore(context, acknowledge_process_global=True):
            torch.backends.cudnn.benchmark = not context.cudnn_benchmark
    assert snapshot() == context


def test_missing_runtime_api_rejected_not_defaulted(context, monkeypatch):
    import torch

    monkeypatch.delattr(torch.backends.cuda, "cudnn_sdp_enabled")
    with pytest.raises(NumericalContextError, match="API unavailable"):
        snapshot()


def test_partial_setup_failure_restores_original(context, monkeypatch):
    import torch

    original = torch.backends.cuda.enable_math_sdp
    calls = 0

    def fail_once(value):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("setup failed")
        return original(value)

    monkeypatch.setattr(torch.backends.cuda, "enable_math_sdp", fail_once)
    changed = replace(
        context,
        cudnn_allow_tf32=not context.cudnn_allow_tf32,
        sdpa_math_enabled=not context.sdpa_math_enabled,
    )
    with pytest.raises(RuntimeError, match="setup failed"):
        with offline_restore(changed, acknowledge_process_global=True):
            pytest.fail("must not enter")
    assert snapshot() == context


def test_already_matching_context_does_not_rewrite_precision(context, monkeypatch):
    import torch

    def unexpected_write(*args, **kwargs):
        pytest.fail("matching policy must not trigger numerical setter side effects")

    monkeypatch.setattr(torch, "set_float32_matmul_precision", unexpected_write)
    monkeypatch.setattr(torch, "set_autocast_dtype", unexpected_write)
    monkeypatch.setattr(torch.backends.cuda, "enable_math_sdp", unexpected_write)
    with offline_restore(context, acknowledge_process_global=True):
        context.require_current()


def test_capture_rejects_policy_change_during_export(context, monkeypatch):
    import torch

    from vlaforge.frontend import capture_region
    from vlaforge.ir.program import TensorRegion, Value
    from vlaforge.ir.types import TensorType

    value = TensorType((2,), "f32")
    region = TensorRegion("numeric-policy-check", (Value("value", value),), (value,))

    class Identity(torch.nn.Module):
        def forward(self, value):
            return value.clone()

    original = torch.export.export

    def changing_export(*args, **kwargs):
        result = original(*args, **kwargs)
        torch.backends.cudnn.benchmark = not context.cudnn_benchmark
        return result

    monkeypatch.setattr(torch.export, "export", changing_export)
    with pytest.raises(NumericalContextError, match="mismatch"):
        with offline_restore(context, acknowledge_process_global=True):
            result = capture_region(region, Identity(), (torch.ones(2),))
            assert not result.supported
            assert "numerical context mismatch" in str(result.report.to_dict())
    assert snapshot() == context
