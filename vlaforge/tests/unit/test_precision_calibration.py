import json
from dataclasses import replace

import numpy as np
import pytest
from vlaforge.analysis.precision_calibration import (
    CalibrationSample,
    CalibrationSite,
    PrecisionCalibration,
    PrecisionPlan,
    activation_statistics,
)


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "float64"))
@pytest.mark.parametrize("shape,stride", (((1,), (1600,)), ((1, 1), (1600, 1600))))
def test_singleton_nonunit_stride_has_canonical_value_hash(dtype, shape, stride):
    import torch

    tensor = torch.empty_strided(shape, stride, dtype=getattr(torch, dtype)).fill_(1.5)
    assert tensor.is_contiguous() and tensor.stride()[-1] == 1600
    expected = torch.full(shape, 1.5, dtype=tensor.dtype)
    assert activation_statistics(tensor) == activation_statistics(expected)


@pytest.mark.parametrize("strategy", ("global", "site", "step", "step-group"))
def test_precision_plan_json_roundtrip_is_typed_and_identity_preserving(strategy):
    collector = complete()
    kwargs = {"step_groups": ((0, 1),)} if strategy == "step-group" else {}
    plan = collector.fit(strategy, **kwargs)
    loaded = PrecisionPlan.from_json(json.dumps(plan.to_data()))
    assert loaded == plan and loaded.sha256 == plan.sha256


@pytest.mark.parametrize(
    "mutation", ("extra", "claim", "dtype", "missing", "nested-extra")
)
def test_precision_plan_loader_rejects_changed_format_or_claims(mutation):
    data = complete().fit("step").to_data()
    if mutation == "extra":
        data["unexpected"] = True
    elif mutation == "claim":
        data["real_low_precision_kernel_verified"] = True
    elif mutation == "dtype":
        data["scale_dtype"] = "f64"
    elif mutation == "missing":
        del data["calibration_sha256"]
    else:
        data["sites"][0]["unexpected"] = 1
    with pytest.raises(ValueError):
        PrecisionPlan.from_data(data)


@pytest.mark.parametrize("text", ('{"a":1,"a":2}', '{"a":NaN}', '{"a":{"b":1,"b":2}}'))
def test_precision_plan_json_rejects_duplicate_and_nonfinite_values(text):
    with pytest.raises(ValueError, match="duplicate|nonfinite"):
        PrecisionPlan.from_json(text)


def sample(name, partition, digit):
    return CalibrationSample(name, partition, digit * 64, "f" * 64)


def calibration(**overrides):
    kwargs = {
        "profile_sha256": "0" * 64,
        "numerical_context_sha256": "1" * 64,
        "step_keys": ("index0:tensor-sha-a", "index1:tensor-sha-b"),
        "sites": (
            CalibrationSite(
                "context", "prefix", "projection", "a" * 64, "context", False
            ),
            CalibrationSite("head", "step", "projection", "b" * 64, "iteration", True),
            CalibrationSite("ada", "step", "affine", "b" * 64, "iteration", True),
        ),
        "calibration_samples": (
            sample("train-a", "episode-a", "2"),
            sample("train-b", "episode-b", "3"),
        ),
        "held_out_samples": (sample("held-a", "episode-c", "4"),),
    }
    kwargs.update(overrides)
    return PrecisionCalibration(**kwargs)


def complete():
    collector = calibration()
    for name in ("train-a", "train-b"):
        collector.observe(
            sample_id=name,
            site="context",
            step=None,
            tensor=np.array([10000], dtype=np.float32),
        )
        for step in range(2):
            collector.observe(
                sample_id=name,
                site="head",
                step=step,
                tensor=np.array([-127, 127], dtype=np.float32) * (step + 1),
            )
            collector.observe(
                sample_id=name,
                site="ada",
                step=step,
                tensor=np.array([-1, 1], dtype=np.float32),
            )
    return collector


def test_four_strategies_preserve_profile_split_and_unquantized_context():
    collector = complete()
    plans = {
        strategy: collector.fit(
            strategy, **({"step_groups": ((0, 1),)} if strategy == "step-group" else {})
        )
        for strategy in ("global", "site", "step", "step-group")
    }
    assert {scale.scale for scale in plans["global"].scales} == {2.0}
    assert [scale.scale for scale in plans["step"].scales if scale.site == "head"] == [
        1.0,
        2.0,
    ]
    assert [scale.scale for scale in plans["site"].scales if scale.site == "head"] == [
        2.0
    ]
    assert len({plan.calibration_sha256 for plan in plans.values()}) == 1
    assert len({plan.sha256 for plan in plans.values()}) == 4
    for plan in plans.values():
        assert "context" not in {scale.site for scale in plan.scales}
        assert not plan.to_data()["real_low_precision_kernel_verified"]
        assert not plan.to_data()["lossless_verified"]
        assert not plan.to_data()["held_out_validation_complete"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_id", "train-a"),
        ("partition_key", "episode-a"),
        ("input_sha256", "2" * 64),
    ],
)
def test_renaming_or_new_noise_does_not_hide_split_leakage(field, value):
    held = replace(sample("held-a", "episode-c", "4"), **{field: value})
    with pytest.raises(ValueError, match="overlap"):
        calibration(held_out_samples=(held,))


def test_incomplete_or_held_out_observations_cannot_fit():
    collector = calibration()
    with pytest.raises(ValueError, match="incomplete"):
        collector.fit("step")
    with pytest.raises(ValueError, match="only declared calibration"):
        collector.observe(
            sample_id="held-a", site="head", step=0, tensor=np.ones(2, dtype=np.float32)
        )


def test_mutating_report_does_not_change_collected_statistics():
    collector = complete()
    before = collector.fit("step").sha256
    report = collector.report()
    report["observations"][0]["shape"][0] = 200
    report["observations"][0]["maximum"] = 0
    report["split"]["calibration"][0]["sample_id"] = "changed"
    assert collector.fit("step").sha256 == before


@pytest.mark.parametrize(
    "groups",
    [None, (), ((0,),), ((0, 0), (1,)), ((0,), (2,)), ((False,), (1,)), ((), (0, 1))],
)
def test_step_groups_must_partition_every_actual_step_once(groups):
    with pytest.raises(ValueError, match="partition"):
        complete().fit("step-group", step_groups=groups)


@pytest.mark.parametrize(
    "value",
    [
        np.array([], dtype=np.float32),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([1], dtype=np.int32),
    ],
)
def test_invalid_activation_does_not_become_a_scale(value):
    with pytest.raises(ValueError):
        activation_statistics(value)


def test_shape_dtype_and_duplicate_calls_fail_closed():
    collector = calibration()
    collector.observe(
        sample_id="train-a", site="head", step=0, tensor=np.ones(2, dtype=np.float32)
    )
    with pytest.raises(ValueError, match="duplicate"):
        collector.observe(
            sample_id="train-a",
            site="head",
            step=0,
            tensor=np.ones(2, dtype=np.float32),
        )
    for tensor in (np.ones(3, dtype=np.float32), np.ones(2, dtype=np.float64)):
        with pytest.raises(ValueError, match="shape/dtype"):
            collector.observe(sample_id="train-a", site="head", step=1, tensor=tensor)


@pytest.mark.parametrize(
    "site,step",
    [("context", 0), ("head", None), ("head", False), ("head", 2), ("unknown", 0)],
)
def test_context_and_iterative_observations_keep_different_semantics(site, step):
    with pytest.raises(ValueError):
        calibration().observe(
            sample_id="train-a",
            site=site,
            step=step,
            tensor=np.ones(2, dtype=np.float32),
        )


def test_plan_rejects_missing_repeated_or_out_of_range_scale_coverage():
    plan = complete().fit("step")
    for scales in (
        plan.scales[:-1],
        plan.scales + plan.scales[:1],
        (replace(plan.scales[0], steps=(9,)),) + plan.scales[1:],
    ):
        with pytest.raises(ValueError, match="cover"):
            replace(plan, scales=scales)


def test_zero_and_subnormal_ranges_have_explicit_nonzero_float32_scale():
    sites = (
        CalibrationSite("head", "step", "projection", "b" * 64, "iteration", True),
    )
    collector = calibration(sites=sites)
    for name in ("train-a", "train-b"):
        collector.observe(
            sample_id=name, site="head", step=0, tensor=np.zeros(2, dtype=np.float32)
        )
        collector.observe(
            sample_id=name,
            site="head",
            step=1,
            tensor=np.array([1e-44, 0], dtype=np.float32),
        )
    plan = collector.fit("step")
    assert plan.scales[0].all_zero is True and plan.scales[1].all_zero is False
    assert all(scale.scale == float(np.finfo(np.float32).tiny) for scale in plan.scales)


def test_actual_torch_bf16_raw_statistics_do_not_silently_cast_the_hash():
    import hashlib

    torch = pytest.importorskip("torch")
    tensor = torch.tensor([1.0, -2.0, 3.0], dtype=torch.bfloat16)
    report = activation_statistics(tensor)
    assert report["dtype"] == "bfloat16" and report["elements"] == 3
    assert report["minimum"] == -2 and report["maximum"] == 3
    assert (
        report["tensor_sha256"]
        == hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    )


def test_site_name_cannot_collide_with_step_group_encoding():
    sites = (
        CalibrationSite(
            "head:group:0", "prefix", "projection", "a" * 64, "context", True
        ),
        CalibrationSite("head", "step", "projection", "b" * 64, "iteration", True),
    )
    collector = calibration(sites=sites)
    for name in ("train-a", "train-b"):
        collector.observe(
            sample_id=name, site="head:group:0", step=None, tensor=np.array([12700.0])
        )
        for step in range(2):
            collector.observe(
                sample_id=name, site="head", step=step, tensor=np.array([127.0])
            )
    plan = collector.fit("step")
    assert [scale.scale for scale in plan.scales if scale.site == "head"] == [1.0, 1.0]
    assert len({scale.group for scale in plan.scales}) == 3


@pytest.mark.parametrize(
    "field",
    [
        "held_out_samples",
        "calibration_samples",
        "profile_sha256",
        "numerical_context_sha256",
        "sites",
        "step_keys",
    ],
)
def test_declarations_cannot_be_rebound_after_collecting_statistics(field):
    collector = complete()
    replacement = {
        "held_out_samples": collector.calibration_samples,
        "calibration_samples": collector.held_out_samples,
        "profile_sha256": "5" * 64,
        "numerical_context_sha256": "6" * 64,
        "sites": collector.sites[:-1],
        "step_keys": tuple(reversed(collector.step_keys)),
    }[field]
    setattr(collector, field, replacement)
    for operation in (
        collector.report,
        lambda: collector.fit("step"),
        lambda: collector.observe(
            sample_id="train-a", site="head", step=0, tensor=np.ones(2)
        ),
    ):
        with pytest.raises(ValueError, match="changed after locking"):
            operation()


@pytest.mark.parametrize("scale", [1e-300, 1e300, 1.1, 3.0])
def test_scale_must_match_actual_fp32_fit_not_just_python_float(scale):
    with pytest.raises(ValueError, match="FP32 fit"):
        replace(complete().fit("step").scales[0], scale=scale)


def test_plan_cannot_relabel_strategy_or_group():
    plan = complete().fit("step")
    with pytest.raises(ValueError, match="group"):
        replace(plan, strategy="site")
    with pytest.raises(ValueError, match="group"):
        replace(
            plan, scales=(replace(plan.scales[0], group="another"),) + plan.scales[1:]
        )


def test_numpy_statistics_and_hash_read_the_same_owned_plain_snapshot():
    import hashlib

    class MutatingArray(np.ndarray):
        def astype(self, *args, **kwargs):
            self[:] = 9
            return super().astype(*args, **kwargs)

    source = np.ones(2, dtype=np.float32).view(MutatingArray)
    report = activation_statistics(source)
    assert report["minimum"] == report["maximum"] == 1
    assert (
        report["tensor_sha256"]
        == hashlib.sha256(np.ones(2, dtype=np.float32).tobytes()).hexdigest()
    )
    assert np.array_equal(source, np.ones(2))
