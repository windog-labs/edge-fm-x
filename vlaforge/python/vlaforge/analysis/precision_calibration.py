"""Fit trace-bound INT8 scales without claiming a low-precision deployment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np


def _sha(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("expected a lowercase SHA256 digest")


def _name(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("identity names must be nonempty normalized strings")


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _fitted_scale(minimum, maximum):
    absolute = max(abs(minimum), abs(maximum))
    unrounded = max(absolute / 127.0, float(np.finfo(np.float32).tiny))
    if unrounded > np.finfo(np.float32).max:
        raise ValueError("calibration range cannot use finite float32 scales")
    return float(np.float32(unrounded))


def _group_key(strategy, site, steps):
    if strategy == "global":
        return ("global",)
    if strategy == "site":
        return ("site", site.name)
    if site.stage == "context":
        return ("context", site.name)
    return (strategy, site.name, tuple(steps))


def _group_name(key):
    return json.dumps(key, separators=(",", ":"))


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    partition_key: str
    input_sha256: str
    noise_sha256: str

    def __post_init__(self):
        _name(self.sample_id)
        _name(self.partition_key)
        _sha(self.input_sha256)
        _sha(self.noise_sha256)


@dataclass(frozen=True)
class CalibrationSite:
    name: str
    region: str
    node: str
    artifact_sha256: str
    stage: str
    quantize: bool

    def __post_init__(self):
        for value in (self.name, self.region, self.node):
            _name(value)
        _sha(self.artifact_sha256)
        if (
            self.stage not in ("context", "iteration")
            or type(self.quantize) is not bool
        ):
            raise ValueError("site requires an explicit stage and precision selection")


@dataclass(frozen=True)
class ScaleRecord:
    site: str
    steps: tuple[int, ...]
    group: str
    scale: float
    minimum: float
    maximum: float
    observations: int
    elements: int
    all_zero: bool

    def __post_init__(self):
        _name(self.site)
        _name(self.group)
        if type(self.steps) is not tuple or any(
            type(step) is not int or step < 0 for step in self.steps
        ):
            raise ValueError(
                "scale steps require an immutable sequence of bounded indices"
            )
        if tuple(sorted(set(self.steps))) != self.steps:
            raise ValueError("scale steps must be unique and ordered")
        for value in (self.scale, self.minimum, self.maximum):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError("scale and bounds require finite floating values")
        if self.scale <= 0 or self.minimum > self.maximum:
            raise ValueError("scale must be positive and bounds ordered")
        if self.scale != _fitted_scale(self.minimum, self.maximum):
            raise ValueError("scale must equal the declared absolute-max FP32 fit")
        if any(
            type(value) is not int or value < 1
            for value in (self.observations, self.elements)
        ):
            raise ValueError(
                "scale fit requires positive observation and element counts"
            )
        if type(self.all_zero) is not bool or self.all_zero != (
            self.minimum == self.maximum == 0
        ):
            raise ValueError("all-zero status must match observed bounds")


@dataclass(frozen=True)
class PrecisionPlan:
    profile_sha256: str
    numerical_context_sha256: str
    calibration_sha256: str
    split_sha256: str
    strategy: str
    sites: tuple[CalibrationSite, ...]
    step_keys: tuple[str, ...]
    scales: tuple[ScaleRecord, ...]

    def __post_init__(self):
        for value in (
            self.profile_sha256,
            self.numerical_context_sha256,
            self.calibration_sha256,
            self.split_sha256,
        ):
            _sha(value)
        if self.strategy not in ("global", "site", "step", "step-group"):
            raise ValueError("unknown scale fitting strategy")
        if (
            not self.sites
            or not self.scales
            or not self.step_keys
            or any(
                type(value) is not tuple
                for value in (self.sites, self.scales, self.step_keys)
            )
        ):
            raise ValueError(
                "a precision plan requires nonempty immutable declarations"
            )
        if not all(isinstance(site, CalibrationSite) for site in self.sites) or not all(
            isinstance(scale, ScaleRecord) for scale in self.scales
        ):
            raise ValueError("a precision plan requires typed sites and scales")
        if len(set(self.step_keys)) != len(self.step_keys):
            raise ValueError("step keys must be unique")
        for key in self.step_keys:
            _name(key)
        declared = {site.name: site for site in self.sites}
        if len(declared) != len(self.sites):
            raise ValueError("site names must be unique")
        coverage = {}
        shared_groups = {}
        for scale in self.scales:
            if scale.site not in declared or not declared[scale.site].quantize:
                raise ValueError("scale must name a declared quantized site")
            site = declared[scale.site]
            if (
                self.strategy == "step"
                and site.stage == "iteration"
                and len(scale.steps) != 1
            ):
                raise ValueError("step scales must cover one step per group")
            if scale.group != _group_name(_group_key(self.strategy, site, scale.steps)):
                raise ValueError(
                    "scale group is inconsistent with its strategy and coverage"
                )
            pooled = (
                scale.scale,
                scale.minimum,
                scale.maximum,
                scale.observations,
                scale.elements,
            )
            if scale.group in shared_groups and pooled != shared_groups[scale.group]:
                raise ValueError(
                    "shared scale group has inconsistent pooled statistics"
                )
            shared_groups[scale.group] = pooled
            if declared[scale.site].stage == "context":
                if scale.steps or scale.site in coverage:
                    raise ValueError(
                        "context scales cannot be step-indexed or duplicated"
                    )
                coverage[scale.site] = [None]
            else:
                if not scale.steps:
                    raise ValueError("iterative scales must cover explicit steps")
                coverage.setdefault(scale.site, []).extend(scale.steps)
        for site in self.sites:
            if not site.quantize:
                continue
            expected = (
                [None] if site.stage == "context" else list(range(len(self.step_keys)))
            )
            actual = coverage.get(site.name, [])
            if site.stage == "iteration":
                actual = sorted(actual)
            if actual != expected:
                raise ValueError(
                    "scales must cover every quantized site and step exactly once"
                )

    def to_data(self):
        return {
            "schema": "vlaforge.precision_plan/1",
            **asdict(self),
            "format": "symmetric_int8",
            "quant_min": -127,
            "quant_max": 127,
            "scale_dtype": "f32",
            "rounding": "nearest_ties_to_even",
            "fit": "calibration_absolute_max",
            "scale_floor": float(np.finfo(np.float32).tiny),
            "backend_lowered": False,
            "real_low_precision_kernel_verified": False,
            "held_out_validation_complete": False,
            "lossless_verified": False,
        }

    @classmethod
    def from_data(cls, data):
        """Restore the typed offline plan without accepting deployment claims."""
        if (
            not isinstance(data, dict)
            or data.get("schema") != "vlaforge.precision_plan/1"
        ):
            raise ValueError("unsupported precision plan schema")
        try:
            result = cls(
                profile_sha256=data["profile_sha256"],
                numerical_context_sha256=data["numerical_context_sha256"],
                calibration_sha256=data["calibration_sha256"],
                split_sha256=data["split_sha256"],
                strategy=data["strategy"],
                sites=tuple(CalibrationSite(**site) for site in data["sites"]),
                step_keys=tuple(data["step_keys"]),
                scales=tuple(
                    ScaleRecord(**{**scale, "steps": tuple(scale["steps"])})
                    for scale in data["scales"]
                ),
            )
        except (TypeError, KeyError) as error:
            raise ValueError("invalid typed precision plan declarations") from error
        if _digest(result.to_data()) != _digest(data):
            raise ValueError("unknown or altered precision plan format/claim fields")
        return result

    @classmethod
    def from_json(cls, text):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate precision plan JSON key")
                result[key] = value
            return result

        def nonfinite(value):
            raise ValueError("nonfinite precision plan JSON value: " + value)

        return cls.from_data(
            json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)
        )

    @property
    def sha256(self):
        return _digest(self.to_data())


def activation_statistics(value):
    """Read a real tensor; this synchronizing diagnostic is outside timing."""
    if isinstance(value, np.ndarray):
        if value.dtype not in (
            np.dtype("float16"),
            np.dtype("float32"),
            np.dtype("float64"),
        ):
            raise ValueError("calibration requires floating tensors")
        owned = np.array(value, copy=True, order="C", subok=False)
        dtype = owned.dtype.name
        raw = owned.tobytes()
        array = owned.astype(np.float64)
    else:
        import torch

        if (
            type(value) not in (torch.Tensor, torch.nn.Parameter)
            or value.layout != torch.strided
            or value.dtype
            not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        ):
            raise ValueError("calibration requires real strided floating tensors")
        # A singleton can be "contiguous" while retaining a non-unit last stride.
        detached = torch.empty(tuple(value.shape), dtype=value.dtype, device="cpu")
        detached.copy_(value.detach())
        dtype = str(detached.dtype).removeprefix("torch.")
        raw = detached.reshape(-1).view(torch.uint8).numpy().tobytes()
        array = detached.double().numpy()
    if not array.size or not bool(np.isfinite(array).all()):
        raise ValueError("calibration tensor must be nonempty and finite")
    return {
        "dtype": dtype,
        "shape": list(array.shape),
        "elements": int(array.size),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "tensor_sha256": hashlib.sha256(raw).hexdigest(),
    }


class PrecisionCalibration:
    """Collect complete calibration trajectories under a locked data split.

    The Adapter supplies actual profile, schedule, input/noise and graph hashes.
    This module never guesses a timestep from a model name or an activation.
    Held-out observations cannot influence fit, including differently named
    copies of a calibration observation or frames from the same partition.
    """

    def __init__(
        self,
        *,
        profile_sha256,
        numerical_context_sha256,
        step_keys,
        sites,
        calibration_samples,
        held_out_samples,
    ):
        _sha(profile_sha256)
        _sha(numerical_context_sha256)
        self.profile_sha256 = profile_sha256
        self.numerical_context_sha256 = numerical_context_sha256
        self.step_keys = tuple(step_keys)
        if not self.step_keys or len(set(self.step_keys)) != len(self.step_keys):
            raise ValueError(
                "step keys must uniquely identify each actual schedule entry"
            )
        for key in self.step_keys:
            _name(key)
        self.sites = tuple(sites)
        if not self.sites or not all(
            isinstance(site, CalibrationSite) for site in self.sites
        ):
            raise ValueError("typed calibration sites are required")
        if len({site.name for site in self.sites}) != len(self.sites):
            raise ValueError("calibration site names must be unique")
        if len({(site.region, site.node) for site in self.sites}) != len(self.sites):
            raise ValueError(
                "an actual graph site cannot be duplicated under another name"
            )
        self.calibration_samples = tuple(calibration_samples)
        self.held_out_samples = tuple(held_out_samples)
        for samples in (self.calibration_samples, self.held_out_samples):
            if not samples or not all(
                isinstance(sample, CalibrationSample) for sample in samples
            ):
                raise ValueError(
                    "nonempty typed calibration and held-out splits are required"
                )
            if len({sample.sample_id for sample in samples}) != len(samples):
                raise ValueError("sample IDs within a split must be unique")
            if len({sample.input_sha256 for sample in samples}) != len(samples):
                raise ValueError(
                    "duplicated observations must not be counted as independent samples"
                )
        for field in ("sample_id", "partition_key", "input_sha256"):
            train = {getattr(sample, field) for sample in self.calibration_samples}
            held = {getattr(sample, field) for sample in self.held_out_samples}
            if train & held:
                raise ValueError("calibration and held-out splits overlap: " + field)
        self._sites = {site.name: site for site in self.sites}
        self._samples = {
            sample.sample_id: sample for sample in self.calibration_samples
        }
        self._records = {}
        self._profiles = {}
        self._locked_declarations = self._declaration_digest()

    def _declaration_digest(self):
        return _digest(
            {
                "profile": self.profile_sha256,
                "numerical_context": self.numerical_context_sha256,
                "steps": self.step_keys,
                "sites": [asdict(site) for site in self.sites],
                "calibration": [asdict(sample) for sample in self.calibration_samples],
                "held_out": [asdict(sample) for sample in self.held_out_samples],
            }
        )

    def _require_locked(self):
        if self._declaration_digest() != self._locked_declarations:
            raise ValueError("calibration declarations changed after locking")

    def observe(self, *, sample_id, site, step, tensor):
        self._require_locked()
        if sample_id not in self._samples:
            raise ValueError(
                "only declared calibration samples may contribute statistics"
            )
        if site not in self._sites:
            raise ValueError("unknown calibration site")
        spec = self._sites[site]
        if spec.stage == "context":
            if step is not None:
                raise ValueError(
                    "context calibration is once per observation, not per step"
                )
        elif type(step) is not int or not 0 <= step < len(self.step_keys):
            raise ValueError(
                "iteration calibration requires an actual bounded step index"
            )
        key = (sample_id, site, step)
        if key in self._records:
            raise ValueError("duplicate calibration observation")
        statistics = activation_statistics(tensor)
        profile = (statistics["dtype"], tuple(statistics["shape"]))
        if site in self._profiles and profile != self._profiles[site]:
            raise ValueError(
                "calibration tensor shape/dtype differs from the locked site profile"
            )
        self._profiles[site] = profile
        self._records[key] = {
            "sample_id": sample_id,
            "site": site,
            "step": step,
            **statistics,
        }

    def _complete_records(self):
        records = []
        for sample in self.calibration_samples:
            for site in self.sites:
                steps = (
                    (None,) if site.stage == "context" else range(len(self.step_keys))
                )
                for step in steps:
                    key = (sample.sample_id, site.name, step)
                    if key not in self._records:
                        raise ValueError(
                            "incomplete calibration trajectory: " + str(key)
                        )
                    records.append(self._records[key])
        return records

    def report(self):
        self._require_locked()
        return {
            "schema": "vlaforge.activation_calibration/1",
            "profile_sha256": self.profile_sha256,
            "numerical_context_sha256": self.numerical_context_sha256,
            "step_keys": list(self.step_keys),
            "sites": [asdict(site) for site in self.sites],
            "split": {
                "calibration": [asdict(sample) for sample in self.calibration_samples],
                "held_out": [asdict(sample) for sample in self.held_out_samples],
            },
            "observations": [
                {**row, "shape": list(row["shape"])} for row in self._complete_records()
            ],
            "timing_evidence": False,
            "held_out_used_for_fit": False,
        }

    def fit(self, strategy, *, step_groups=None):
        self._require_locked()
        if strategy not in ("global", "site", "step", "step-group"):
            raise ValueError("unknown scale fitting strategy")
        if strategy == "step-group":
            groups = (
                tuple(tuple(group) for group in step_groups)
                if step_groups is not None
                else ()
            )
            if (
                not groups
                or any(not group for group in groups)
                or any(type(step) is not int for group in groups for step in group)
                or sorted(step for group in groups for step in group)
                != list(range(len(self.step_keys)))
            ):
                raise ValueError(
                    "step groups must partition the complete schedule exactly once"
                )
        elif step_groups is not None:
            raise ValueError(
                "step groups only apply to the explicit step-group strategy"
            )
        else:
            groups = tuple((step,) for step in range(len(self.step_keys)))
        report = self.report()
        grouped = {}
        for row in report["observations"]:
            site = self._sites[row["site"]]
            if not site.quantize:
                continue
            steps = (
                ()
                if site.stage == "context"
                else tuple(
                    sorted(next(group for group in groups if row["step"] in group))
                )
            )
            key = _group_key(strategy, site, steps)
            grouped.setdefault(key, []).append(row)
        if not grouped:
            raise ValueError("at least one site must explicitly select quantization")
        scales = []
        for key, rows in sorted(grouped.items()):
            minimum = min(row["minimum"] for row in rows)
            maximum = max(row["maximum"] for row in rows)
            absolute = max(abs(minimum), abs(maximum))
            scale = _fitted_scale(minimum, maximum)
            for site in sorted({row["site"] for row in rows}):
                selected = [row for row in rows if row["site"] == site]
                steps = tuple(
                    sorted({row["step"] for row in selected if row["step"] is not None})
                )
                scales.append(
                    ScaleRecord(
                        site,
                        steps,
                        _group_name(key),
                        scale,
                        minimum,
                        maximum,
                        len(rows),
                        sum(row["elements"] for row in rows),
                        absolute == 0,
                    )
                )
        return PrecisionPlan(
            self.profile_sha256,
            self.numerical_context_sha256,
            _digest(report),
            _digest(report["split"]),
            strategy,
            self.sites,
            self.step_keys,
            tuple(scales),
        )
