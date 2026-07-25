"""Runtime push bindings for passive Session invocations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vlaforge.ir.program import InputPort
from vlaforge.ir.types import ScalarType, TensorType


@dataclass(frozen=True, slots=True)
class InputStamp:
    revision: int | None = None
    timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        if self.revision is not None and self.revision < 0:
            raise ValueError("input revision must be non-negative")
        if self.timestamp_ns is not None and self.timestamp_ns < 0:
            raise ValueError("input timestamp must be non-negative")


@dataclass(frozen=True, slots=True)
class InputBinding:
    value: object
    stamp: InputStamp = InputStamp()


@dataclass(frozen=True, slots=True)
class TensorView:
    """A non-owning external tensor contract valid until Run returns."""

    data: object
    shape: tuple[int, ...]
    dtype: str
    layout: str = "contiguous"
    device: str = "cpu"
    alignment: int = 1

    def __post_init__(self) -> None:
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("TensorView shape must be static and non-negative")
        if not self.dtype or not self.layout or not self.device:
            raise ValueError("TensorView dtype/layout/device must be non-empty")
        if self.alignment < 1 or self.alignment & (self.alignment - 1):
            raise ValueError("TensorView alignment must be a power of two")


@dataclass(frozen=True, slots=True)
class ScalarValue:
    value: object
    dtype: str

    def __post_init__(self) -> None:
        if not self.dtype:
            raise ValueError("ScalarValue dtype must be non-empty")


def resolve_binding(port: "InputPort", binding: InputBinding) -> object:
    """Validate a pushed binding and return its static ABI payload."""

    payload = port.payload
    if isinstance(payload, TensorType):
        view = binding.value
        if not isinstance(view, TensorView):
            raise TypeError(
                f"tensor input @{port.name} requires TensorView, "
                f"got {type(view).__name__}"
            )
        expected_shape = tuple(int(item) for item in payload.shape)
        mismatches = []
        if view.shape != expected_shape:
            mismatches.append(
                f"shape expected={expected_shape} actual={view.shape}"
            )
        if view.dtype != payload.dtype:
            mismatches.append(
                f"dtype expected={payload.dtype} actual={view.dtype}"
            )
        if view.layout != payload.layout:
            mismatches.append(
                f"layout expected={payload.layout} actual={view.layout}"
            )
        if view.device != port.device:
            mismatches.append(
                f"device expected={port.device} actual={view.device}"
            )
        if view.alignment < port.alignment:
            mismatches.append(
                f"alignment expected>={port.alignment} actual={view.alignment}"
            )
        if mismatches:
            raise ValueError(
                f"tensor input @{port.name} contract mismatch: "
                + "; ".join(mismatches)
            )
        return view.data

    if isinstance(payload, ScalarType):
        scalar = binding.value
        if isinstance(scalar, ScalarValue):
            if scalar.dtype != payload.name:
                raise ValueError(
                    f"scalar input @{port.name} dtype mismatch: "
                    f"expected={payload.name} actual={scalar.dtype}"
                )
            value = scalar.value
        else:
            value = scalar
        if not _scalar_matches(payload.name, value):
            raise ValueError(
                f"scalar input @{port.name} value does not match "
                f"{payload.name}: {value!r}"
            )
        if port.value_range is not None:
            lower, upper = port.value_range
            if value < lower or value > upper:
                raise ValueError(
                    f"scalar input @{port.name} outside bounded profile "
                    f"[{lower}, {upper}]: {value!r}"
                )
        return value
    raise TypeError(f"unsupported input payload {payload!r}")


def default_binding(port: "InputPort") -> InputBinding:
    payload = port.payload
    value = port.default
    if isinstance(payload, TensorType):
        value = TensorView(
            value,
            tuple(int(item) for item in payload.shape),
            payload.dtype,
            payload.layout,
            port.device,
            port.alignment,
        )
    return InputBinding(value, InputStamp(revision=0))


def _scalar_matches(dtype: str, value: object) -> bool:
    if dtype == "bool":
        return isinstance(value, bool)
    if dtype in {"index", "i32", "i64", "u64"}:
        return isinstance(value, int) and not isinstance(value, bool) and (
            dtype != "u64" or value >= 0
        )
    if dtype in {"f16", "bf16", "f32", "f64"}:
        return isinstance(value, int | float) and not isinstance(value, bool)
    if dtype == "string":
        return isinstance(value, str)
    return True
