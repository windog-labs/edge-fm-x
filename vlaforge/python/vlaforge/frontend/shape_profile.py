"""Bounded shape profiles shared by torch.export and artifact contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vlaforge.deployment.contract import ShapeDimension, ValueContract
from vlaforge.ir.program import Value
from vlaforge.ir.types import TensorType


@dataclass(frozen=True, slots=True)
class DynamicDimension:
    value: str
    index: int
    symbol: str
    minimum: int
    optimum: int
    maximum: int

    def __post_init__(self) -> None:
        ShapeDimension.bounded(
            self.symbol, self.minimum, self.optimum, self.maximum
        )
        if not self.value:
            raise ValueError("dynamic dimension value name must be non-empty")
        if self.index < 0:
            raise ValueError("dynamic dimension index must be non-negative")


@dataclass(frozen=True, slots=True)
class ShapeProfile:
    dimensions: tuple[DynamicDimension, ...] = ()

    def __post_init__(self) -> None:
        keys = [(item.value, item.index) for item in self.dimensions]
        symbols = [item.symbol for item in self.dimensions]
        if len(keys) != len(set(keys)):
            raise ValueError("shape profile contains duplicate value dimensions")
        if len(symbols) != len(set(symbols)):
            raise ValueError("shape profile symbols must be globally unique")

    def bounds_for(self, value: str) -> dict[int, tuple[str, int, int, int]]:
        return {
            item.index: (
                item.symbol,
                item.minimum,
                item.optimum,
                item.maximum,
            )
            for item in self.dimensions
            if item.value == value
        }

    def value_contracts(
        self,
        values: tuple[Value, ...],
        *,
        devices: Mapping[str, str],
        alignments: Mapping[str, int] | None = None,
    ) -> tuple[ValueContract, ...]:
        known = {value.name for value in values}
        unknown = sorted({item.value for item in self.dimensions} - known)
        if unknown:
            raise ValueError(
                f"shape profile references unknown values: {unknown}"
            )
        missing_devices = sorted(known - set(devices))
        if missing_devices:
            raise ValueError(
                f"device mapping is missing values: {missing_devices}"
            )
        alignment_map = dict(alignments or {})
        return tuple(
            ValueContract.from_ir(
                value.name,
                value.type,
                device=devices[value.name],
                dynamic_bounds=self.bounds_for(value.name),
                alignment=alignment_map.get(value.name, 1),
            )
            for value in values
        )

    def validate_examples(
        self, values: tuple[Value, ...], examples: tuple[object, ...]
    ) -> None:
        if len(values) != len(examples):
            raise ValueError(
                f"expected {len(values)} examples, got {len(examples)}"
            )
        for value, example in zip(values, examples, strict=True):
            if not isinstance(value.type, TensorType):
                continue
            shape = getattr(example, "shape", None)
            if shape is None:
                raise ValueError(f"value {value.name} requires a tensor example")
            if len(shape) != len(value.type.shape):
                raise ValueError(
                    f"value {value.name}: example rank {len(shape)} does not "
                    f"match declared rank {len(value.type.shape)}"
                )
            bounds = self.bounds_for(value.name)
            for index, (declared, actual) in enumerate(
                zip(value.type.shape, shape, strict=True)
            ):
                actual_size = int(actual)
                if declared is not None and actual_size != declared:
                    raise ValueError(
                        f"value {value.name}: dimension {index} expected "
                        f"{declared}, got {actual_size}"
                    )
                if declared is None:
                    if index not in bounds:
                        raise ValueError(
                            f"value {value.name}: missing bounded profile for "
                            f"dynamic dimension {index}"
                        )
                    _, minimum, _, maximum = bounds[index]
                    if not minimum <= actual_size <= maximum:
                        raise ValueError(
                            f"value {value.name}: dimension {index}={actual_size} "
                            f"is outside [{minimum}, {maximum}]"
                        )

    def torch_dynamic_shapes(self, values: tuple[Value, ...]) -> object | None:
        """Build the positional dynamic-shape structure accepted by torch.export."""

        if not self.dimensions:
            return None
        import torch

        result: list[dict[int, Any] | None] = []
        any_dynamic = False
        for value in values:
            dimensions: dict[int, Any] = {}
            for index, (symbol, minimum, _, maximum) in self.bounds_for(
                value.name
            ).items():
                dimensions[index] = torch.export.Dim(
                    symbol, min=minimum, max=maximum
                )
            any_dynamic |= bool(dimensions)
            result.append(dimensions or None)
        return tuple(result) if any_dynamic else None
