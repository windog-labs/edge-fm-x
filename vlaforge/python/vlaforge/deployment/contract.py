"""Stable TensorRegion artifact contract.

The contract describes a compiled pure TensorRegion without importing a model
class or a backend Python package.  It is deliberately independent from the
internal scheduled plan so that an artifact can be compiled, cached, audited,
and loaded through the C ABI using only stable scalar metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from vlaforge.ir.types import IRType, TensorType, type_from_dict


ARTIFACT_SCHEMA = "vlaforge.region_artifact/2"
CALLABLE_ABI_VERSION = 2
REGION_PLUGIN_ABI = "vlaforge.region_executable/2"


def _require_nonempty(value: str, field: str) -> None:
    if not value:
        raise ValueError(f"{field} must be non-empty")


def _require_power_of_two(value: int, field: str) -> None:
    if value < 1 or value & (value - 1):
        raise ValueError(f"{field} must be a positive power of two")


class ArtifactKind(str, Enum):
    CPU_FIXTURE = "cpu_fixture"
    AOTI_PACKAGE = "aoti_package"
    TORCHSCRIPT_ARCHIVE = "torchscript_archive"
    SHARED_LIBRARY = "shared_library"
    STATIC_LIBRARY = "static_library"
    CUDA_BINARY = "cuda_binary"


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ArtifactDiagnostic:
    code: str
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    source: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.code, "diagnostic code")
        _require_nonempty(self.message, "diagnostic message")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactDiagnostic":
        return cls(
            code=str(data["code"]),
            message=str(data["message"]),
            severity=DiagnosticSeverity(str(data.get("severity", "error"))),
            source=None if data.get("source") is None else str(data["source"]),
        )


@dataclass(frozen=True, slots=True)
class ShapeDimension:
    """A static dimension or a bounded symbolic dimension."""

    static: int | None = None
    symbol: str | None = None
    minimum: int | None = None
    optimum: int | None = None
    maximum: int | None = None

    def __post_init__(self) -> None:
        if (self.static is None) == (self.symbol is None):
            raise ValueError(
                "shape dimension must define exactly one of static or symbol"
            )
        if self.static is not None:
            if self.static < 0:
                raise ValueError("static dimension must be non-negative")
            if any(
                value is not None
                for value in (self.minimum, self.optimum, self.maximum)
            ):
                raise ValueError("static dimension cannot have symbolic bounds")
            return
        _require_nonempty(str(self.symbol), "shape symbol")
        if self.minimum is None or self.optimum is None or self.maximum is None:
            raise ValueError("symbolic dimension requires min/opt/max bounds")
        if self.minimum < 0 or not (
            self.minimum <= self.optimum <= self.maximum
        ):
            raise ValueError("symbolic bounds must satisfy 0 <= min <= opt <= max")

    @classmethod
    def fixed(cls, size: int) -> "ShapeDimension":
        return cls(static=size)

    @classmethod
    def bounded(
        cls, symbol: str, minimum: int, optimum: int, maximum: int
    ) -> "ShapeDimension":
        return cls(
            symbol=symbol,
            minimum=minimum,
            optimum=optimum,
            maximum=maximum,
        )

    def to_dict(self) -> dict[str, object]:
        if self.static is not None:
            return {"static": self.static}
        return {
            "symbol": self.symbol,
            "minimum": self.minimum,
            "optimum": self.optimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShapeDimension":
        if data.get("static") is not None:
            return cls.fixed(int(data["static"]))
        return cls.bounded(
            str(data["symbol"]),
            int(data["minimum"]),
            int(data["optimum"]),
            int(data["maximum"]),
        )


@dataclass(frozen=True, slots=True)
class ValueContract:
    """ABI-visible value signature plus shape/device constraints."""

    name: str
    type: IRType
    device: str
    dimensions: tuple[ShapeDimension, ...] = ()
    alignment: int = 1

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "value name")
        _require_nonempty(self.device, "value device")
        _require_power_of_two(self.alignment, "value alignment")
        if isinstance(self.type, TensorType):
            if len(self.dimensions) != len(self.type.shape):
                raise ValueError(
                    f"value {self.name}: shape profile rank does not match "
                    f"TensorType rank {len(self.type.shape)}"
                )
            for index, (declared, profiled) in enumerate(
                zip(self.type.shape, self.dimensions, strict=True)
            ):
                if declared is not None and profiled.static != declared:
                    raise ValueError(
                        f"value {self.name}: dimension {index} requires static "
                        f"{declared}, got {profiled}"
                    )
                if declared is None and profiled.symbol is None:
                    raise ValueError(
                        f"value {self.name}: dynamic dimension {index} requires "
                        "a bounded symbol"
                    )
        elif self.dimensions:
            raise ValueError("non-tensor value cannot have shape dimensions")

    @classmethod
    def from_ir(
        cls,
        name: str,
        type: IRType,
        *,
        device: str,
        dynamic_bounds: Mapping[int, tuple[str, int, int, int]] | None = None,
        alignment: int = 1,
    ) -> "ValueContract":
        dimensions: list[ShapeDimension] = []
        bounds = dict(dynamic_bounds or {})
        if isinstance(type, TensorType):
            for index, size in enumerate(type.shape):
                if size is not None:
                    dimensions.append(ShapeDimension.fixed(size))
                    continue
                if index not in bounds:
                    raise ValueError(
                        f"value {name}: missing bounds for dynamic dimension {index}"
                    )
                symbol, minimum, optimum, maximum = bounds[index]
                dimensions.append(
                    ShapeDimension.bounded(symbol, minimum, optimum, maximum)
                )
        unknown = sorted(set(bounds) - set(range(len(type.shape)))) if isinstance(
            type, TensorType
        ) else sorted(bounds)
        if unknown:
            raise ValueError(f"value {name}: bounds reference unknown dimensions {unknown}")
        return cls(name, type, device, tuple(dimensions), alignment)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": self.type.to_dict(),
            "device": self.device,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "alignment": self.alignment,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValueContract":
        return cls(
            name=str(data["name"]),
            type=type_from_dict(data["type"]),
            device=str(data["device"]),
            dimensions=tuple(
                ShapeDimension.from_dict(item)
                for item in data.get("dimensions", ())
            ),
            alignment=int(data.get("alignment", 1)),
        )


@dataclass(frozen=True, slots=True)
class WorkspaceContract:
    size_bytes: int = 0
    alignment: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError("workspace size must be non-negative")
        _require_power_of_two(self.alignment, "workspace alignment")
        _require_nonempty(self.device, "workspace device")

    def to_dict(self) -> dict[str, object]:
        return {
            "size_bytes": self.size_bytes,
            "alignment": self.alignment,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkspaceContract":
        return cls(
            size_bytes=int(data.get("size_bytes", 0)),
            alignment=int(data.get("alignment", 1)),
            device=str(data.get("device", "cpu")),
        )


@dataclass(frozen=True, slots=True)
class BackendCapability:
    backend: str
    target: str
    supported_dtypes: tuple[str, ...]
    supports_dynamic_shapes: bool = False
    supports_device_resident_io: bool = False
    requires_synchronize: bool = False

    def __post_init__(self) -> None:
        _require_nonempty(self.backend, "backend")
        _require_nonempty(self.target, "backend target")
        if not self.supported_dtypes:
            raise ValueError("backend capability requires at least one dtype")
        if any(not item for item in self.supported_dtypes):
            raise ValueError("backend capability contains an empty dtype")

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "target": self.target,
            "supported_dtypes": list(self.supported_dtypes),
            "supports_dynamic_shapes": self.supports_dynamic_shapes,
            "supports_device_resident_io": self.supports_device_resident_io,
            "requires_synchronize": self.requires_synchronize,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BackendCapability":
        return cls(
            backend=str(data["backend"]),
            target=str(data["target"]),
            supported_dtypes=tuple(str(item) for item in data["supported_dtypes"]),
            supports_dynamic_shapes=bool(
                data.get("supports_dynamic_shapes", False)
            ),
            supports_device_resident_io=bool(
                data.get("supports_device_resident_io", False)
            ),
            requires_synchronize=bool(data.get("requires_synchronize", False)),
        )


@dataclass(frozen=True, slots=True)
class EffectAudit:
    hidden_mutation: bool = False
    hidden_rng: bool = False
    external_io: bool = False
    explicit_rng: bool = False
    lifted_states: tuple[str, ...] = ()
    diagnostics: tuple[ArtifactDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if len(self.lifted_states) != len(set(self.lifted_states)):
            raise ValueError("effect audit contains duplicate lifted state names")

    @property
    def passed(self) -> bool:
        return (
            not self.hidden_mutation
            and not self.hidden_rng
            and not self.external_io
            and all(
                item.severity is not DiagnosticSeverity.ERROR
                for item in self.diagnostics
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_mutation": self.hidden_mutation,
            "hidden_rng": self.hidden_rng,
            "external_io": self.external_io,
            "explicit_rng": self.explicit_rng,
            "lifted_states": list(self.lifted_states),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EffectAudit":
        audit = cls(
            hidden_mutation=bool(data.get("hidden_mutation", False)),
            hidden_rng=bool(data.get("hidden_rng", False)),
            external_io=bool(data.get("external_io", False)),
            explicit_rng=bool(data.get("explicit_rng", False)),
            lifted_states=tuple(str(item) for item in data.get("lifted_states", ())),
            diagnostics=tuple(
                ArtifactDiagnostic.from_dict(item)
                for item in data.get("diagnostics", ())
            ),
        )
        if "passed" in data and bool(data["passed"]) != audit.passed:
            raise ValueError("serialized effect audit has inconsistent passed flag")
        return audit


@dataclass(frozen=True, slots=True)
class RegionArtifactContract:
    region_id: int
    region_name: str
    inputs: tuple[ValueContract, ...]
    outputs: tuple[ValueContract, ...]
    artifact_kind: ArtifactKind
    artifact_path: str
    artifact_sha256: str
    artifact_size_bytes: int
    workspace: WorkspaceContract
    capability: BackendCapability
    effect_audit: EffectAudit
    plugin_abi: str = REGION_PLUGIN_ABI
    callable_abi_version: int = CALLABLE_ABI_VERSION
    schema: str = ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported artifact schema: {self.schema!r}")
        if self.callable_abi_version != CALLABLE_ABI_VERSION:
            raise ValueError(
                f"unsupported callable ABI {self.callable_abi_version}; "
                f"expected {CALLABLE_ABI_VERSION}"
            )
        if self.plugin_abi != REGION_PLUGIN_ABI:
            raise ValueError(
                f"unsupported Region plugin ABI {self.plugin_abi!r}; "
                f"expected {REGION_PLUGIN_ABI!r}"
            )
        if self.region_id < 0:
            raise ValueError("region id must be non-negative")
        _require_nonempty(self.region_name, "region name")
        _validate_relative_path(self.artifact_path)
        _validate_sha256(self.artifact_sha256)
        if self.artifact_size_bytes < 0:
            raise ValueError("artifact size must be non-negative")
        _require_unique_names(self.inputs, "input")
        _require_unique_names(self.outputs, "output")
        if not self.effect_audit.passed:
            raise ValueError(
                f"region {self.region_name}: effect audit did not pass"
            )
        dtypes = set(self.capability.supported_dtypes)
        used_dtypes = {
            value.type.dtype
            for value in self.inputs + self.outputs
            if isinstance(value.type, TensorType)
        }
        unsupported = sorted(used_dtypes - dtypes)
        if unsupported:
            raise ValueError(
                f"region {self.region_name}: backend does not support dtypes "
                f"{unsupported}"
            )
        has_dynamic = any(
            dimension.symbol is not None
            for value in self.inputs + self.outputs
            for dimension in value.dimensions
        )
        if has_dynamic and not self.capability.supports_dynamic_shapes:
            raise ValueError(
                f"region {self.region_name}: dynamic shape profile requires "
                "backend dynamic-shape capability"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "plugin_abi": self.plugin_abi,
            "callable_abi_version": self.callable_abi_version,
            "region_id": self.region_id,
            "region_name": self.region_name,
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "artifact_kind": self.artifact_kind.value,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "workspace": self.workspace.to_dict(),
            "capability": self.capability.to_dict(),
            "effect_audit": self.effect_audit.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegionArtifactContract":
        return cls(
            schema=str(data["schema"]),
            plugin_abi=str(data["plugin_abi"]),
            callable_abi_version=int(data["callable_abi_version"]),
            region_id=int(data["region_id"]),
            region_name=str(data["region_name"]),
            inputs=tuple(
                ValueContract.from_dict(item) for item in data.get("inputs", ())
            ),
            outputs=tuple(
                ValueContract.from_dict(item) for item in data.get("outputs", ())
            ),
            artifact_kind=ArtifactKind(str(data["artifact_kind"])),
            artifact_path=str(data["artifact_path"]),
            artifact_sha256=str(data["artifact_sha256"]),
            artifact_size_bytes=int(data["artifact_size_bytes"]),
            workspace=WorkspaceContract.from_dict(data["workspace"]),
            capability=BackendCapability.from_dict(data["capability"]),
            effect_audit=EffectAudit.from_dict(data["effect_audit"]),
        )


def _validate_relative_path(path: str) -> None:
    from pathlib import PurePosixPath

    _require_nonempty(path, "artifact path")
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in path:
        raise ValueError(f"artifact path must be a normalized relative path: {path!r}")
    if str(candidate) != path or path == ".":
        raise ValueError(f"artifact path must be normalized: {path!r}")


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")


def _require_unique_names(
    values: tuple[ValueContract, ...], category: str
) -> None:
    names = [item.name for item in values]
    if len(names) != len(set(names)):
        raise ValueError(f"region artifact contains duplicate {category} names")
