"""Verified static composition of multiple AOTI TensorRegion artifacts.

An AOTI sequence is a backend artifact, not a Semantic IR extension.  It
allows one logical TensorRegion to be physically partitioned into a bounded
dataflow of raw/package AOTI callables while keeping the model-facing Region
contract unchanged. Sequence edges use a canonical dense tensor ABI; the
runtime explicitly materializes a contiguous tensor when a physical AOTI
callable returns a padded or strided view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


AOTI_SEQUENCE_SCHEMA = "vlaforge.aoti_sequence/1"
_MAGIC = "VLAFORGE_AOTI_SEQUENCE"
_VERSION = 1
_ROLES = {"input", "output", "temporary"}
_DTYPES = {"bool", "i32", "i64", "f16", "bf16", "f32", "f64", "u64", "u8"}


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
        raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")


def _validate_token(value: str, field: str) -> None:
    if not value or any(item.isspace() for item in value):
        raise ValueError(f"{field} must be one non-empty whitespace-free token")


def _validate_relative_path(value: str) -> None:
    _validate_token(value, "artifact path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or value == "."
    ):
        raise ValueError(
            f"artifact path must be normalized and relative: {value!r}"
        )


@dataclass(frozen=True, slots=True)
class AotiSequenceValue:
    value_id: int
    name: str
    role: str
    binding: int | None
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.value_id < 0:
            raise ValueError("value id must be non-negative")
        _validate_token(self.name, "value name")
        if self.role not in _ROLES:
            raise ValueError(f"unsupported sequence value role: {self.role!r}")
        if self.dtype not in _DTYPES:
            raise ValueError(f"unsupported sequence value dtype: {self.dtype!r}")
        if any(item < 0 for item in self.shape):
            raise ValueError("sequence value dimensions must be non-negative")
        if self.role == "temporary":
            if self.binding is not None:
                raise ValueError("temporary value cannot have an external binding")
        elif self.binding is None or self.binding < 0:
            raise ValueError("external value requires a non-negative binding")

    def to_dict(self) -> dict[str, object]:
        return {
            "value_id": self.value_id,
            "name": self.name,
            "role": self.role,
            "binding": self.binding,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


@dataclass(frozen=True, slots=True)
class AotiSequenceArtifact:
    artifact_id: int
    name: str
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.artifact_id < 0:
            raise ValueError("artifact id must be non-negative")
        _validate_token(self.name, "artifact name")
        _validate_relative_path(self.path)
        _validate_sha256(self.sha256)
        if self.size_bytes < 1:
            raise ValueError("AOTI sequence artifact must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class AotiSequenceNode:
    artifact_id: int
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.artifact_id < 0:
            raise ValueError("node artifact id must be non-negative")
        if not self.inputs or not self.outputs:
            raise ValueError("AOTI sequence node requires inputs and outputs")
        if any(item < 0 for item in self.inputs + self.outputs):
            raise ValueError("node value ids must be non-negative")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError("node cannot produce the same value twice")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }


@dataclass(frozen=True, slots=True)
class AotiSequenceManifest:
    region_name: str
    target: str
    device: str
    values: tuple[AotiSequenceValue, ...]
    artifacts: tuple[AotiSequenceArtifact, ...]
    nodes: tuple[AotiSequenceNode, ...]
    schema: str = AOTI_SEQUENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != AOTI_SEQUENCE_SCHEMA:
            raise ValueError(f"unsupported AOTI sequence schema: {self.schema!r}")
        _validate_token(self.region_name, "Region name")
        _validate_token(self.target, "target")
        _validate_token(self.device, "device")
        if self.device != "cpu" and not (
            self.device.startswith("cuda:")
            and self.device[5:].isdigit()
        ):
            raise ValueError("sequence device must be cpu or explicit cuda:N")
        if not self.values or not self.artifacts or not self.nodes:
            raise ValueError("AOTI sequence requires values, artifacts, and nodes")

        value_ids = [item.value_id for item in self.values]
        artifact_ids = [item.artifact_id for item in self.artifacts]
        if value_ids != list(range(len(self.values))):
            raise ValueError("sequence value ids must be dense and ordered")
        if artifact_ids != list(range(len(self.artifacts))):
            raise ValueError("sequence artifact ids must be dense and ordered")
        if len({item.name for item in self.values}) != len(self.values):
            raise ValueError("sequence value names must be unique")
        if len({item.name for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("sequence artifact names must be unique")
        if len({item.path for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("sequence artifact paths must be unique")

        for role in ("input", "output"):
            bindings = sorted(
                int(item.binding)
                for item in self.values
                if item.role == role and item.binding is not None
            )
            if bindings != list(range(len(bindings))):
                raise ValueError(
                    f"sequence {role} bindings must be dense and unique"
                )

        value_by_id = {item.value_id: item for item in self.values}
        defined = {
            item.value_id for item in self.values if item.role == "input"
        }
        used_artifacts: set[int] = set()
        for index, node in enumerate(self.nodes):
            if node.artifact_id >= len(self.artifacts):
                raise ValueError(
                    f"sequence node {index} references unknown artifact"
                )
            used_artifacts.add(node.artifact_id)
            unknown_inputs = [item for item in node.inputs if item not in defined]
            if unknown_inputs:
                raise ValueError(
                    f"sequence node {index} reads undefined values "
                    f"{unknown_inputs}"
                )
            for value_id in node.outputs:
                value = value_by_id.get(value_id)
                if value is None:
                    raise ValueError(
                        f"sequence node {index} produces unknown value {value_id}"
                    )
                if value.role == "input" or value_id in defined:
                    raise ValueError(
                        f"sequence node {index} redefines value {value_id}"
                    )
                defined.add(value_id)
        if used_artifacts != set(range(len(self.artifacts))):
            raise ValueError("every sequence artifact must be used")
        missing_outputs = [
            item.value_id
            for item in self.values
            if item.role == "output" and item.value_id not in defined
        ]
        if missing_outputs:
            raise ValueError(
                f"sequence does not produce external outputs {missing_outputs}"
            )
        used_values = {
            value_id for node in self.nodes for value_id in node.inputs
        }
        unused_temporaries = [
            item.value_id
            for item in self.values
            if item.role == "temporary"
            and (
                item.value_id not in defined
                or item.value_id not in used_values
            )
        ]
        if unused_temporaries:
            raise ValueError(
                f"sequence contains unused temporary values "
                f"{unused_temporaries}"
            )

    @property
    def input_count(self) -> int:
        return sum(item.role == "input" for item in self.values)

    @property
    def output_count(self) -> int:
        return sum(item.role == "output" for item in self.values)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "region_name": self.region_name,
            "target": self.target,
            "device": self.device,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "values": [item.to_dict() for item in self.values],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "nodes": [item.to_dict() for item in self.nodes],
        }

    def canonical_text(self) -> str:
        lines = [
            f"{_MAGIC} {_VERSION}",
            f"region {self.region_name}",
            f"target {self.target}",
            f"device {self.device}",
            f"values {len(self.values)}",
        ]
        for item in self.values:
            binding = -1 if item.binding is None else item.binding
            dimensions = " ".join(str(value) for value in item.shape)
            suffix = f" {dimensions}" if dimensions else ""
            lines.append(
                f"value {item.value_id} {item.role} {binding} "
                f"{item.dtype} {len(item.shape)}{suffix}"
            )
        lines.append(f"artifacts {len(self.artifacts)}")
        for item in self.artifacts:
            lines.append(
                f"artifact {item.artifact_id} {item.path} "
                f"{item.sha256} {item.size_bytes}"
            )
        lines.append(f"nodes {len(self.nodes)}")
        for item in self.nodes:
            inputs = " ".join(str(value) for value in item.inputs)
            outputs = " ".join(str(value) for value in item.outputs)
            lines.append(
                f"node {item.artifact_id} {len(item.inputs)} {inputs} "
                f"{len(item.outputs)} {outputs}"
            )
        lines.append("end")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_text(cls, text: str) -> "AotiSequenceManifest":
        tokens = text.split()
        cursor = 0

        def take(expected: str | None = None) -> str:
            nonlocal cursor
            if cursor >= len(tokens):
                raise ValueError("truncated AOTI sequence manifest")
            value = tokens[cursor]
            cursor += 1
            if expected is not None and value != expected:
                raise ValueError(
                    f"expected AOTI sequence token {expected!r}, got {value!r}"
                )
            return value

        take(_MAGIC)
        if int(take()) != _VERSION:
            raise ValueError("unsupported AOTI sequence text version")
        take("region")
        region_name = take()
        take("target")
        target = take()
        take("device")
        device = take()
        take("values")
        values = []
        for _ in range(int(take())):
            take("value")
            value_id = int(take())
            role = take()
            raw_binding = int(take())
            dtype = take()
            rank = int(take())
            shape = tuple(int(take()) for _ in range(rank))
            values.append(
                AotiSequenceValue(
                    value_id,
                    f"value_{value_id}",
                    role,
                    None if raw_binding < 0 else raw_binding,
                    dtype,
                    shape,
                )
            )
        take("artifacts")
        artifacts = []
        for _ in range(int(take())):
            take("artifact")
            artifact_id = int(take())
            artifacts.append(
                AotiSequenceArtifact(
                    artifact_id,
                    f"artifact_{artifact_id}",
                    take(),
                    take(),
                    int(take()),
                )
            )
        take("nodes")
        nodes = []
        for _ in range(int(take())):
            take("node")
            artifact_id = int(take())
            inputs = tuple(int(take()) for _ in range(int(take())))
            outputs = tuple(int(take()) for _ in range(int(take())))
            nodes.append(AotiSequenceNode(artifact_id, inputs, outputs))
        take("end")
        if cursor != len(tokens):
            raise ValueError("AOTI sequence manifest has trailing tokens")
        return cls(
            region_name,
            target,
            device,
            tuple(values),
            tuple(artifacts),
            tuple(nodes),
        )
