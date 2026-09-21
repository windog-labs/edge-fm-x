"""Backend-neutral numerical requirements and compile provenance, not fidelity.

These immutable records preserve observed/configured evidence. Runtime checking
and output validation are separate capabilities and are not implemented here.
No Torch import, global-state setter or model-specific vocabulary belongs here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

POLICY_SCHEMA = "vlaforge.numerical_policy/1"
COMPILE_SCHEMA = "vlaforge.numerical_compile_record/1"
REQUIREMENT_SCHEMA = "vlaforge.numerical_requirement/1"
BINDING_SCHEMA = "vlaforge.region_numerical_binding/1"
PROVIDER_BINDING_SCHEMA = "vlaforge.region_numerical_binding/2"
RUNTIME_ENFORCEMENT = "unimplemented"
PROVIDER_REQUIRED = "provider-required/1"


class NumericalContractError(ValueError):
    """Malformed, inconsistent or downgraded numerical provenance."""


class NumericalEnforcementUnavailable(NotImplementedError):
    """A valid requirement cannot yet be enforced by this deployment runtime."""


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise NumericalContractError(f"{name} must be a nonempty string")


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise NumericalContractError(f"{name} must be lowercase SHA-256")


def strict_json(text: str) -> Any:
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise NumericalContractError(f"duplicate JSON key: {name}")
            result[name] = value
        return result

    def invalid(value):
        raise NumericalContractError(f"non-finite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def canonical_json(value: object) -> str:
    def validate(item):
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise NumericalContractError("JSON object keys must be strings")
            for entry in item.values():
                validate(entry)
        elif type(item) is list:
            for entry in item:
                validate(entry)
        elif type(item) is float:
            if not math.isfinite(item):
                raise NumericalContractError("JSON numbers must be finite")
        elif item is not None and type(item) not in (str, int, bool):
            raise NumericalContractError("value is not strict JSON data")

    validate(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NumericalContractError(f"invalid canonical JSON: {exc}") from exc


def _fields(value: Mapping[str, Any], schema: str, names: tuple[str, ...]) -> None:
    if not isinstance(value, Mapping):
        raise NumericalContractError("numerical record must be an object")
    expected = {"schema", *names}
    if set(value) != expected:
        raise NumericalContractError(
            f"numerical fields differ: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )
    if value["schema"] != schema:
        raise NumericalContractError(f"unsupported numerical schema: {value['schema']}")


class _CanonicalRecord:
    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_json(cls, text: str):
        return cls.from_dict(strict_json(text))


@dataclass(frozen=True, slots=True)
class NumericalPolicy(_CanonicalRecord):
    namespace: str
    values: tuple[tuple[str, bool | int | float | str], ...]

    def __post_init__(self) -> None:
        _text(self.namespace, "policy namespace")
        if re.fullmatch(r"[A-Za-z0-9_.-]+/[1-9][0-9]*", self.namespace) is None:
            raise NumericalContractError(
                "policy namespace must have an explicit version"
            )
        if type(self.values) is not tuple or not self.values:
            raise NumericalContractError("policy values must be a nonempty tuple")
        names = []
        for pair in self.values:
            if type(pair) is not tuple or len(pair) != 2:
                raise NumericalContractError("policy fields must be immutable pairs")
            name, value = pair
            _text(name, "policy field")
            names.append(name)
            if type(value) not in (bool, int, float, str):
                raise NumericalContractError(
                    f"policy field {name} must be a typed scalar"
                )
            if type(value) is float and not math.isfinite(value):
                raise NumericalContractError(f"policy field {name} must be finite")
        if names != sorted(set(names)):
            raise NumericalContractError("policy fields must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": POLICY_SCHEMA,
            "namespace": self.namespace,
            "values": dict(self.values),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NumericalPolicy:
        _fields(value, POLICY_SCHEMA, ("namespace", "values"))
        if not isinstance(value["values"], Mapping):
            raise NumericalContractError("policy values must be an object")
        if any(not isinstance(name, str) for name in value["values"]):
            raise NumericalContractError("policy field names must be strings")
        return cls(value["namespace"], tuple(sorted(value["values"].items())))


@dataclass(frozen=True, slots=True)
class NumericalCompileRecord(_CanonicalRecord):
    backend: str
    target: str
    compiler_version: str
    exported_program_sha256: str
    graph_sha256: str
    artifact_sha256: str
    artifact_size_bytes: int
    reference_policy: NumericalPolicy
    requested_compile_policy: NumericalPolicy
    observed_compile_policy: NumericalPolicy
    configuration_json: str

    def __post_init__(self) -> None:
        for name in ("backend", "target", "compiler_version"):
            _text(getattr(self, name), name)
        for name in ("exported_program_sha256", "graph_sha256", "artifact_sha256"):
            _sha(getattr(self, name), name)
        if type(self.artifact_size_bytes) is not int or self.artifact_size_bytes < 0:
            raise NumericalContractError("artifact size must be a nonnegative integer")
        for name in (
            "reference_policy",
            "requested_compile_policy",
            "observed_compile_policy",
        ):
            if not isinstance(getattr(self, name), NumericalPolicy):
                raise NumericalContractError(f"{name} must be a NumericalPolicy")
        if (
            self.requested_compile_policy.digest()
            != self.observed_compile_policy.digest()
        ):
            raise NumericalContractError(
                "requested and observed compilation policies differ"
            )
        if not isinstance(self.configuration_json, str):
            raise NumericalContractError("configuration_json must be canonical text")
        config = strict_json(self.configuration_json)
        if not isinstance(config, dict) or not config:
            raise NumericalContractError(
                "compile configuration must be a nonempty object"
            )
        if canonical_json(config) != self.configuration_json:
            raise NumericalContractError("compile configuration is not canonical JSON")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": COMPILE_SCHEMA,
            **{
                name: getattr(self, name)
                for name in (
                    "backend",
                    "target",
                    "compiler_version",
                    "exported_program_sha256",
                    "graph_sha256",
                    "artifact_sha256",
                    "artifact_size_bytes",
                )
            },
            "reference_policy": self.reference_policy.to_dict(),
            "requested_compile_policy": self.requested_compile_policy.to_dict(),
            "observed_compile_policy": self.observed_compile_policy.to_dict(),
            "configuration": strict_json(self.configuration_json),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NumericalCompileRecord:
        names = (
            "backend",
            "target",
            "compiler_version",
            "exported_program_sha256",
            "graph_sha256",
            "artifact_sha256",
            "artifact_size_bytes",
        )
        _fields(
            value,
            COMPILE_SCHEMA,
            (
                *names,
                "reference_policy",
                "requested_compile_policy",
                "observed_compile_policy",
                "configuration",
            ),
        )
        return cls(
            **{name: value[name] for name in names},
            **{
                name: NumericalPolicy.from_dict(value[name])
                for name in (
                    "reference_policy",
                    "requested_compile_policy",
                    "observed_compile_policy",
                )
            },
            configuration_json=canonical_json(value["configuration"]),
        )


@dataclass(frozen=True, slots=True)
class NumericalRequirement(_CanonicalRecord):
    policy: NumericalPolicy
    execution_lane: str
    reference_context_sha256: str
    compile_record_sha256: str
    enforcement: str = "require-current"

    def __post_init__(self) -> None:
        if not isinstance(self.policy, NumericalPolicy):
            raise NumericalContractError("requirement policy must be a NumericalPolicy")
        if self.execution_lane not in ("same-precision", "quantized"):
            raise NumericalContractError("unknown numerical execution lane")
        if self.enforcement != "require-current":
            raise NumericalContractError(
                "only require-current enforcement is specified"
            )
        _sha(self.reference_context_sha256, "reference context")
        _sha(self.compile_record_sha256, "compile record")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REQUIREMENT_SCHEMA,
            "policy": self.policy.to_dict(),
            "policy_sha256": self.policy.digest(),
            "execution_lane": self.execution_lane,
            "reference_context_sha256": self.reference_context_sha256,
            "compile_record_sha256": self.compile_record_sha256,
            "enforcement": self.enforcement,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NumericalRequirement:
        _fields(
            value,
            REQUIREMENT_SCHEMA,
            (
                "policy",
                "policy_sha256",
                "execution_lane",
                "reference_context_sha256",
                "compile_record_sha256",
                "enforcement",
            ),
        )
        result = cls(
            NumericalPolicy.from_dict(value["policy"]),
            value["execution_lane"],
            value["reference_context_sha256"],
            value["compile_record_sha256"],
            value["enforcement"],
        )
        if value["policy_sha256"] != result.policy.digest():
            raise NumericalContractError("numerical policy digest mismatch")
        return result


@dataclass(frozen=True, slots=True)
class RegionNumericalBinding(_CanonicalRecord):
    region_name: str
    requirement: NumericalRequirement
    compile_record: NumericalCompileRecord
    runtime_enforcement: str = RUNTIME_ENFORCEMENT

    def __post_init__(self) -> None:
        if self.runtime_enforcement not in (RUNTIME_ENFORCEMENT, PROVIDER_REQUIRED):
            raise NumericalContractError("unsupported numerical runtime enforcement")
        _text(self.region_name, "numerical Region name")
        if not isinstance(self.requirement, NumericalRequirement):
            raise NumericalContractError("invalid numerical requirement")
        if not isinstance(self.compile_record, NumericalCompileRecord):
            raise NumericalContractError("invalid numerical compile record")
        if self.requirement.compile_record_sha256 != self.compile_record.digest():
            raise NumericalContractError("numerical compile record digest mismatch")
        if (
            self.requirement.reference_context_sha256
            != self.compile_record.reference_policy.digest()
        ):
            raise NumericalContractError("numerical reference context digest mismatch")
        # A non-identity runtime projection needs its own future checked contract.
        if (
            self.requirement.policy.digest()
            != self.compile_record.observed_compile_policy.digest()
        ):
            raise NumericalContractError("runtime policy projection is unsupported")

    def require_runtime_deployable(self) -> None:
        """Check codegen eligibility, not actual runtime provider acceptance."""
        if self.runtime_enforcement != PROVIDER_REQUIRED:
            raise NumericalEnforcementUnavailable(
                f"Region {self.region_name}: numerical runtime enforcement is unimplemented"
            )
        if self.compile_record.backend in ("aoti", "torchscript"):
            from vlaforge.deployment.libtorch_numerical import require_libtorch_policy

            require_libtorch_policy(self.requirement.policy)
        elif self.compile_record.backend != "shared_plugin":
            raise NumericalEnforcementUnavailable(
                f"Region {self.region_name}: numerical provider for "
                f"{self.compile_record.backend} is unsupported"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": (
                PROVIDER_BINDING_SCHEMA
                if self.runtime_enforcement == PROVIDER_REQUIRED
                else BINDING_SCHEMA
            ),
            "region_name": self.region_name,
            "requirement": self.requirement.to_dict(),
            "compile_record": self.compile_record.to_dict(),
            "runtime_enforcement": self.runtime_enforcement,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RegionNumericalBinding:
        _fields(
            value,
            (
                PROVIDER_BINDING_SCHEMA
                if isinstance(value, Mapping)
                and value.get("schema") == PROVIDER_BINDING_SCHEMA
                else BINDING_SCHEMA
            ),
            ("region_name", "requirement", "compile_record", "runtime_enforcement"),
        )
        expected = (
            PROVIDER_REQUIRED
            if value["schema"] == PROVIDER_BINDING_SCHEMA
            else RUNTIME_ENFORCEMENT
        )
        if value["runtime_enforcement"] != expected:
            raise NumericalContractError(
                f"numerical runtime enforcement for this binding version is {expected}"
            )
        return cls(
            value["region_name"],
            NumericalRequirement.from_dict(value["requirement"]),
            NumericalCompileRecord.from_dict(value["compile_record"]),
            value["runtime_enforcement"],
        )


def numerical_enforcement(bindings) -> str:
    modes = {item.runtime_enforcement for item in bindings}
    if len(modes) != 1:
        raise NumericalContractError("mixed or empty numerical enforcement modes")
    return next(iter(modes))


def validate_numerical_version(
    data: Mapping[str, Any],
    *,
    legacy_schema: str,
    policy_schema: str,
    field: str,
    legacy_fields: set[str],
) -> None:
    """Do not let old document versions silently discard required policy fields."""
    if data.get("schema") == legacy_schema:
        if field in data or "runtime_enforcement" in data:
            raise NumericalContractError("numerical policy cannot use legacy schema")
    elif data.get("schema") == policy_schema:
        expected = legacy_fields | {field, "runtime_enforcement"}
        if set(data) != expected:
            raise NumericalContractError("policy-bearing document fields differ")
        if data["runtime_enforcement"] not in (RUNTIME_ENFORCEMENT, PROVIDER_REQUIRED):
            raise NumericalContractError("unsupported numerical runtime enforcement")
