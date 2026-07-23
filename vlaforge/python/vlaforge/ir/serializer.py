"""Canonical, deterministic serialization for VLAForge IR."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from vlaforge.ir.attrs import (
    CheckpointPolicy,
    ConsistencyPolicy,
    Effect,
    FreshnessConstraint,
    Ownership,
    ResetPolicy,
    StateScope,
)
from vlaforge.ir.program import (
    Block,
    ClockDomain,
    InputStream,
    Module,
    Operation,
    Policy,
    StateSlot,
    TensorRegion,
    Value,
)
from vlaforge.ir.types import IRType, type_from_dict
from vlaforge.ir.versioning import require_supported


def _json_value(value: Any) -> Any:
    if isinstance(value, IRType):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(
        value,
        StateScope
        | ConsistencyPolicy
        | ResetPolicy
        | Ownership
        | CheckpointPolicy
        | Effect,
    ):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"attribute is not serializable: {type(value).__name__}")


def _value_to_data(value: Value) -> dict[str, Any]:
    return {"name": value.name, "type": value.type.to_dict()}


def _block_to_data(block: Block) -> dict[str, Any]:
    return {
        "arguments": [_value_to_data(value) for value in block.arguments],
        "operations": [_operation_to_data(operation) for operation in block.operations],
    }


def _operation_to_data(operation: Operation) -> dict[str, Any]:
    data: dict[str, Any] = {
        "op": operation.opcode,
        "results": [_value_to_data(value) for value in operation.results],
        "operands": list(operation.operands),
        "attributes": _json_value(operation.attributes),
        "regions": [_block_to_data(block) for block in operation.regions],
    }
    if operation.location is not None:
        data["location"] = operation.location
    return data


def module_to_data(module: Module) -> dict[str, Any]:
    return {
        "schema": module.schema_version,
        "module": module.name,
        "clocks": [
            {
                "name": clock.name,
                "period_ns": clock.period_ns,
                "deadline_ns": clock.deadline_ns,
                "jitter_ns": clock.jitter_ns,
            }
            for clock in module.clocks
        ],
        "inputs": [
            {
                "name": stream.name,
                "payload": stream.payload.to_dict(),
                "clock": stream.clock,
                "freshness": (
                    None if stream.freshness is None else stream.freshness.to_dict()
                ),
            }
            for stream in module.inputs
        ],
        "states": [
            {
                "name": state.name,
                "payload": state.payload.to_dict(),
                "scope": state.scope.value,
                "version_clock": state.version_clock,
                "retention": state.retention,
                "consistency": state.consistency.value,
                "initializer": state.initializer,
                "reset": state.reset.value,
                "authoritative": state.authoritative,
                "freshness": (
                    None if state.freshness is None else state.freshness.to_dict()
                ),
                "ownership": state.ownership.value,
                "checkpoint": state.checkpoint.value,
            }
            for state in module.states
        ],
        "regions": [
            {
                "name": region.name,
                "inputs": [_value_to_data(value) for value in region.inputs],
                "outputs": [result.to_dict() for result in region.outputs],
                "effects": [effect.value for effect in region.effects],
                "metadata": _json_value(region.metadata),
            }
            for region in module.regions
        ],
        "policies": [
            {
                "name": policy.name,
                "clock": policy.clock,
                "inputs": [_value_to_data(value) for value in policy.inputs],
                "body": _block_to_data(policy.body),
                "metadata": _json_value(policy.metadata),
            }
            for policy in module.policies
        ],
        "metadata": _json_value(module.metadata),
    }


def canonical_json(module: Module, *, indent: int | None = None) -> str:
    separators = (",", ":") if indent is None else None
    return json.dumps(
        module_to_data(module),
        sort_keys=True,
        separators=separators,
        indent=indent,
        ensure_ascii=False,
    )


def module_digest(module: Module) -> str:
    return hashlib.sha256(canonical_json(module).encode("utf-8")).hexdigest()


def _value_from_data(data: Mapping[str, Any]) -> Value:
    return Value(str(data["name"]), type_from_dict(data["type"]))


def _operation_from_data(data: Mapping[str, Any]) -> Operation:
    return Operation(
        opcode=str(data["op"]),
        results=tuple(_value_from_data(item) for item in data.get("results", ())),
        operands=tuple(str(item) for item in data.get("operands", ())),
        attributes=dict(data.get("attributes", {})),
        regions=tuple(_block_from_data(item) for item in data.get("regions", ())),
        location=None if data.get("location") is None else str(data["location"]),
    )


def _block_from_data(data: Mapping[str, Any]) -> Block:
    return Block(
        arguments=tuple(_value_from_data(item) for item in data.get("arguments", ())),
        operations=tuple(
            _operation_from_data(item) for item in data.get("operations", ())
        ),
    )


def module_from_data(data: Mapping[str, Any]) -> Module:
    version = str(data["schema"])
    require_supported(version)
    return Module(
        name=str(data["module"]),
        schema_version=version,
        clocks=tuple(
            ClockDomain(
                name=str(item["name"]),
                period_ns=(
                    None if item.get("period_ns") is None else int(item["period_ns"])
                ),
                deadline_ns=(
                    None
                    if item.get("deadline_ns") is None
                    else int(item["deadline_ns"])
                ),
                jitter_ns=int(item.get("jitter_ns", 0)),
            )
            for item in data.get("clocks", ())
        ),
        inputs=tuple(
            InputStream(
                name=str(item["name"]),
                payload=type_from_dict(item["payload"]),
                clock=str(item["clock"]),
                freshness=(
                    None
                    if item.get("freshness") is None
                    else FreshnessConstraint.from_dict(item["freshness"])
                ),
            )
            for item in data.get("inputs", ())
        ),
        states=tuple(
            StateSlot(
                name=str(item["name"]),
                payload=type_from_dict(item["payload"]),
                scope=StateScope(item["scope"]),
                version_clock=str(item["version_clock"]),
                retention=int(item["retention"]),
                consistency=ConsistencyPolicy(
                    item.get("consistency", ConsistencyPolicy.SNAPSHOT.value)
                ),
                initializer=(
                    None if item.get("initializer") is None else str(item["initializer"])
                ),
                reset=ResetPolicy(
                    item.get("reset", ResetPolicy.EPISODE_START.value)
                ),
                authoritative=bool(item.get("authoritative", False)),
                freshness=(
                    None
                    if item.get("freshness") is None
                    else FreshnessConstraint.from_dict(item["freshness"])
                ),
                ownership=Ownership(item.get("ownership", Ownership.HOST.value)),
                checkpoint=CheckpointPolicy(
                    item.get("checkpoint", CheckpointPolicy.ON_COMMIT.value)
                ),
            )
            for item in data.get("states", ())
        ),
        regions=tuple(
            TensorRegion(
                name=str(item["name"]),
                inputs=tuple(
                    _value_from_data(value) for value in item.get("inputs", ())
                ),
                outputs=tuple(
                    type_from_dict(result) for result in item.get("outputs", ())
                ),
                effects=tuple(Effect(effect) for effect in item.get("effects", ("pure",))),
                metadata=dict(item.get("metadata", {})),
            )
            for item in data.get("regions", ())
        ),
        policies=tuple(
            Policy(
                name=str(item["name"]),
                clock=str(item["clock"]),
                inputs=tuple(
                    _value_from_data(value) for value in item.get("inputs", ())
                ),
                body=_block_from_data(item["body"]),
                metadata=dict(item.get("metadata", {})),
            )
            for item in data.get("policies", ())
        ),
        metadata=dict(data.get("metadata", {})),
    )


def parse_canonical_json(text: str) -> Module:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("serialized module must be a JSON object")
    return module_from_data(data)

