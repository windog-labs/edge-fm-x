"""Canonical, deterministic serialization for Invocation IR v0.2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from vlaforge.ir.attrs import Effect, Ownership
from vlaforge.ir.program import (
    Block,
    InputPort,
    Invocation,
    Module,
    Operation,
    OutputPort,
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
    if isinstance(value, Ownership | Effect):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"attribute is not serializable: {type(value).__name__}")


def _value_to_data(value: Value) -> dict[str, Any]:
    return {"name": value.name, "type": value.type.to_dict()}


def _block_to_data(block: Block) -> dict[str, Any]:
    return {
        "arguments": [_value_to_data(value) for value in block.arguments],
        "operations": [
            _operation_to_data(operation) for operation in block.operations
        ],
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
        "inputs": [
            {
                "id": port.input_id,
                "name": port.name,
                "payload": port.payload.to_dict(),
                "required": port.required,
                "default": _json_value(port.default),
                "device": port.device,
                "ownership": port.ownership.value,
                "alignment": port.alignment,
                "extension": port.extension,
                "value_range": (
                    None
                    if port.value_range is None
                    else list(port.value_range)
                ),
                "valid_for": port.valid_for,
            }
            for port in module.inputs
        ],
        "outputs": [
            {
                "id": port.output_id,
                "name": port.name,
                "group": port.group,
                "payload": port.payload.to_dict(),
                "device": port.device,
                "alignment": port.alignment,
            }
            for port in module.outputs
        ],
        "states": [
            {
                "name": state.name,
                "payload": state.payload.to_dict(),
                "retention": state.retention,
                "reset_on_episode": state.reset_on_episode,
                "ownership": state.ownership.value,
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
        "invocations": [
            {
                "name": invocation.name,
                "body": _block_to_data(invocation.body),
                "metadata": _json_value(invocation.metadata),
            }
            for invocation in module.invocations
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


def io_schema_data(module: Module) -> dict[str, object]:
    data = module_to_data(module)
    return {
        "schema": "vlaforge.io_schema/2",
        "inputs": data["inputs"],
        "outputs": data["outputs"],
    }


def io_schema_digest(module: Module) -> str:
    payload = json.dumps(
        io_schema_data(module),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _value_from_data(data: Mapping[str, Any]) -> Value:
    return Value(str(data["name"]), type_from_dict(data["type"]))


def _operation_from_data(data: Mapping[str, Any]) -> Operation:
    return Operation(
        opcode=str(data["op"]),
        results=tuple(
            _value_from_data(item) for item in data.get("results", ())
        ),
        operands=tuple(str(item) for item in data.get("operands", ())),
        attributes=dict(data.get("attributes", {})),
        regions=tuple(
            _block_from_data(item) for item in data.get("regions", ())
        ),
        location=(
            None if data.get("location") is None else str(data["location"])
        ),
    )


def _block_from_data(data: Mapping[str, Any]) -> Block:
    return Block(
        arguments=tuple(
            _value_from_data(item) for item in data.get("arguments", ())
        ),
        operations=tuple(
            _operation_from_data(item)
            for item in data.get("operations", ())
        ),
    )


def module_from_data(data: Mapping[str, Any]) -> Module:
    version = str(data["schema"])
    require_supported(version)
    return Module(
        name=str(data["module"]),
        schema_version=version,
        inputs=tuple(
            InputPort(
                name=str(item["name"]),
                payload=type_from_dict(item["payload"]),
                input_id=int(item["id"]),
                required=bool(item.get("required", True)),
                default=item.get("default"),
                device=str(item.get("device", "cpu")),
                ownership=Ownership(
                    item.get("ownership", Ownership.EXTERNAL.value)
                ),
                alignment=int(item.get("alignment", 1)),
                extension=bool(item.get("extension", False)),
                value_range=(
                    None
                    if item.get("value_range") is None
                    else tuple(item["value_range"])
                ),
                valid_for=(
                    None
                    if item.get("valid_for") is None
                    else str(item["valid_for"])
                ),
            )
            for item in data.get("inputs", ())
        ),
        outputs=tuple(
            OutputPort(
                name=str(item["name"]),
                payload=type_from_dict(item["payload"]),
                output_id=int(item["id"]),
                group=str(item.get("group", "default")),
                device=str(item.get("device", "cpu")),
                alignment=int(item.get("alignment", 1)),
            )
            for item in data.get("outputs", ())
        ),
        states=tuple(
            StateSlot(
                name=str(item["name"]),
                payload=type_from_dict(item["payload"]),
                retention=int(item.get("retention", 2)),
                reset_on_episode=bool(
                    item.get("reset_on_episode", True)
                ),
                ownership=Ownership(
                    item.get("ownership", Ownership.HOST.value)
                ),
            )
            for item in data.get("states", ())
        ),
        regions=tuple(
            TensorRegion(
                name=str(item["name"]),
                inputs=tuple(
                    _value_from_data(value)
                    for value in item.get("inputs", ())
                ),
                outputs=tuple(
                    type_from_dict(result)
                    for result in item.get("outputs", ())
                ),
                effects=tuple(
                    Effect(effect)
                    for effect in item.get("effects", ("pure",))
                ),
                metadata=dict(item.get("metadata", {})),
            )
            for item in data.get("regions", ())
        ),
        invocations=tuple(
            Invocation(
                name=str(item["name"]),
                body=_block_from_data(item["body"]),
                metadata=dict(item.get("metadata", {})),
            )
            for item in data.get("invocations", ())
        ),
        metadata=dict(data.get("metadata", {})),
    )


def parse_canonical_json(text: str) -> Module:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("serialized module must be a JSON object")
    return module_from_data(data)
