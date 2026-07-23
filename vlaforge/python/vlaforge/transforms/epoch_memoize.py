"""Synthesize cache signatures from logical epoch and state-version values."""

from __future__ import annotations

from dataclasses import replace

from vlaforge.analysis.verifier import verify
from vlaforge.ir.program import Block, Module, Policy, Value
from vlaforge.ir.types import EpochType, SnapshotType


class MemoizationSynthesisError(ValueError):
    pass


def _transform_block(
    block: Block,
    *,
    module: Module,
    inherited: dict[str, Value],
) -> Block:
    definitions = dict(inherited)
    epoch_for_payload: dict[str, str] = {}
    operations = []
    region_map = {region.name: region for region in module.regions}

    for operation in block.operations:
        transformed_regions = tuple(
            _transform_block(region, module=module, inherited=definitions)
            for region in operation.regions
        )
        transformed = replace(operation, regions=transformed_regions)
        if operation.opcode == "vla.sample_input" and len(operation.results) == 2:
            epoch_for_payload[operation.results[0].name] = operation.results[1].name

        if operation.opcode == "vla.invoke":
            region_name = str(operation.attributes["region"])
            region = region_map[region_name]
            if bool(region.metadata.get("memoize", False)):
                key_values: list[str] = []
                state_versions: list[str] = []
                for operand_name in operation.operands:
                    value = definitions.get(operand_name)
                    if value is None:
                        continue
                    if isinstance(value.type, EpochType):
                        key_values.append(operand_name)
                    elif isinstance(value.type, SnapshotType):
                        key_values.append(operand_name)
                        state_versions.append(
                            f"{value.type.state}:%{operand_name}"
                        )
                    elif operand_name in epoch_for_payload:
                        key_values.append(epoch_for_payload[operand_name])
                if not key_values:
                    raise MemoizationSynthesisError(
                        f"program={module.name} rule=memoize.missing_epoch_or_state "
                        f"region={region_name}: cacheable invoke has no epoch/state "
                        "version in its dependency signature"
                    )
                transformed = transformed.with_attributes(
                    memoize_key=key_values,
                    state_version_signature=state_versions,
                    memoize_semantics="epoch_state_signature",
                )
        operations.append(transformed)
        for result in operation.results:
            definitions[result.name] = result
    return replace(block, operations=tuple(operations))


def synthesize_epoch_memoization(module: Module) -> Module:
    """Attach legal cache keys derived from logical epochs/state snapshots."""

    verify(module)
    policies = tuple(
        replace(
            policy,
            body=_transform_block(
                policy.body,
                module=module,
                inherited={value.name: value for value in policy.inputs},
            ),
        )
        for policy in module.policies
    )
    transformed = replace(module, policies=policies)
    verify(transformed)
    return transformed

