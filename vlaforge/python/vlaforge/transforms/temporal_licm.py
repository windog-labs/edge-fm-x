"""Freshness- and version-proven loop-invariant region motion."""

from __future__ import annotations

from dataclasses import dataclass, replace

from vlaforge.analysis.verifier import verify
from vlaforge.ir.program import Block, Module, Operation, Policy, Value


@dataclass(frozen=True, slots=True)
class TemporalLICMDecision:
    region: str
    loop: str
    disposition: str
    reason: str


@dataclass(frozen=True, slots=True)
class TemporalLICMResult:
    module: Module
    decisions: tuple[TemporalLICMDecision, ...]

    @property
    def moved(self) -> tuple[TemporalLICMDecision, ...]:
        return tuple(
            item
            for item in self.decisions
            if item.disposition == "moved"
        )

    @property
    def prehoisted(self) -> tuple[TemporalLICMDecision, ...]:
        return tuple(
            item
            for item in self.decisions
            if item.disposition == "prehoisted"
        )


def temporal_loop_invariant_code_motion(
    module: Module,
) -> TemporalLICMResult:
    """Hoist pure TensorRegions with complete temporal cache signatures.

    The pass does not infer stability from tensor values.  A candidate must
    already carry the dependency certificate produced by
    ``synthesize_epoch_memoization`` and must be independent of the induction
    and loop-carried block arguments.
    """

    verify(module)
    decisions: list[TemporalLICMDecision] = []
    policies = tuple(
        replace(
            policy,
            body=_transform_block(
                policy.body,
                module=module,
                inherited={
                    value.name: value for value in policy.inputs
                },
                path=f"policy:{policy.name}",
                decisions=decisions,
            ),
        )
        for policy in module.policies
    )
    transformed = replace(module, policies=policies)
    verify(transformed)
    return TemporalLICMResult(transformed, tuple(decisions))


def _transform_block(
    block: Block,
    *,
    module: Module,
    inherited: dict[str, Value],
    path: str,
    decisions: list[TemporalLICMDecision],
) -> Block:
    definitions = dict(inherited)
    for argument in block.arguments:
        definitions[argument.name] = argument
    direct_result_names = {
        result.name
        for operation in block.operations
        for result in operation.results
    }
    region_map = {region.name: region for region in module.regions}
    result: list[Operation] = []
    producer_indices: dict[str, int] = {}

    for index, operation in enumerate(block.operations):
        operation_path = f"{path}/{index}:{operation.opcode}"
        transformed_regions = tuple(
            _transform_block(
                region,
                module=module,
                inherited=definitions,
                path=f"{operation_path}/region:{region_index}",
                decisions=decisions,
            )
            for region_index, region in enumerate(operation.regions)
        )
        transformed = replace(operation, regions=transformed_regions)
        if transformed.opcode == "vla.for" and len(transformed.regions) == 1:
            transformed, hoisted = _hoist_from_for(
                transformed,
                module=module,
                definitions=definitions,
                parent_result_names=direct_result_names,
                path=operation_path,
                decisions=decisions,
            )
            for candidate in hoisted:
                result.append(candidate)
                for value in candidate.results:
                    definitions[value.name] = value
                    producer_indices[value.name] = len(result) - 1

            loop_external_uses = _external_uses(
                transformed.regions[0],
                local_names={
                    item.name for item in transformed.regions[0].arguments
                },
            )
            loop_external_uses.update(transformed.operands)
            for value_name in sorted(loop_external_uses):
                producer_index = producer_indices.get(value_name)
                if producer_index is None:
                    continue
                producer = result[producer_index]
                if not _is_temporal_candidate(producer, region_map):
                    continue
                if producer.attributes.get("temporal_licm") == (
                    "moved_to_preheader"
                ):
                    continue
                if not producer.attributes.get("memoize_key"):
                    continue
                loops = list(
                    producer.attributes.get(
                        "temporal_licm_consumers",
                        [],
                    )
                )
                if operation_path not in loops:
                    loops.append(operation_path)
                producer = producer.with_attributes(
                    temporal_licm="preheader_proven",
                    temporal_licm_consumers=loops,
                )
                result[producer_index] = producer
                decisions.append(
                    TemporalLICMDecision(
                        str(producer.attributes["region"]),
                        operation_path,
                        "prehoisted",
                        "pure invoke is epoch/version-stable in loop preheader",
                    )
                )

        result.append(transformed)
        for value in transformed.results:
            definitions[value.name] = value
            producer_indices[value.name] = len(result) - 1
    return replace(block, operations=tuple(result))


def _hoist_from_for(
    loop: Operation,
    *,
    module: Module,
    definitions: dict[str, Value],
    parent_result_names: set[str],
    path: str,
    decisions: list[TemporalLICMDecision],
) -> tuple[Operation, tuple[Operation, ...]]:
    body = loop.regions[0]
    loop_arguments = {value.name for value in body.arguments}
    preheader_definitions = dict(definitions)
    region_map = {region.name: region for region in module.regions}
    hoisted: list[Operation] = []
    retained: list[Operation] = []

    for operation in body.operations:
        if not _is_temporal_candidate(operation, region_map):
            retained.append(operation)
            continue
        region_name = str(operation.attributes["region"])
        reason = _motion_rejection_reason(
            operation,
            loop_arguments=loop_arguments,
            preheader_definitions=preheader_definitions,
            parent_result_names=parent_result_names,
        )
        if reason is not None:
            decisions.append(
                TemporalLICMDecision(
                    region_name,
                    path,
                    "rejected",
                    reason,
                )
            )
            retained.append(operation)
            continue
        moved = operation.with_attributes(
            temporal_licm="moved_to_preheader",
            temporal_licm_loop=path,
        )
        hoisted.append(moved)
        for value in moved.results:
            preheader_definitions[value.name] = value
        decisions.append(
            TemporalLICMDecision(
                region_name,
                path,
                "moved",
                "pure invoke depends only on epoch/version-stable preheader values",
            )
        )

    transformed_body = replace(body, operations=tuple(retained))
    return replace(loop, regions=(transformed_body,)), tuple(hoisted)


def _is_temporal_candidate(
    operation: Operation,
    region_map: dict[str, object],
) -> bool:
    if operation.opcode != "vla.invoke":
        return False
    region = region_map[str(operation.attributes["region"])]
    return bool(region.metadata.get("loop_invariant", False))


def _motion_rejection_reason(
    operation: Operation,
    *,
    loop_arguments: set[str],
    preheader_definitions: dict[str, Value],
    parent_result_names: set[str],
) -> str | None:
    loop_dependencies = sorted(
        set(operation.operands) & loop_arguments
    )
    if loop_dependencies:
        return (
            "depends on induction or loop-carried values: "
            + ", ".join(loop_dependencies)
        )
    unavailable = sorted(
        operand
        for operand in operation.operands
        if operand not in preheader_definitions
    )
    if unavailable:
        return (
            "operand is not available in loop preheader: "
            + ", ".join(unavailable)
        )
    if not operation.attributes.get("memoize_key"):
        return "missing complete epoch/state-version dependency certificate"
    collisions = sorted(
        value.name
        for value in operation.results
        if value.name in parent_result_names
    )
    if collisions:
        return (
            "hoisted result would collide in parent scope: "
            + ", ".join(collisions)
        )
    return None


def _external_uses(
    block: Block,
    *,
    local_names: set[str],
) -> set[str]:
    defined = set(local_names)
    used = set()
    for operation in block.operations:
        for operand in operation.operands:
            if operand not in defined:
                used.add(operand)
        for region in operation.regions:
            used.update(
                _external_uses(
                    region,
                    local_names={
                        *defined,
                        *(value.name for value in region.arguments),
                    },
                )
            )
        defined.update(value.name for value in operation.results)
    return used
