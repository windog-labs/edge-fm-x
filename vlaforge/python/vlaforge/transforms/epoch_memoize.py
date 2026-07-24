"""Synthesize cache signatures from logical epoch and state-version values."""

from __future__ import annotations

from dataclasses import dataclass, replace

from vlaforge.analysis.verifier import verify
from vlaforge.ir.program import Block, Module, Operation, Value
from vlaforge.ir.types import EpochType, SnapshotType


class MemoizationSynthesisError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CacheDependency:
    kind: str
    value: str
    subject: str
    max_age_ns: int | None = None
    max_versions: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "value": self.value,
            "subject": self.subject,
            "max_age_ns": self.max_age_ns,
            "max_versions": self.max_versions,
        }


@dataclass(frozen=True, slots=True)
class MemoizationLegality:
    legal: bool
    dependencies: tuple[CacheDependency, ...]
    reason: str


def memoization_legality(
    operation: Operation,
    *,
    definitions: dict[str, Value],
    provenance: dict[str, tuple[CacheDependency, ...]],
) -> MemoizationLegality:
    """Prove that an invoke has a complete temporal dependency signature."""

    dependencies = []
    unversioned = []
    for operand_name in operation.operands:
        operand_dependencies = provenance.get(operand_name, ())
        if not operand_dependencies:
            value = definitions.get(operand_name)
            if value is not None and isinstance(value.type, EpochType):
                operand_dependencies = (
                    CacheDependency(
                        "epoch",
                        operand_name,
                        value.type.clock,
                    ),
                )
            elif value is not None and isinstance(value.type, SnapshotType):
                operand_dependencies = (
                    CacheDependency(
                        "state_version",
                        operand_name,
                        value.type.state,
                    ),
                )
        if not operand_dependencies:
            unversioned.append(operand_name)
        dependencies.extend(operand_dependencies)
    unique = tuple(
        {
            (
                item.kind,
                item.value,
                item.subject,
                item.max_age_ns,
                item.max_versions,
            ): item
            for item in dependencies
        }.values()
    )
    if unversioned:
        return MemoizationLegality(
            False,
            unique,
            "unversioned operands: " + ", ".join(unversioned),
        )
    if not unique:
        return MemoizationLegality(
            False,
            (),
            "invoke has no epoch or state-version dependency",
        )
    return MemoizationLegality(
        True,
        unique,
        "all transitive inputs are guarded by epoch or state version",
    )


def _transform_block(
    block: Block,
    *,
    module: Module,
    inherited: dict[str, Value],
    inherited_provenance: dict[
        str, tuple[CacheDependency, ...]
    ] | None = None,
) -> Block:
    definitions = dict(inherited)
    provenance = dict(inherited_provenance or {})
    operations = []
    region_map = {region.name: region for region in module.regions}
    input_map = {stream.name: stream for stream in module.inputs}
    state_map = {state.name: state for state in module.states}

    for operation in block.operations:
        transformed_regions = tuple(
            _transform_block(
                region,
                module=module,
                inherited=definitions,
                inherited_provenance=provenance,
            )
            for region in operation.regions
        )
        transformed = replace(operation, regions=transformed_regions)
        if operation.opcode == "vla.sample_input" and len(operation.results) == 2:
            stream = input_map[str(operation.attributes["stream"])]
            freshness = stream.freshness
            dependency = CacheDependency(
                "epoch",
                operation.results[1].name,
                stream.name,
                max_age_ns=(
                    None
                    if freshness is None
                    else freshness.max_age_ns
                ),
                max_versions=(
                    None
                    if freshness is None
                    else freshness.max_versions
                ),
            )
            provenance[operation.results[0].name] = (dependency,)
            provenance[operation.results[1].name] = (dependency,)
        elif (
            operation.opcode == "vla.state.read"
            and len(operation.results) == 1
        ):
            state = state_map[str(operation.attributes["state"])]
            freshness = state.freshness
            provenance[operation.results[0].name] = (
                CacheDependency(
                    "state_version",
                    operation.results[0].name,
                    state.name,
                    max_age_ns=(
                        None
                        if freshness is None
                        else freshness.max_age_ns
                    ),
                    max_versions=(
                        None
                        if freshness is None
                        else freshness.max_versions
                    ),
                ),
            )
        elif (
            operation.opcode == "vla.snapshot.value"
            and len(operation.results) == 1
            and operation.operands
        ):
            provenance[operation.results[0].name] = provenance.get(
                operation.operands[0],
                (),
            )

        if operation.opcode == "vla.invoke":
            region_name = str(operation.attributes["region"])
            region = region_map[region_name]
            if bool(region.metadata.get("memoize", False)):
                legality = memoization_legality(
                    operation,
                    definitions=definitions,
                    provenance=provenance,
                )
                if not legality.legal:
                    raise MemoizationSynthesisError(
                        f"program={module.name} rule=memoize.missing_epoch_or_state "
                        f"region={region_name}: cacheable invoke has no epoch/state "
                        "version in its dependency signature"
                    )
                key_values = [
                    item.value for item in legality.dependencies
                ]
                state_versions = [
                    f"{item.subject}:%{item.value}"
                    for item in legality.dependencies
                    if item.kind == "state_version"
                ]
                transformed = transformed.with_attributes(
                    memoize_key=key_values,
                    state_version_signature=state_versions,
                    memoize_semantics="epoch_state_signature",
                    memoize_dependencies=[
                        item.to_dict() for item in legality.dependencies
                    ],
                    memoize_invalidation=(
                        "epoch_or_state_version_or_episode_change"
                    ),
                )
            output_dependencies = []
            for operand_name in operation.operands:
                output_dependencies.extend(provenance.get(operand_name, ()))
            output_provenance = tuple(
                {
                    (
                        item.kind,
                        item.value,
                        item.subject,
                        item.max_age_ns,
                        item.max_versions,
                    ): item
                    for item in output_dependencies
                }.values()
            )
            for result in operation.results:
                provenance[result.name] = output_provenance
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
                inherited_provenance={},
            ),
        )
        for policy in module.policies
    )
    transformed = replace(module, policies=policies)
    verify(transformed)
    return transformed
