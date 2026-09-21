"""Model-neutral checked output composition, preserving incoming acceptance.

An opaque Region declaration does not prove its mathematics: implementations
must return (native, incoming_acceptance AND their own acceptance) and test it.
"""

from dataclasses import dataclass, replace

from vlaforge.analysis import verify
from vlaforge.ir import ops
from vlaforge.ir.program import Block, OutputPort
from vlaforge.ir.types import PendingOutputType, TensorType


@dataclass(frozen=True)
class OutputStageInput:
    kind: str
    name: str | None = None

    def __post_init__(self):
        if self.kind not in ("source-output", "acceptance", "module-input"):
            raise ValueError("unknown output-stage input source")
        if self.kind == "acceptance":
            if self.name is not None:
                raise ValueError("incoming acceptance has no user-selected name")
        elif not isinstance(self.name, str) or not self.name:
            raise ValueError("output-stage data source requires an explicit port name")


def attach_checked_output_stage(module, region, *, inputs, source_output, output_name):
    """Compose an explicitly conjunction-preserving Region and one output group."""
    verify(module)
    inputs = tuple(inputs)
    if len(module.invocations) != 1 or len(module.outputs) != 1:
        raise ValueError("output extension requires one invocation and one source output")
    if (len(inputs) != len(region.inputs) or any(not isinstance(item, OutputStageInput) for item in inputs)
            or sum(item.kind == "acceptance" for item in inputs) != 1
            or sum(item.kind == "source-output" for item in inputs) != 1):
        raise ValueError("output stage must explicitly consume source output and incoming acceptance")
    old_port = module.output(source_output)
    if region.name in {item.name for item in module.regions} or output_name == source_output:
        raise ValueError("output extension names must be new")
    if (not region.pure or len(region.outputs) != 2 or not isinstance(region.outputs[0], TensorType)
            or region.outputs[1] != TensorType((1,), "bool")):
        raise ValueError("output stage must be pure and return native Tensor plus boolean acceptance")
    invocation = module.invocations[0]
    operations = invocation.body.operations
    expected = ("vla.validate", "vla.output.create", "vla.output.group", "vla.txn.commit", "vla.return")
    if tuple(item.opcode for item in operations[-5:]) != expected:
        raise ValueError("output extension refuses an unknown transaction suffix")
    validate, create, group, commit, returned = operations[-5:]
    if validate.attributes.get("contract") != "vlaforge_predicate_true":
        raise ValueError("output extension cannot reinterpret an unknown acceptance validator")
    if (len(validate.operands) != 1 or create.attributes.get("output") != source_output
            or group.operands != (create.results[0].name,)
            or commit.operands[1:] != (group.results[0].name, validate.results[0].name)
            or returned.operands != (commit.results[0].name,)):
        raise ValueError("source output transaction wiring differs")
    types = {value.name: value.type for value in invocation.body.arguments}
    types.update({value.name: value.type for operation in operations[:-5] for value in operation.results})
    operands = []
    for binding, argument in zip(inputs, region.inputs, strict=True):
        if binding.kind == "acceptance":
            name = validate.operands[0]
        elif binding.kind == "source-output":
            if binding.name != source_output:
                raise ValueError("output stage must consume the retained source output")
            name = create.operands[0]
        else:
            port = module.input(binding.name)
            if port.device != old_port.device:
                raise ValueError("output-stage input and output devices differ")
            reads = [operation for operation in operations[:-5]
                     if operation.opcode == "vla.input.read" and operation.attributes.get("input") == binding.name]
            if len(reads) != 1:
                raise ValueError("output-stage input requires one dominating input read")
            name = reads[0].results[0].name
        if name not in types or types[name] != argument.type:
            raise ValueError("output-stage input type differs from its explicit source")
        operands.append(name)
    names = ("vf_output_stage_value", "vf_output_stage_accepted", "vf_output_stage_pending")

    def all_values(block):
        yield from (value.name for value in block.arguments)
        for operation in block.operations:
            yield from (value.name for value in operation.results)
            for nested in operation.regions:
                yield from all_values(nested)

    if set(names) & set(all_values(invocation.body)):
        raise ValueError("output extension SSA names collide")
    output = OutputPort(output_name, region.outputs[0], group=old_port.group, device=old_port.device, output_id=1)
    pending_type = PendingOutputType(output.name, output.payload)
    pending = ((create.results[0].name, create.results[0].type), (names[2], pending_type))
    tail = (ops.invoke(names[:2], region.outputs, region.name, tuple(operands)),
            replace(validate, operands=(names[1],)), create,
            ops.output_create(names[2], names[0], output.payload, output.name),
            ops.output_group(group.results[0].name, old_port.group, pending),
            ops.transaction_commit(commit.results[0].name, (item[1] for item in pending), old_port.group,
                commit.operands[0], group.results[0].name, validate.results[0].name), returned)
    updated = replace(invocation, body=Block(invocation.body.arguments, (*operations[:-5], *tail)))
    result = replace(module, outputs=(*module.outputs, output), regions=(*module.regions, region), invocations=(updated,),
        metadata={**module.metadata, "output_stage": {"region": region.name, "incoming_acceptance": "explicit-conjunction-required"},
                  "physical_units_verified": False, "robot_calibration_verified": False})
    verify(result)
    return result
