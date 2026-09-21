"""Python orchestration lowered to the existing Invocation IR and runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from vlaforge.analysis import verify
from vlaforge.frontend.annotations import RegionSpec
from vlaforge.frontend.builder import ModuleBuilder
from vlaforge.ir import ops
from vlaforge.ir.program import (
    Block,
    InputPort,
    Invocation,
    Module,
    OutputPort,
    StateSlot,
    Value,
)
from vlaforge.ir.types import IRType, PendingOutputType, ScalarType, TensorType


_PREDICATE = "vlaforge_predicate_true"


def _predicate_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if hasattr(value, "numel") and hasattr(value, "item"):
        return value.numel() == 1 and value.item() is True
    if isinstance(value, tuple | list) and len(value) == 1:
        return _predicate_true(value[0])
    return False


@dataclass(frozen=True, slots=True)
class SymbolicValue:
    """A typed reference, not a tensor or an executable Python value."""

    value: Value
    _owner: object
    _scope: tuple[int, ...] = ()
    _inputs: frozenset[str] = frozenset()
    _states: frozenset[str] = frozenset()
    _loops: frozenset[int] = frozenset()

    def __bool__(self) -> bool:
        raise TypeError("symbolic values cannot drive Python control flow")


@dataclass(frozen=True, slots=True)
class InvocationProgram:
    """One verified module plus reference implementations for its Regions."""

    module: Module
    regions: Mapping[str, Callable[..., object]]

    @property
    def validators(self) -> Mapping[str, Callable[[object], bool]]:
        return {_PREDICATE: _predicate_true}

    def cpp_validators(self):
        from vlaforge.codegen import CppValidatorDefinition

        return {
            _PREDICATE: CppValidatorDefinition(
                _PREDICATE,
                "return data != nullptr && size_bytes == 1u && "
                "*static_cast<const std::uint8_t*>(data) == 1u;",
            )
        }


class InvocationBuilder:
    """Lower annotated Python stage calls, bounded loops and persistent state.

    Model adapters supply pure ``@tensor_region`` functions and a boolean
    acceptance predicate. This builder owns SSA names, cache dependencies,
    state snapshots, and the atomic output/state transaction. Memory placement
    remains the responsibility of ``compile_module`` and its Plan passes.
    """

    def __init__(
        self,
        name: str,
        *,
        inputs: Iterable[InputPort],
        outputs: Iterable[OutputPort],
        states: Iterable[StateSlot] = (),
        metadata: Mapping[str, object] | None = None,
    ):
        self._module = ModuleBuilder(name)
        self._module.metadata.update(metadata or {})
        for port in inputs:
            self._module.add_input(port)
        for port in outputs:
            self._module.add_output(port)
        for state in states:
            self._module.add_state(state)
        # Validate names, shapes and stable port IDs before tracing callbacks.
        self._module.build()
        if (
            not self._module.outputs
            or len({p.group for p in self._module.outputs}) != 1
        ):
            raise ValueError("an invocation requires one nonempty atomic output group")
        self._owner = object()
        self._scope: tuple[int, ...] = ()
        self._counter = 0
        self._operations = [ops.transaction_begin("vf_transaction")]
        self._input_values: dict[str, SymbolicValue] = {}
        self._state_values: dict[str, SymbolicValue] = {}
        self._updates: dict[str, SymbolicValue] = {}
        self._regions: dict[str, Callable[..., object]] = {}
        self._finished = False
        self._failed = False

    def _check_open(self, *, root: bool = False) -> None:
        if self._finished or self._failed:
            raise ValueError("invocation is already finished or a callback failed")
        if root and self._scope:
            raise ValueError(
                "inputs and persistent state must be managed outside loops"
            )

    def _name(self, label: str) -> str:
        self._counter += 1
        return f"vf_{label}_{self._counter}"

    def _check_value(self, value: SymbolicValue) -> None:
        if not isinstance(value, SymbolicValue) or value._owner is not self._owner:
            raise ValueError("value belongs to another invocation")
        if self._scope[: len(value._scope)] != value._scope:
            raise ValueError(
                "loop-local value escaped its scope; return a carried value"
            )

    def _result(self, type_: IRType, sources: Iterable[SymbolicValue]) -> SymbolicValue:
        items = tuple(sources)
        return SymbolicValue(
            Value(self._name("value"), type_),
            self._owner,
            self._scope,
            frozenset().union(*(v._inputs for v in items)),
            frozenset().union(*(v._states for v in items)),
            frozenset().union(*(v._loops for v in items)),
        )

    def input(self, name: str) -> SymbolicValue:
        self._check_open(root=True)
        if name not in self._input_values:
            port = self._module.build().input(name)
            result = self._result(port.payload, ())
            result = replace(result, _inputs=frozenset((name,)))
            self._operations.append(
                ops.input_read(
                    result.value.name,
                    self._name("revision"),
                    port.payload,
                    name,
                )
            )
            self._input_values[name] = result
        return self._input_values[name]

    def state(self, name: str) -> SymbolicValue:
        self._check_open(root=True)
        if name not in self._state_values:
            slot = self._module.build().state(name)
            snapshot = self._name("snapshot")
            result = replace(self._result(slot.payload, ()), _states=frozenset((name,)))
            self._operations.extend(
                (
                    ops.state_read_latest(
                        snapshot, slot.payload, name, "vf_transaction"
                    ),
                    ops.snapshot_value(result.value.name, slot.payload, snapshot),
                )
            )
            self._state_values[name] = result
        return self._state_values[name]

    def update_state(self, name: str, value: SymbolicValue) -> None:
        self._check_open(root=True)
        self._check_value(value)
        if self._module.build().state(name).payload != value.value.type:
            raise ValueError(f"state {name} payload type mismatch")
        if name in self._updates:
            raise ValueError(f"state {name} already has a pending update")
        self._updates[name] = value

    def call(
        self,
        function: Callable[..., object],
        *arguments: SymbolicValue,
        cache: bool = False,
    ) -> tuple[SymbolicValue, ...]:
        self._check_open()
        spec = getattr(function, "__vlaforge_region__", None)
        if not isinstance(spec, RegionSpec):
            raise TypeError("stage must declare its Interface with @tensor_region")
        if len(arguments) != len(spec.inputs):
            raise ValueError(f"stage {spec.name} input arity mismatch")
        for value, expected in zip(arguments, spec.inputs, strict=True):
            self._check_value(value)
            if value.value.type != expected.type:
                raise ValueError(
                    f"stage {spec.name} input {expected.name} type mismatch"
                )
        region = spec.as_ir()
        if not region.pure:
            raise ValueError("stages must be pure; lift persistent state explicitly")
        results = tuple(self._result(type_, arguments) for type_ in spec.outputs)
        dependencies = self._result(ScalarType("bool"), arguments)
        if cache and dependencies._loops:
            raise ValueError("cannot memoize a loop-dependent stage by input revisions")
        if cache:
            # Cache identity is per call site; two calls may bind different inputs.
            region = replace(
                region,
                name=self._name(f"cached_{spec.name}"),
                metadata={
                    "memoize": True,
                    "cache_input_ports": sorted(dependencies._inputs),
                    "cache_state_slots": sorted(dependencies._states),
                    "loop_invariant": not self._scope,
                },
            )
        else:
            region = replace(region, metadata={"loop_invariant": not self._scope})
        if region.name in self._regions:
            declared = next(r for r in self._module.regions if r.name == region.name)
            if self._regions[region.name] is not function or (
                declared.inputs != region.inputs or declared.outputs != region.outputs
            ):
                raise ValueError(f"conflicting stage declaration: {region.name}")
        else:
            self._regions[region.name] = function
            self._module.add_region(region)
        self._operations.append(
            ops.invoke(
                (v.value.name for v in results),
                spec.outputs,
                region.name,
                (v.value.name for v in arguments),
            )
        )
        return results

    def iterate(
        self,
        initial: Iterable[SymbolicValue],
        body: Callable[..., Iterable[SymbolicValue]],
        *,
        steps: int,
        replay: str = "off",
    ) -> tuple[SymbolicValue, ...]:
        """Trace a step callback once; execute its IR ``steps`` times."""

        self._check_open()
        if replay not in {"off", "batch-only", "prefer", "required"}:
            raise ValueError("replay must be off, batch-only, prefer or required")
        values = tuple(initial)
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 1
            or not values
        ):
            raise ValueError(
                "iterate requires positive steps and nonempty carried values"
            )
        for value in values:
            self._check_value(value)
        self._counter += 1
        loop_id = self._counter
        parent_scope, parent_operations = self._scope, self._operations
        self._scope = (*parent_scope, loop_id)
        self._operations = []
        index = replace(
            self._result(ScalarType("index"), ()), _loops=frozenset((loop_id,))
        )
        carried = tuple(
            replace(
                self._result(value.value.type, (value,)),
                _loops=value._loops | {loop_id},
            )
            for value in values
        )
        try:
            yielded = tuple(body(index, *carried))
            if len(yielded) != len(values):
                raise ValueError(
                    "step callback must return the same number of carried values"
                )
            for old, new in zip(values, yielded, strict=True):
                self._check_value(new)
                if old.value.type != new.value.type:
                    raise ValueError("step callback changed a carried value type")
            operations = (
                *self._operations,
                ops.yield_values(*(v.value.name for v in yielded)),
            )
        except Exception:
            self._failed = True
            raise
        finally:
            self._scope, self._operations = parent_scope, parent_operations
        # All carried inputs can influence every result through later iterations.
        sources = (*values, *yielded)
        results = tuple(
            replace(
                self._result(v.value.type, sources),
                _loops=frozenset().union(*(x._loops for x in sources)) - {loop_id},
            )
            for v in values
        )
        self._operations.append(
            ops.for_loop_values(
                (v.value for v in results),
                (v.value.name for v in values),
                index.value,
                (v.value for v in carried),
                Block.of(operations),
                lower=0,
                upper=steps,
                replay=replay,
            )
        )
        return results

    def finish(
        self,
        outputs: Mapping[str, SymbolicValue],
        *,
        accepted: SymbolicValue,
        metadata: Mapping[str, object] | None = None,
    ) -> InvocationProgram:
        """Validate a model-supplied predicate, then atomically publish all state/output."""

        self._check_open(root=True)
        self._check_value(accepted)
        if accepted.value.type not in (
            ScalarType("bool"),
            TensorType((), "bool"),
            TensorType((1,), "bool"),
        ):
            raise ValueError("acceptance predicate must be a single boolean")
        ports = self._module.outputs
        if set(outputs) != {port.name for port in ports}:
            raise ValueError("finish must supply every declared named output")
        for port in ports:
            self._check_value(outputs[port.name])
            if outputs[port.name].value.type != port.payload:
                raise ValueError(f"output {port.name} payload type mismatch")
        for name, value in self._updates.items():
            self._operations.append(
                ops.stage_write(
                    self._name("pending_state"),
                    value.value.type,
                    name,
                    "vf_transaction",
                    value.value.name,
                )
            )
        valid = self._name("accepted")
        self._operations.append(ops.validate(valid, accepted.value.name, _PREDICATE))
        pending = []
        for port in ports:
            name = self._name("pending_output")
            self._operations.append(
                ops.output_create(
                    name,
                    outputs[port.name].value.name,
                    port.payload,
                    port.name,
                )
            )
            pending.append((name, PendingOutputType(port.name, port.payload)))
        group = ports[0].group
        self._operations.extend(
            (
                ops.output_group("vf_outputs", group, pending),
                ops.transaction_commit(
                    "vf_committed",
                    (item[1] for item in pending),
                    group,
                    "vf_transaction",
                    "vf_outputs",
                    valid,
                ),
                ops.return_values("vf_committed"),
            )
        )
        self._module.add_invocation(
            Invocation(
                "act",
                Block.of(self._operations),
                metadata=dict(metadata or {}),
            )
        )
        module = self._module.build()
        verify(module)
        self._finished = True
        return InvocationProgram(module, MappingProxyType(dict(self._regions)))
