"""Small explicit builder for Invocation IR v0.2."""

from __future__ import annotations

from dataclasses import replace

from vlaforge.ir.program import (
    InputPort,
    Invocation,
    Module,
    OutputPort,
    StateSlot,
    TensorRegion,
)


class ModuleBuilder:
    def __init__(self, name: str):
        self.name = name
        self.inputs: list[InputPort] = []
        self.outputs: list[OutputPort] = []
        self.states: list[StateSlot] = []
        self.regions: list[TensorRegion] = []
        self.invocations: list[Invocation] = []
        self.metadata: dict[str, object] = {}

    def add_input(self, port: InputPort) -> InputPort:
        if port.input_id is None:
            port = replace(port, input_id=len(self.inputs))
        self.inputs.append(port)
        return port

    def add_output(self, port: OutputPort) -> OutputPort:
        if port.output_id is None:
            port = replace(port, output_id=len(self.outputs))
        self.outputs.append(port)
        return port

    def add_state(self, state: StateSlot) -> StateSlot:
        self.states.append(state)
        return state

    def add_region(self, region: TensorRegion) -> TensorRegion:
        self.regions.append(region)
        return region

    def add_invocation(self, invocation: Invocation) -> Invocation:
        self.invocations.append(invocation)
        return invocation

    def build(self) -> Module:
        return Module(
            name=self.name,
            inputs=tuple(self.inputs),
            outputs=tuple(self.outputs),
            states=tuple(self.states),
            regions=tuple(self.regions),
            invocations=tuple(self.invocations),
            metadata=self.metadata,
        )
