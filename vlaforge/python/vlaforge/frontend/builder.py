"""Small explicit builder for the immutable Python IR."""

from __future__ import annotations

from vlaforge.ir.program import (
    ClockDomain,
    InputStream,
    Module,
    Policy,
    StateSlot,
    TensorRegion,
)


class ModuleBuilder:
    def __init__(self, name: str):
        self.name = name
        self.clocks: list[ClockDomain] = []
        self.inputs: list[InputStream] = []
        self.states: list[StateSlot] = []
        self.regions: list[TensorRegion] = []
        self.policies: list[Policy] = []
        self.metadata: dict[str, object] = {}

    def add_clock(self, clock: ClockDomain) -> ClockDomain:
        self.clocks.append(clock)
        return clock

    def add_input(self, stream: InputStream) -> InputStream:
        self.inputs.append(stream)
        return stream

    def add_state(self, state: StateSlot) -> StateSlot:
        self.states.append(state)
        return state

    def add_region(self, region: TensorRegion) -> TensorRegion:
        self.regions.append(region)
        return region

    def add_policy(self, policy: Policy) -> Policy:
        self.policies.append(policy)
        return policy

    def build(self) -> Module:
        return Module(
            name=self.name,
            clocks=tuple(self.clocks),
            inputs=tuple(self.inputs),
            states=tuple(self.states),
            regions=tuple(self.regions),
            policies=tuple(self.policies),
            metadata=self.metadata,
        )

