"""Logical clock runtime values."""

from __future__ import annotations

from dataclasses import dataclass

from vlaforge.ir.attrs import EpochExpr


@dataclass(frozen=True, order=True, slots=True)
class Epoch:
    clock: str
    sequence: int
    timestamp_ns: int
    episode: int = 0

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("epoch sequence must be non-negative")
        if self.episode < 0:
            raise ValueError("episode must be non-negative")


@dataclass(frozen=True, slots=True)
class InputSample:
    value: object
    epoch: Epoch


def resolve_epoch(
    expression: EpochExpr,
    epochs: dict[str, Epoch],
    *,
    fallback: Epoch,
) -> Epoch:
    if expression.kind == "constant":
        return Epoch(expression.clock or fallback.clock, expression.offset, 0, fallback.episode)
    if expression.kind == "unknown":
        raise ValueError("cannot resolve unknown epoch expression at runtime")
    if expression.clock is None:
        raise ValueError(f"epoch expression {expression.kind} has no clock")
    base = epochs.get(expression.clock)
    if base is None:
        if expression.clock != fallback.clock:
            raise KeyError(f"no runtime epoch for clock {expression.clock}")
        base = fallback
    if expression.kind in {"current", "input", "solver", "action_chunk"}:
        offset = expression.offset
    elif expression.kind == "next":
        offset = expression.offset or 1
    elif expression.kind == "previous":
        offset = expression.offset or -1
    else:
        raise ValueError(f"unsupported runtime epoch expression: {expression.kind}")
    return Epoch(
        clock=base.clock,
        sequence=base.sequence + offset,
        timestamp_ns=base.timestamp_ns,
        episode=base.episode,
    )

