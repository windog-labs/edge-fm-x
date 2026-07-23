"""Stable state/solver/action trace representation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from vlaforge.interpreter.clocks import Epoch
from vlaforge.interpreter.transaction import (
    CommittedAction,
    PendingAction,
    PendingValue,
    SnapshotValue,
)


def normalize_value(value: Any) -> Any:
    trace_summary = getattr(value, "__vlaforge_trace__", None)
    if callable(trace_summary):
        return normalize_value(trace_summary())
    if isinstance(value, Epoch):
        return {
            "clock": value.clock,
            "sequence": value.sequence,
            "timestamp_ns": value.timestamp_ns,
            "episode": value.episode,
        }
    if isinstance(value, SnapshotValue):
        return {
            "state": value.state,
            "version": value.version,
            "epoch": normalize_value(value.epoch),
            "value": normalize_value(value.value),
        }
    if isinstance(value, PendingValue):
        return {
            "state": value.state,
            "epoch": normalize_value(value.epoch),
            "value": normalize_value(value.value),
        }
    if isinstance(value, PendingAction | CommittedAction):
        result = {
            "epoch": normalize_value(value.epoch),
            "value": normalize_value(value.value),
        }
        if isinstance(value, CommittedAction):
            result["transaction_id"] = value.transaction_id
        return result
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): normalize_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple | list):
        return [normalize_value(item) for item in value]

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        detached = value
        if hasattr(detached, "detach"):
            detached = detached.detach()
        if hasattr(detached, "cpu"):
            detached = detached.cpu()
        if hasattr(detached, "numpy"):
            try:
                detached = detached.numpy()
            except Exception:
                pass
        if hasattr(detached, "tobytes"):
            raw = detached.tobytes()
            digest = hashlib.sha256(raw).hexdigest()
            result: dict[str, Any] = {
                "tensor": True,
                "shape": [int(dim) for dim in shape],
                "dtype": str(dtype),
                "sha256": digest,
            }
            size = 1
            for dim in shape:
                size *= int(dim)
            if size <= 64 and hasattr(detached, "tolist"):
                result["values"] = detached.tolist()
            return result
    return {"opaque_type": type(value).__qualname__, "repr": repr(value)}


@dataclass(frozen=True, slots=True)
class TraceEvent:
    index: int
    kind: str
    policy: str
    tick: dict[str, Any]
    op: str
    data: Any


class Trace:
    def __init__(self, events: Iterable[TraceEvent] = ()):
        self._events = list(events)

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def record(
        self,
        kind: str,
        policy: str,
        tick: Epoch,
        op: str,
        data: Any,
    ) -> None:
        self._events.append(
            TraceEvent(
                len(self._events),
                kind,
                policy,
                normalize_value(tick),
                op,
                normalize_value(data),
            )
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": "vlaforge.trace/0.1",
            "events": [asdict(event) for event in self._events],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_data(), sort_keys=True, ensure_ascii=False, indent=indent
        )

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "Trace":
        if data.get("schema") != "vlaforge.trace/0.1":
            raise ValueError(f"unsupported trace schema: {data.get('schema')!r}")
        return cls(TraceEvent(**event) for event in data.get("events", ()))

    @classmethod
    def read(cls, path: str | Path) -> "Trace":
        return cls.from_data(json.loads(Path(path).read_text(encoding="utf-8")))
