"""Stable state/region/output trace for passive invocations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from vlaforge.interpreter.transaction import (
    CommittedOutputGroup,
    PendingOutput,
    PendingOutputGroup,
    PendingValue,
    SnapshotValue,
)


def normalize_value(value: Any) -> Any:
    trace_summary = getattr(value, "__vlaforge_trace__", None)
    if callable(trace_summary):
        return normalize_value(trace_summary())
    if isinstance(value, SnapshotValue):
        return {
            "state": value.state,
            "version": value.version,
            "episode": value.episode,
            "value": normalize_value(value.value),
        }
    if isinstance(value, PendingValue):
        return {"state": value.state, "value": normalize_value(value.value)}
    if isinstance(value, PendingOutput):
        return {
            "output": value.output,
            "value": normalize_value(value.value),
        }
    if isinstance(value, PendingOutputGroup | CommittedOutputGroup):
        result = {
            "group": value.group,
            "outputs": [normalize_value(item) for item in value.outputs],
        }
        if isinstance(value, CommittedOutputGroup):
            result["transaction_id"] = value.transaction_id
            result["episode"] = value.episode
        return result
    if (
        type(value).__name__ == "StateVersion"
        and hasattr(value, "state")
        and hasattr(value, "version")
        and hasattr(value, "episode")
        and hasattr(value, "value")
    ):
        return {
            "state": str(value.state),
            "version": int(value.version),
            "episode": int(value.episode),
            "value": normalize_value(value.value),
        }
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
        size = 1
        for dim in shape:
            size *= int(dim)
        result: dict[str, Any] = {
            "tensor": True,
            "shape": [int(dim) for dim in shape],
            "dtype": str(dtype),
        }
        # Runtime trace is a control/state-identity contract, not a dump of
        # model activations. Avoid device-to-host copies of large KV/feature
        # tensors; numerical parity is checked separately at Region/output
        # boundaries.
        if size > 4096:
            result["content"] = "omitted_large_tensor"
            return result
        detached = value
        if hasattr(detached, "detach"):
            detached = detached.detach()
        if hasattr(detached, "cpu"):
            detached = detached.cpu()
        raw: bytes | None = None
        if hasattr(detached, "numpy"):
            try:
                array = detached.numpy()
                raw = array.tobytes()
            except Exception:
                pass
        if raw is None and hasattr(detached, "untyped_storage"):
            try:
                contiguous = (
                    detached.contiguous()
                    if hasattr(detached, "contiguous")
                    else detached
                )
                raw = bytes(contiguous.untyped_storage())
            except Exception:
                pass
        if raw is not None:
            result["sha256"] = hashlib.sha256(raw).hexdigest()
        if size <= 64 and hasattr(detached, "tolist"):
            try:
                result["values"] = detached.tolist()
            except Exception:
                pass
        if raw is None and "values" not in result:
            result["content"] = "unavailable"
        return result
    return {"opaque_type": type(value).__qualname__, "repr": repr(value)}


@dataclass(frozen=True, slots=True)
class TraceEvent:
    index: int
    kind: str
    invocation: str
    run: int
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
        invocation: str,
        run: int,
        op: str,
        data: Any,
    ) -> None:
        self._events.append(
            TraceEvent(
                len(self._events),
                kind,
                invocation,
                run,
                op,
                normalize_value(data),
            )
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": "vlaforge.trace/0.2",
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
        if data.get("schema") != "vlaforge.trace/0.2":
            raise ValueError(f"unsupported trace schema: {data.get('schema')!r}")
        return cls(TraceEvent(**event) for event in data.get("events", ()))

    @classmethod
    def read(cls, path: str | Path) -> "Trace":
        return cls.from_data(json.loads(Path(path).read_text(encoding="utf-8")))
