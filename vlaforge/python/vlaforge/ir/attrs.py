"""Semantic attributes for clocks, persistent state, and effects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class StateScope(str, Enum):
    PROCESS = "process"
    SESSION = "session"
    EPISODE = "episode"
    OBSERVATION = "observation"
    CONTROL = "control"
    SOLVER = "solver"


class ConsistencyPolicy(str, Enum):
    EXCLUSIVE = "exclusive"
    SNAPSHOT = "snapshot"
    APPEND_ONLY = "append_only"
    MERGE = "merge"


class ResetPolicy(str, Enum):
    NEVER = "never"
    SESSION_START = "session_start"
    EPISODE_START = "episode_start"
    ERROR = "error"
    EXPLICIT = "explicit"


class Ownership(str, Enum):
    HOST = "host"
    DEVICE = "device"
    BACKEND = "backend"
    EXTERNAL = "external"


class CheckpointPolicy(str, Enum):
    ALWAYS = "always"
    ON_COMMIT = "on_commit"
    NEVER = "never"


class Effect(str, Enum):
    PURE = "pure"
    READ_STATE = "read_state"
    STAGE_WRITE = "stage_write"
    COMMIT_STATE = "commit_state"
    SAMPLE_INPUT = "sample_input"
    PUBLISH_ACTION = "publish_action"
    AWAIT = "await"
    EXTERNAL_IO = "external_io"
    RANDOM = "random"


@dataclass(frozen=True, slots=True)
class FreshnessConstraint:
    max_age_ns: int | None = None
    max_versions: int | None = None

    def __post_init__(self) -> None:
        if self.max_age_ns is not None and self.max_age_ns < 0:
            raise ValueError("max_age_ns must be non-negative")
        if self.max_versions is not None and self.max_versions < 0:
            raise ValueError("max_versions must be non-negative")
        if self.max_age_ns is None and self.max_versions is None:
            raise ValueError("freshness requires max_age_ns or max_versions")

    def to_dict(self) -> dict[str, int]:
        result: dict[str, int] = {}
        if self.max_age_ns is not None:
            result["max_age_ns"] = self.max_age_ns
        if self.max_versions is not None:
            result["max_versions"] = self.max_versions
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FreshnessConstraint":
        return cls(
            max_age_ns=(
                None if data.get("max_age_ns") is None else int(data["max_age_ns"])
            ),
            max_versions=(
                None
                if data.get("max_versions") is None
                else int(data["max_versions"])
            ),
        )


@dataclass(frozen=True, slots=True)
class EpochExpr:
    """Symbolic logical epoch used by state and input operations."""

    kind: str
    clock: str | None = None
    offset: int = 0
    symbols: tuple[str, ...] = ()

    _KINDS = {
        "constant",
        "current",
        "next",
        "previous",
        "input",
        "solver",
        "action_chunk",
        "unknown",
    }

    def __post_init__(self) -> None:
        if self.kind not in self._KINDS:
            raise ValueError(f"unsupported epoch expression: {self.kind}")
        if self.kind not in {"constant", "unknown"} and not self.clock:
            raise ValueError(f"epoch expression {self.kind} requires a clock")

    @classmethod
    def current(cls, clock: str) -> "EpochExpr":
        return cls("current", clock)

    @classmethod
    def next(cls, clock: str) -> "EpochExpr":
        return cls("next", clock, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "clock": self.clock,
            "offset": self.offset,
            "symbols": list(self.symbols),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpochExpr":
        return cls(
            kind=str(data["kind"]),
            clock=None if data.get("clock") is None else str(data["clock"]),
            offset=int(data.get("offset", 0)),
            symbols=tuple(str(item) for item in data.get("symbols", ())),
        )

