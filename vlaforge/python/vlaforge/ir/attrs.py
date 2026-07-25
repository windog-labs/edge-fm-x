"""Small semantic attributes for Invocation IR v0.2."""

from __future__ import annotations

from enum import Enum


class Ownership(str, Enum):
    HOST = "host"
    DEVICE = "device"
    BACKEND = "backend"
    EXTERNAL = "external"


class Effect(str, Enum):
    PURE = "pure"
    READ_STATE = "read_state"
    STAGE_WRITE = "stage_write"
    COMMIT_STATE = "commit_state"
    READ_INPUT = "read_input"
    RETURN_OUTPUT = "return_output"
    EXTERNAL_IO = "external_io"
    RANDOM = "random"
