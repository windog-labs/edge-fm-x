"""Exact derived-cache storage; all entries are disposable."""

from __future__ import annotations

import copy
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheLookup:
    hit: bool
    value: object | None


class ExactCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[object, ...], object] = {}
        self.hits = 0
        self.misses = 0

    def lookup(self, key: tuple[object, ...]) -> CacheLookup:
        if key in self._entries:
            self.hits += 1
            return CacheLookup(True, copy.deepcopy(self._entries[key]))
        self.misses += 1
        return CacheLookup(False, None)

    def store(self, key: tuple[object, ...], value: object) -> None:
        self._entries[key] = copy.deepcopy(value)

    def clear(self) -> None:
        self._entries.clear()
