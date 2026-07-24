"""Structured unsupported reports for the restricted source frontend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


UNSUPPORTED_SCHEMA = "vlaforge.frontend_unsupported/1"


class UnsupportedSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class UnsupportedItem:
    code: str
    message: str
    severity: UnsupportedSeverity = UnsupportedSeverity.ERROR
    source: str | None = None
    remediation: str | None = None

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("unsupported item requires code and message")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "remediation": self.remediation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UnsupportedItem":
        return cls(
            code=str(data["code"]),
            message=str(data["message"]),
            severity=UnsupportedSeverity(str(data.get("severity", "error"))),
            source=None if data.get("source") is None else str(data["source"]),
            remediation=(
                None
                if data.get("remediation") is None
                else str(data["remediation"])
            ),
        )


@dataclass(frozen=True, slots=True)
class UnsupportedReport:
    region: str
    stage: str
    items: tuple[UnsupportedItem, ...]
    schema: str = UNSUPPORTED_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != UNSUPPORTED_SCHEMA:
            raise ValueError(f"unsupported report schema: {self.schema!r}")
        if not self.region or not self.stage:
            raise ValueError("unsupported report requires region and stage")

    @property
    def supported(self) -> bool:
        return not any(
            item.severity is UnsupportedSeverity.ERROR for item in self.items
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "region": self.region,
            "stage": self.stage,
            "supported": self.supported,
            "items": [item.to_dict() for item in self.items],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False, indent=indent
        )

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UnsupportedReport":
        report = cls(
            schema=str(data["schema"]),
            region=str(data["region"]),
            stage=str(data["stage"]),
            items=tuple(
                UnsupportedItem.from_dict(item) for item in data.get("items", ())
            ),
        )
        if "supported" in data and bool(data["supported"]) != report.supported:
            raise ValueError("unsupported report has inconsistent supported flag")
        return report

    @classmethod
    def read(cls, path: str | Path) -> "UnsupportedReport":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("unsupported report root must be an object")
        return cls.from_dict(data)


class FrontendUnsupportedError(RuntimeError):
    def __init__(self, report: UnsupportedReport):
        self.report = report
        summary = "; ".join(f"{item.code}: {item.message}" for item in report.items)
        super().__init__(f"region {report.region} is unsupported: {summary}")
