"""Versioned real-model frontend audit reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from vlaforge.frontend.region_capture import CaptureOutcome


FRONTEND_AUDIT_SCHEMA = "vlaforge.frontend_model_audit/2"


@dataclass(frozen=True, slots=True)
class RegionAuditRecord:
    name: str
    source_location: str
    major_compute: bool
    supported: bool
    graph_digest: str | None
    graph_nodes: int | None
    export_seconds: float | None
    maximum_absolute_error: float | None
    effect_audit: Mapping[str, object] | None
    unsupported_report: Mapping[str, object] | None

    @classmethod
    def from_capture(
        cls,
        capture: CaptureOutcome,
        *,
        source_location: str,
        major_compute: bool,
    ) -> "RegionAuditRecord":
        evidence = capture.evidence
        return cls(
            name=capture.region.name,
            source_location=source_location,
            major_compute=major_compute,
            supported=capture.supported,
            graph_digest=None if evidence is None else evidence.graph_digest,
            graph_nodes=(
                None
                if capture.exported_program is None
                else len(tuple(capture.exported_program.graph_module.graph.nodes))
            ),
            export_seconds=None if evidence is None else evidence.export_seconds,
            maximum_absolute_error=(
                None if evidence is None else evidence.maximum_absolute_error
            ),
            effect_audit=(
                None if evidence is None else evidence.effect_audit.to_dict()
            ),
            unsupported_report=(
                None if capture.supported else capture.report.to_dict()
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_location": self.source_location,
            "major_compute": self.major_compute,
            "supported": self.supported,
            "graph_digest": self.graph_digest,
            "graph_nodes": self.graph_nodes,
            "export_seconds": self.export_seconds,
            "maximum_absolute_error": self.maximum_absolute_error,
            "effect_audit": self.effect_audit,
            "unsupported_report": self.unsupported_report,
        }


@dataclass(frozen=True, slots=True)
class ModelFrontendAudit:
    model: str
    checkpoint_path: str
    checkpoint_revision: str
    checkpoint_digests: tuple[tuple[str, str], ...]
    torch_version: str
    device: str
    persistent_states: tuple[str, ...]
    persistent_state_evidence_complete: bool
    regions: tuple[RegionAuditRecord, ...]
    validation_passed: bool = True
    validation_checks: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()
    schema: str = FRONTEND_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FRONTEND_AUDIT_SCHEMA:
            raise ValueError(f"unsupported frontend audit schema: {self.schema!r}")
        if not self.model or not self.checkpoint_path or not self.torch_version:
            raise ValueError("frontend audit is missing model provenance")
        names = [item.name for item in self.regions]
        if not self.regions or len(names) != len(set(names)):
            raise ValueError("frontend audit regions must be non-empty and unique")
        if tuple(sorted(self.checkpoint_digests)) != self.checkpoint_digests:
            raise ValueError("checkpoint digests must be sorted by path")
        if tuple(sorted(self.validation_checks)) != self.validation_checks:
            raise ValueError("validation checks must be sorted by name")

    @property
    def passed(self) -> bool:
        return (
            self.persistent_state_evidence_complete
            and self.validation_passed
            and all(
                record.supported
                for record in self.regions
                if record.major_compute
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "model": self.model,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_revision": self.checkpoint_revision,
            "checkpoint_digests": [
                {"path": path, "sha256": digest}
                for path, digest in self.checkpoint_digests
            ],
            "torch_version": self.torch_version,
            "device": self.device,
            "persistent_states": list(self.persistent_states),
            "persistent_state_evidence_complete": (
                self.persistent_state_evidence_complete
            ),
            "regions": [item.to_dict() for item in self.regions],
            "validation_passed": self.validation_passed,
            "validation_checks": [
                {"name": name, "value": value}
                for name, value in self.validation_checks
            ],
            "notes": list(self.notes),
            "passed": self.passed,
        }

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
