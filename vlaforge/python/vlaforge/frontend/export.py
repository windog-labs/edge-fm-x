"""Persistence helpers for verified torch.export captures."""

from __future__ import annotations

import json
from pathlib import Path

from vlaforge.frontend.region_capture import CaptureOutcome


def save_exported_region(
    capture: CaptureOutcome,
    *,
    program_path: str | Path,
    evidence_path: str | Path,
) -> None:
    capture.require_supported()
    assert capture.exported_program is not None
    assert capture.evidence is not None
    import torch

    program_output = Path(program_path)
    evidence_output = Path(evidence_path)
    program_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(capture.exported_program, program_output)
    evidence_output.write_text(
        json.dumps(
            capture.evidence.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_exported_region(path: str | Path) -> object:
    import torch

    return torch.export.load(Path(path))
