from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_paper_figure_renderer_is_deterministic(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    tool = repository / "vlaforge/tools/render_vlaforge_paper_figures.py"
    matrix = (
        repository
        / "doc/reports/vlaforge_cuda_matrix_v01/cuda_paper_matrix.json"
    )
    ablations = (
        repository
        / "doc/reports/vlaforge_ablations_v01/paper_ablations.json"
    )
    outputs = (tmp_path / "first", tmp_path / "second")
    for output in outputs:
        subprocess.run(
            [
                sys.executable,
                str(tool),
                "--matrix",
                str(matrix),
                "--ablations",
                str(ablations),
                "--output-dir",
                str(output),
            ],
            check=True,
        )
    names = (
        "architecture.svg",
        "performance.svg",
        "ablations.svg",
        "figures_manifest.json",
    )
    for name in names:
        assert (outputs[0] / name).read_bytes() == (
            outputs[1] / name
        ).read_bytes()
    assert b"<svg" in (outputs[0] / "architecture.svg").read_bytes()
