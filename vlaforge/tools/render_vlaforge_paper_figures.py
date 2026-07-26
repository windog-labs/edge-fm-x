#!/usr/bin/env python3
"""Render deterministic dependency-free SVG figures for the VLAForge paper."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Iterable


COLORS = {
    "ink": "#17212B",
    "muted": "#627180",
    "grid": "#D8E0E7",
    "paper": "#FFFFFF",
    "eager": "#D95F59",
    "direct_artifact": "#4C78A8",
    "generated_session": "#59A14F",
    "accent": "#F2C14E",
    "state": "#9C6ADE",
    "cache": "#4C9FBE",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 14,
    weight: int = 400,
    anchor: str = "start",
    fill: str = COLORS["ink"],
) -> str:
    escaped = html.escape(str(value))
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{escaped}</text>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = COLORS["grid"],
    width: float = 1,
    dash: str | None = None,
    marker: bool = False,
) -> str:
    attributes = (
        f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"'
    )
    if dash:
        attributes += f' stroke-dasharray="{dash}"'
    if marker:
        attributes += ' marker-end="url(#arrow)"'
    return f"<line {attributes}/>"


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    radius: float = 0,
    stroke_width: float = 1,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
        f'height="{height:.1f}" rx="{radius:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}"/>'
    )


def _svg(width: int, height: int, body: Iterable[str]) -> str:
    joined = "\n".join(body)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        "<defs>\n"
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" '
        'refY="3" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L0,6 L9,3 z" fill="#627180"/></marker>\n'
        "</defs>\n"
        f"{joined}\n</svg>\n"
    )


def render_architecture() -> str:
    width, height = 1280, 640
    body = [_rect(0, 0, width, height, fill=COLORS["paper"])]
    body.append(
        _text(
            60,
            60,
            "VLAForge compiles one passive, stateful model invocation",
            size=26,
            weight=700,
        )
    )
    body.append(
        _text(
            60,
            88,
            "Bottom software owns sensors, synchronization, scheduling, and publish",
            size=14,
            fill=COLORS["muted"],
        )
    )

    boxes = (
        (60, 150, 190, 115, "Typed push inputs", "Tensor/Scalar + revision", COLORS["accent"]),
        (300, 150, 190, 115, "Semantic IR", "15 verified VLA ops", "#E7EEF6"),
        (540, 150, 190, 115, "Scheduled Plan", "legality + static memory", "#E7EEF6"),
        (780, 150, 190, 115, "Compile Bundle", "schema + artifact hashes", "#E7EEF6"),
        (1020, 150, 200, 115, "C++ Session", "Bind / Run / ReadOutput", "#DCEFD9"),
    )
    for x, y, w, h, title, subtitle, fill in boxes:
        body.append(
            _rect(
                x,
                y,
                w,
                h,
                fill=fill,
                stroke=COLORS["grid"],
                radius=12,
            )
        )
        body.append(_text(x + w / 2, y + 45, title, size=18, weight=700, anchor="middle"))
        body.append(
            _text(
                x + w / 2,
                y + 75,
                subtitle,
                size=13,
                anchor="middle",
                fill=COLORS["muted"],
            )
        )
    for x1, x2 in ((250, 300), (490, 540), (730, 780), (970, 1020)):
        body.append(
            _line(
                x1 + 5,
                207,
                x2 - 8,
                207,
                stroke=COLORS["muted"],
                width=2,
                marker=True,
            )
        )

    body.append(_text(60, 350, "Verified invocation semantics", size=20, weight=700))
    semantics = (
        ("InputRevision", "exact logical identity", COLORS["accent"]),
        ("Authoritative state", "version only on commit", COLORS["state"]),
        ("Derived cache", "recomputable, exact key", COLORS["cache"]),
        ("Bounded control", "structured if / for", "#8CD0C3"),
        ("Named outputs", "validated atomic group", "#F0A6A6"),
    )
    for index, (title, subtitle, fill) in enumerate(semantics):
        x = 60 + index * 235
        body.append(
            _rect(
                x,
                385,
                205,
                100,
                fill=fill,
                stroke=COLORS["grid"],
                radius=10,
            )
        )
        body.append(_text(x + 102.5, 425, title, size=16, weight=700, anchor="middle"))
        body.append(
            _text(
                x + 102.5,
                453,
                subtitle,
                size=12,
                anchor="middle",
                fill=COLORS["ink"],
            )
        )
    body.append(
        _text(
            640,
            545,
            "Pure TensorRegions bind existing AOTI / cuDNN / CUTLASS / Triton / C++ plugins",
            size=16,
            weight=600,
            anchor="middle",
        )
    )
    body.append(
        _text(
            640,
            580,
            "VLAForge owns orchestration semantics, not model CUDA kernels",
            size=14,
            anchor="middle",
            fill=COLORS["muted"],
        )
    )
    return _svg(width, height, body)


def _baseline_cells(matrix: dict[str, Any]) -> dict[tuple[str, str], dict]:
    return {
        (cell["model"], cell["path"]): cell
        for cell in matrix["cells"]
        if cell["workload"] == "baseline"
    }


def render_performance(matrix: dict[str, Any]) -> str:
    width, height = 1280, 700
    body = [_rect(0, 0, width, height, fill=COLORS["paper"])]
    body.append(
        _text(
            60,
            55,
            "Full-compute steady latency on RTX 3060",
            size=26,
            weight=700,
        )
    )
    body.append(
        _text(
            60,
            84,
            "Baseline workload; 5 independent processes × 30 samples; lower is better",
            size=14,
            fill=COLORS["muted"],
        )
    )
    cells = _baseline_cells(matrix)
    models = ("smolvla", "diffusiondrive")
    paths = ("eager", "direct_artifact", "generated_session")
    labels = {
        "eager": "PyTorch eager",
        "direct_artifact": "Direct AOTI",
        "generated_session": "Generated C++",
    }
    model_labels = {"smolvla": "SmolVLA", "diffusiondrive": "DiffusionDrive"}
    maximum = 120.0
    plot_left, plot_top, plot_width, plot_height = 90, 130, 1120, 430
    body.append(
        _text(
            plot_left,
            plot_top - 14,
            "Latency (ms)",
            size=12,
            weight=600,
            fill=COLORS["muted"],
        )
    )
    for tick in range(0, 121, 20):
        y = plot_top + plot_height - plot_height * tick / maximum
        body.append(_line(plot_left, y, plot_left + plot_width, y))
        body.append(_text(plot_left - 14, y + 5, tick, size=12, anchor="end", fill=COLORS["muted"]))
    group_width = 430
    bar_width = 92
    gap = 18
    for group, model in enumerate(models):
        group_x = 210 + group * 580
        for index, path in enumerate(paths):
            cell = cells[(model, path)]
            value = cell["steady_latency"]["mean_ns"]["estimate"] / 1e6
            low, high = (
                item / 1e6
                for item in cell["steady_latency"]["mean_ns"]["ci95"]
            )
            x = group_x + index * (bar_width + gap)
            height_px = plot_height * value / maximum
            y = plot_top + plot_height - height_px
            body.append(
                _rect(
                    x,
                    y,
                    bar_width,
                    height_px,
                    fill=COLORS[path],
                    radius=5,
                )
            )
            center = x + bar_width / 2
            body.append(_text(center, y - 14, f"{value:.2f}", size=13, weight=700, anchor="middle"))
            low_y = plot_top + plot_height - plot_height * low / maximum
            high_y = plot_top + plot_height - plot_height * high / maximum
            body.append(_line(center, high_y, center, low_y, stroke=COLORS["ink"], width=1.5))
            body.append(_line(center - 7, high_y, center + 7, high_y, stroke=COLORS["ink"], width=1.5))
            body.append(_line(center - 7, low_y, center + 7, low_y, stroke=COLORS["ink"], width=1.5))
        body.append(
            _text(
                group_x + group_width / 2 - 55,
                plot_top + plot_height + 42,
                model_labels[model],
                size=18,
                weight=700,
                anchor="middle",
            )
        )
    legend_y = 650
    for index, path in enumerate(paths):
        x = 335 + index * 230
        body.append(_rect(x, legend_y - 16, 18, 18, fill=COLORS[path], radius=3))
        body.append(_text(x + 28, legend_y, labels[path], size=13))
    return _svg(width, height, body)


def render_ablations(ablations: dict[str, Any]) -> str:
    width, height = 1280, 700
    body = [_rect(0, 0, width, height, fill=COLORS["paper"])]
    body.append(_text(60, 55, "Contribution ablations", size=26, weight=700))
    body.append(
        _text(
            60,
            84,
            "Exact reuse is a real-model cache-only control; memory emphasizes boundedness",
            size=14,
            fill=COLORS["muted"],
        )
    )
    cells = {
        cell["mode"]: cell
        for cell in ablations["exact_reuse"]["cells"]
        if cell["model"] == "diffusiondrive"
    }
    order = ("full", "same", "new", "missing")
    labels = ("Cache off", "Same revision", "New revision", "No revision")
    plot_left, plot_top, plot_width, plot_height = 80, 150, 570, 390
    maximum = 18.0
    body.append(_text(80, 125, "A. Exact reuse latency", size=18, weight=700))
    body.append(
        _text(
            plot_left,
            plot_top - 5,
            "Latency (ms)",
            size=11,
            weight=600,
            fill=COLORS["muted"],
        )
    )
    for tick in range(0, 19, 3):
        y = plot_top + plot_height - plot_height * tick / maximum
        body.append(_line(plot_left, y, plot_left + plot_width, y))
        body.append(_text(plot_left - 12, y + 5, tick, size=11, anchor="end", fill=COLORS["muted"]))
    for index, (mode, label) in enumerate(zip(order, labels, strict=True)):
        value = cells[mode]["steady_latency"]["mean_ns"]["estimate"] / 1e6
        x = plot_left + 45 + index * 135
        height_px = plot_height * value / maximum
        y = plot_top + plot_height - height_px
        fill = COLORS["cache"] if mode == "same" else COLORS["muted"]
        body.append(_rect(x, y, 82, height_px, fill=fill, radius=5))
        body.append(_text(x + 41, y - 12, f"{value:.2f}", size=13, weight=700, anchor="middle"))
        body.append(_text(x + 41, plot_top + plot_height + 28, label, size=11, anchor="middle"))
    body.append(
        _text(
            365,
            610,
            "Same revision: 500/500 hits, 5.35× speedup",
            size=15,
            weight=700,
            anchor="middle",
            fill=COLORS["cache"],
        )
    )

    panel_x = 720
    body.append(_text(panel_x, 125, "B. Static memory classes", size=18, weight=700))
    arena = {item["model"]: item for item in ablations["static_arena"]}
    model_labels = {"smolvla": "SmolVLA", "diffusiondrive": "DiffusionDrive"}
    for index, model in enumerate(("smolvla", "diffusiondrive")):
        record = arena[model]
        y = 185 + index * 175
        width_scale = 420 / record["compiled_bytes"]
        state_width = record["authoritative_state_bytes"] * width_scale
        cache_width = record["derived_cache_bytes"] * width_scale
        other_width = max(0, 420 - state_width - cache_width)
        body.append(_text(panel_x, y - 18, model_labels[model], size=16, weight=700))
        current_x = panel_x
        if state_width:
            body.append(_rect(current_x, y, max(state_width, 3), 45, fill=COLORS["state"]))
            current_x += max(state_width, 3)
        body.append(_rect(current_x, y, cache_width, 45, fill=COLORS["cache"]))
        current_x += cache_width
        body.append(_rect(current_x, y, other_width, 45, fill=COLORS["grid"]))
        body.append(
            _text(
                panel_x,
                y + 75,
                f"packed {record['compiled_bytes']:,} B; saved "
                f"{record['saved_bytes']:,} B ({record['saved_percent']:.3f}%)",
                size=12,
                fill=COLORS["muted"],
            )
        )
        body.append(
            _text(
                panel_x,
                y + 100,
                f"10,000 Runs; CUDA drift {record['soak_cuda_drift_bytes']} B",
                size=12,
                weight=600,
            )
        )
    legend_y = 565
    for index, (label, color) in enumerate(
        (
            ("authoritative state", COLORS["state"]),
            ("derived cache", COLORS["cache"]),
            ("other static", COLORS["grid"]),
        )
    ):
        x = panel_x + index * 160
        body.append(_rect(x, legend_y, 18, 18, fill=color, radius=3))
        body.append(_text(x + 27, legend_y + 14, label, size=11))
    body.append(
        _text(
            panel_x,
            640,
            "Packing savings are small; the claim is verified boundedness and class separation.",
            size=12,
            fill=COLORS["muted"],
        )
    )
    return _svg(width, height, body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--ablations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matrix = _json(args.matrix)
    ablations = _json(args.ablations)
    if not matrix.get("passed") or not ablations.get("passed"):
        raise ValueError("paper figures require passing source reports")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "architecture.svg": render_architecture(),
        "performance.svg": render_performance(matrix),
        "ablations.svg": render_ablations(ablations),
    }
    records = []
    for name, payload in figures.items():
        path = args.output_dir / name
        path.write_text(payload, encoding="utf-8")
        records.append(
            {
                "name": name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema": "vlaforge.paper_figures/1",
        "deterministic": True,
        "sources": {
            "matrix": {
                "path": str(args.matrix),
                "sha256": _sha256(args.matrix),
            },
            "ablations": {
                "path": str(args.ablations),
                "sha256": _sha256(args.ablations),
            },
        },
        "figures": records,
        "reproduction": [
            "python",
            "vlaforge/tools/render_vlaforge_paper_figures.py",
            "--matrix",
            str(args.matrix),
            "--ablations",
            str(args.ablations),
            "--output-dir",
            "<output-dir>",
        ],
    }
    manifest_path = args.output_dir / "figures_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
