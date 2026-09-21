"""Render exact latency CDFs from raw per-sample CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cdf(path: Path) -> tuple[list[int], list[float]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["latency_ns", "cdf"]:
            raise ValueError(f"unexpected CDF columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CDF: {path}")
    latencies = [int(row["latency_ns"]) for row in rows]
    cdf = [float(row["cdf"]) for row in rows]
    expected = [(index + 1) / len(rows) for index in range(len(rows))]
    if latencies != sorted(latencies) or any(value <= 0 for value in latencies):
        raise ValueError(f"latencies are not positive and sorted: {path}")
    if cdf != expected:
        raise ValueError(f"CDF ranks differ from raw samples: {path}")
    return latencies, cdf


def parse_series(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("series must be LABEL=CSV")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("series must be LABEL=CSV")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="append",
        type=parse_series,
        required=True,
        metavar="LABEL=CSV",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Latency CDF")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    records = []
    for label, path in args.series:
        latencies, cdf = load_cdf(path)
        milliseconds = [value / 1e6 for value in latencies]
        axis.step(
            milliseconds,
            cdf,
            where="post",
            linewidth=1.6,
            label=f"{label} ({len(latencies)} samples)",
        )
        records.append(
            {
                "label": label,
                "source": str(path.resolve()),
                "source_sha256": sha256(path),
                "samples": len(latencies),
                "minimum_ns": latencies[0],
                "maximum_ns": latencies[-1],
                "outliers_removed": False,
            }
        )
    axis.set_xlabel("Latency (ms)")
    axis.set_ylabel("Empirical CDF")
    axis.set_ylim(0.0, 1.01)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title(args.title)
    figure.tight_layout()
    png = args.output / "latency-cdf.png"
    pdf = args.output / "latency-cdf.pdf"
    figure.savefig(png)
    figure.savefig(pdf)
    plt.close(figure)
    report = {
        "schema": "vlaforge.latency_cdf_figure/1",
        "status": "rendered",
        "series": records,
        "png_sha256": sha256(png),
        "pdf_sha256": sha256(pdf),
        "outliers_removed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": report["status"], "series": len(records)}, indent=2))


if __name__ == "__main__":
    main()
