"""Summarize repeated hrt_model_exec infer logs into raw samples and CDF data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from decimal import Decimal
from pathlib import Path


_LATENCY_RE = re.compile(r"^\s*Infer time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_latency_ns(path: Path) -> list[int]:
    samples: list[int] = []
    with path.open(errors="replace") as stream:
        for line in stream:
            match = _LATENCY_RE.match(line)
            if match is not None:
                samples.append(int(Decimal(match.group(1)) * Decimal(1_000_000)))
    if not samples:
        raise ValueError(f"no Infer time records found in {path}")
    return samples


def nearest_rank(values: list[int], probability: float) -> int:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    if not 0.0 < probability <= 1.0:
        raise ValueError("percentile probability must be in (0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def summarize(samples: list[int]) -> dict[str, int | float]:
    if not samples:
        raise ValueError("cannot summarize an empty sample")
    return {
        "samples": len(samples),
        "mean_ns": statistics.fmean(samples),
        "min_ns": min(samples),
        "p50_ns": nearest_rank(samples, 0.50),
        "p95_ns": nearest_rank(samples, 0.95),
        "p99_ns": nearest_rank(samples, 0.99),
        "max_ns": max(samples),
        "std_ns": statistics.pstdev(samples),
    }


def write_samples(path: Path, rows: list[dict[str, int | str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["process", "sample_index", "phase", "latency_ns"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_percentiles(path: Path, samples: list[int]) -> None:
    ordered = sorted(samples)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["latency_ns", "cdf"])
        for index, latency_ns in enumerate(ordered, start=1):
            writer.writerow([latency_ns, index / len(ordered)])


def label_from_path(path: Path) -> str:
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="hrt_model_exec infer log; repeat for independent processes",
    )
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label",
        action="append",
        help="optional label matching each --input in order",
    )
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    labels = args.label or [label_from_path(path) for path in args.input]
    if len(labels) != len(args.input):
        parser.error("--label count must match --input count")
    if len(set(labels)) != len(labels):
        parser.error("--label values must be unique")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    measured_rows: list[dict[str, int | str]] = []
    process_reports: list[dict[str, object]] = []
    pooled_samples: list[int] = []
    for label, source in zip(labels, args.input, strict=True):
        all_samples = parse_latency_ns(source)
        if len(all_samples) <= args.warmup:
            raise ValueError(
                f"{source} has {len(all_samples)} calls, not more than "
                f"warmup={args.warmup}"
            )
        measured = all_samples[args.warmup :]
        pooled_samples.extend(measured)
        for sample_index, latency_ns in enumerate(measured):
            measured_rows.append(
                {
                    "process": label,
                    "sample_index": sample_index,
                    "phase": "measured",
                    "latency_ns": latency_ns,
                }
            )
        cdf_path = args.output_dir / f"latency-cdf-{label}.csv"
        write_percentiles(cdf_path, measured)
        process_reports.append(
            {
                "label": label,
                "source": str(source.resolve()),
                "source_sha256": sha256(source),
                "calls": len(all_samples),
                "warmup": args.warmup,
                "measured": len(measured),
                "statistics": summarize(measured),
                "cdf_csv": str(cdf_path.resolve()),
                "cdf_csv_sha256": sha256(cdf_path),
            }
        )

    all_samples_path = args.output_dir / "latency-measured.csv"
    write_samples(all_samples_path, measured_rows)
    pooled_cdf_path = args.output_dir / "latency-cdf-all.csv"
    write_percentiles(pooled_cdf_path, pooled_samples)

    report = {
        "schema": "vlaforge.hrt_latency_summary/1",
        "status": "passed",
        "warmup_per_process": args.warmup,
        "processes": process_reports,
        "pooled": summarize(pooled_samples),
        "measured_csv": str(all_samples_path.resolve()),
        "measured_csv_sha256": sha256(all_samples_path),
        "cdf_csv": str(pooled_cdf_path.resolve()),
        "cdf_csv_sha256": sha256(pooled_cdf_path),
        "outliers_removed": False,
    }
    report_path = args.output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
