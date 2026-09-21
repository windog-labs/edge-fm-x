#!/usr/bin/env python3
"""Report paired full action chunks and existing raw latency CSV records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from vlaforge.validation.contracts import NumericContract
from vlaforge.validation.deployment_metrics import (
    action_fidelity_report,
    latency_report,
)


def _source(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actions",
        type=Path,
        help="paired action JSON with a locked numerical contract",
    )
    parser.add_argument(
        "--latencies",
        type=Path,
        action="append",
        default=[],
        help="raw latency CSV; repeat for independently collected files",
    )
    parser.add_argument("--deadline-ns", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.actions is None and not args.latencies:
        parser.error("at least one of --actions or --latencies is required")
    if args.deadline_ns is not None and not args.latencies:
        parser.error("--deadline-ns requires --latencies")
    report: dict[str, Any] = {"schema": "vlaforge.deployment_metrics.v1", "sources": []}
    if args.actions is not None:
        payload = json.loads(args.actions.read_text(encoding="utf-8"))
        report["actions"] = action_fidelity_report(
            payload["samples"],
            space=payload["space"],
            contract=NumericContract(**payload["contract"]),
            near_zero_norm=payload.get("near_zero_norm", 1e-12),
            discrete_dimensions=payload.get("discrete_dimensions", ()),
        )
        report["sources"].append(_source(args.actions))
    if args.latencies:
        records = []
        for repeat_index, path in enumerate(args.latencies):
            with path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    row["index"] = int(row["index"])
                    row["latency_ns"] = int(row["latency_ns"])
                    row.setdefault("repeat_id", str(repeat_index))
                    records.append(row)
            report["sources"].append(_source(path))
        report["latency"] = latency_report(records, deadline_ns=args.deadline_ns)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(encoded, encoding="utf-8")
    if "latency" in report:
        for filename, records in (
            ("latency_cdf.csv", report["latency"]["cdf"]),
            ("latency_raw.csv", report["latency"]["raw_samples"]),
        ):
            fields = list(dict.fromkeys(key for item in records for key in item))
            with (args.output / filename).open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
    return 1 if "actions" in report and not report["actions"]["within_tolerance"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
