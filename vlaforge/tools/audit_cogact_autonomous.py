"""Independently audit a completed autonomous CogACT native campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


OUTPUTS = {
    "raw_action_chunk": 448,
    "normalized_action_chunk": 448,
    "native_action_chunk": 896,
    "rng_after": 16,
    "draws_consumed": 8,
}

NATIVE_TIMING_BOUNDARY = (
    "RNG preparation + Session model execution + CUDA synchronization; "
    "static input H2D/bind and output copies excluded"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_exact(path: Path, size: int) -> bytes:
    data = path.read_bytes()
    if len(data) != size:
        raise ValueError(f"unexpected size for {path}: {len(data)} != {size}")
    return data


def audit_repeat(folder: Path, expected: Path, warmup: int, measured: int) -> dict[str, object]:
    calls = warmup + measured
    references = {name: [read_exact(expected / f"run-{sample}-{name}.bin", bytes_per_call)
                         for sample in range(3)]
                  for name, bytes_per_call in OUTPUTS.items()}
    stdout = folder / "monitor/stdout.log"
    with stdout.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if len(rows) != calls:
        raise ValueError(f"{folder}: expected {calls} rows, got {len(rows)}")
    for index, row in enumerate(rows):
        if len(row) != 6 or [int(row[0]), int(row[1]), int(row[2]), int(row[3])] != [
            index, index % 3, int(index >= warmup), index + 1
        ] or int(row[4]) <= 0 or row[5] != "1":
            raise ValueError(f"{folder}: invalid timing row {index}: {row}")

    files = {}
    for name, bytes_per_call in OUTPUTS.items():
        path = folder / f"raw-{name}.bin"
        data = path.read_bytes()
        if len(data) != calls * bytes_per_call:
            raise ValueError(f"{folder}: invalid archive size for {name}")
        for index in range(calls):
            sample = index % 3
            actual = data[index * bytes_per_call:(index + 1) * bytes_per_call]
            reference = references[name][sample]
            if actual != reference:
                raise ValueError(f"{folder}: {name} mismatch at call {index}")
        files[path.name] = {"sha256": sha(path), "size_bytes": len(data),
                            "records": calls, "bytes_per_record": bytes_per_call}

    latencies = [int(row[4]) for row in rows[warmup:]]
    return {"calls": calls, "measured": measured, "outputs_exact": True,
            "latency_ns": {"count": len(latencies), "min": min(latencies),
                            "max": max(latencies), "mean": sum(latencies) / len(latencies)},
            "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.campaign / "report.json").read_text())
    protocol = report["protocol"]
    if protocol.get("timing_boundary") != NATIVE_TIMING_BOUNDARY:
        raise ValueError("native timing boundary differs from the frozen contract")
    if protocol.get("seed_schedule") != [42, 43, 42]:
        raise ValueError("native seed schedule differs from the comparison contract")
    expected = Path(protocol["expected_root"])
    warmup, measured = int(protocol["warmup"]), int(protocol["measured"])
    repeats = []
    for process in report["processes"]:
        folder = args.campaign / "runs" / f"{int(process['repeat']):02d}"
        repeats.append(audit_repeat(folder, expected, warmup, measured))
    all_measured = [int(row[4]) for process in report["processes"]
                    for row in csv.reader((args.campaign / "runs" /
                                           f"{int(process['repeat']):02d}/monitor/stdout.log").open())]
    all_measured = [value for index, value in enumerate(all_measured)
                    if index % (warmup + measured) >= warmup]
    ordered = sorted(all_measured)
    output = {"schema": "vlaforge.cogact.autonomous_independent_audit/1",
              "status": "passed", "campaign": str(args.campaign),
              "campaign_report_sha256": sha(args.campaign / "report.json"),
              "repeats": repeats, "calls": len(ordered),
              "all_outputs_exact": True,
              "latency_ns": {"min": min(ordered), "max": max(ordered),
                             "mean": sum(ordered) / len(ordered),
                             "p50": ordered[len(ordered) // 2],
                             "p95": ordered[int(len(ordered) * .95) - 1],
                             "p99": ordered[int(len(ordered) * .99) - 1]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
