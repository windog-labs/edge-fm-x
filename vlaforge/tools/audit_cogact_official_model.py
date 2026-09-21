"""Independently audit the official CogACT model-only campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


OUTPUTS = {
    "raw_action_chunk": 448,
    "normalized_action_chunk": 448,
    "native_action_chunk": 896,
    "rng_after": 16,
    "draws_consumed": 8,
}

OFFICIAL_TIMING_BOUNDARY = (
    "RNG state restore + original VLM/DDIM model and sampler + CUDA synchronization; "
    "static input H2D, preprocessing and output conversion excluded"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(root: Path, sample: int, name: str) -> bytes:
    return (root / f"run-{sample}-{name}.bin").read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = json.loads((args.campaign / "report.json").read_text())
    protocol = json.loads((args.campaign / "protocol.json").read_text())
    if (protocol.get("timing_boundary") != OFFICIAL_TIMING_BOUNDARY
            or campaign.get("boundary") != OFFICIAL_TIMING_BOUNDARY):
        raise ValueError("official timing boundary differs from the frozen contract")
    if protocol.get("seed_schedule") != [42, 43, 42]:
        raise ValueError("official seed schedule differs from the comparison contract")
    if campaign.get("protocol_sha256") != sha(args.campaign / "protocol.json"):
        raise ValueError("official protocol identity differs")
    gpus = protocol.get("gpus")
    gpu_uuids = protocol.get("gpu_uuids")
    if (not isinstance(gpus, list) or not isinstance(gpu_uuids, list)
            or len(gpus) != int(protocol["processes"])
            or len(gpu_uuids) != int(protocol["processes"])):
        raise ValueError("official GPU/UUID process binding differs")
    expected = Path(protocol["expected"])
    references = {
        name: [reference(expected, sample, name) for sample in range(3)]
        for name in OUTPUTS
    }
    latencies = []
    workers = []
    for worker in range(int(protocol["processes"])):
        folder = args.campaign / f"campaign-{worker:02d}"
        execution = json.loads((folder / "execution.json").read_text())
        monitor = json.loads((args.campaign / f"monitor-{worker:02d}" / "monitor.json").read_text())
        if (execution["status"] != "passed" or execution["pid"] != monitor["container_pid"]
                or monitor["status"] != "exited" or monitor["exitcode"] != 0
                or monitor["monitored_gpu"] != gpus[worker]):
            raise ValueError(f"worker {worker}: terminal process evidence differs")
        rows = list(csv.DictReader((folder / "samples.csv").open(newline="")))
        if len(rows) != int(protocol["warmup"]) + int(protocol["measured"]):
            raise ValueError(f"worker {worker}: row count differs")
        for index, row in enumerate(rows):
            if (int(row["run"]) != index or int(row["sample"]) != index % 3
                    or int(row["measured"]) != int(index >= int(protocol["warmup"]))
                    or int(row["latency_ns"]) <= 0 or int(row["exact"]) != 1):
                raise ValueError(f"worker {worker}: invalid row {index}")
        for name, size in OUTPUTS.items():
            path = folder / f"raw-{name}.bin"
            data = path.read_bytes()
            if len(data) != len(rows) * size:
                raise ValueError(f"worker {worker}: archive size differs for {name}")
            for index in range(len(rows)):
                actual = data[index * size:(index + 1) * size]
                if actual != references[name][index % 3]:
                    raise ValueError(f"worker {worker}: {name} differs at call {index}")
        worker_latencies = [int(row["latency_ns"]) for row in rows[int(protocol["warmup"]):]]
        latencies.extend(worker_latencies)
        workers.append({"worker": worker, "calls": len(rows), "outputs_exact": True,
                        "mean_ns": statistics.fmean(worker_latencies),
                        "pid": execution["pid"], "nvml_pid": monitor["nvml_pid"],
                        "gpu_uuid": gpu_uuids[worker],
                        "initialization_ns": execution["initialization_ns"],
                        "execution_sha256": sha(folder / "execution.json"),
                        "monitor_sha256": sha(args.campaign / f"monitor-{worker:02d}" / "monitor.json"),
                        "raw_archive_sha256": {name: sha(folder / f"raw-{name}.bin") for name in OUTPUTS}})
    ordered = sorted(latencies)
    recomputed = {
        "count": len(ordered),
        "min_ns": ordered[0],
        "max_ns": ordered[-1],
        "mean_ns": statistics.fmean(ordered),
        "p50_ns": ordered[len(ordered) // 2],
        "p95_ns": ordered[int(len(ordered) * 0.95) - 1],
        "p99_ns": ordered[int(len(ordered) * 0.99) - 1],
        "std_ns": statistics.pstdev(ordered),
        "calls_per_second": 1e9 / statistics.fmean(ordered),
    }
    if recomputed != campaign["summary"]:
        raise ValueError("recomputed summary differs from campaign report")
    result = {
        "schema": "vlaforge.cogact.official_model_independent_audit/1",
        "status": "passed",
        "campaign": str(args.campaign),
        "campaign_report_sha256": sha(args.campaign / "report.json"),
        "workers": workers,
        "all_outputs_exact": True,
        "latency_summary": recomputed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
