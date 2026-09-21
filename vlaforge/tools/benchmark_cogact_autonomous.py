"""Formally measure the native CogACT bundle with its in-process RNG provider."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

from cogact_gpu_monitor import run_monitored


OUTPUT_NAMES = ("raw_action_chunk", "normalized_action_chunk", "native_action_chunk",
                "rng_after", "draws_consumed")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse_rows(path: Path, warmup: int, measured: int) -> list[dict[str, int]]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream, fieldnames=(
            "run", "sample", "measured", "revision", "latency_ns", "exact")))
    if len(rows) != warmup + measured:
        raise ValueError(f"expected {warmup + measured} rows, got {len(rows)}")
    parsed = []
    for index, row in enumerate(rows):
        expected_sample = index % 3
        if (int(row["run"]) != index or int(row["sample"]) != expected_sample
                or int(row["measured"]) != int(index >= warmup)
                or int(row["revision"]) != index + 1
                or int(row["latency_ns"]) <= 0 or int(row["exact"]) != 1):
            raise ValueError(f"formal output audit failed at row {index}: {row}")
        parsed.append({"run": index, "sample": expected_sample,
                       "measured": int(index >= warmup),
                       "latency_ns": int(row["latency_ns"]), "exact": 1})
    return parsed


def hash_outputs(folder: Path, calls: int) -> dict[str, object]:
    files = {}
    for name in OUTPUT_NAMES:
        path = folder / f"raw-{name}.bin"
        if not path.is_file():
            raise ValueError(f"missing complete output archive {path}")
        size = path.stat().st_size
        if size % calls != 0:
            raise ValueError(f"output archive is not call aligned: {path} ({size} bytes)")
        files[path.name] = {"sha256": sha(path), "size_bytes": size,
                            "bytes_per_call": size // calls,
                            "record_order": "invocation order, one contiguous tensor per call"}
    return {"calls": calls, "output_tensors": calls * len(OUTPUT_NAMES), "files": files}


def cdf(values: list[int]) -> list[tuple[int, float]]:
    return [(value, (index + 1) / len(values))
            for index, value in enumerate(sorted(values))]


def plot_cdf(points: list[tuple[int, float]], output: Path) -> None:
    import matplotlib.pyplot as plt

    x = [value / 1e6 for value, _ in points]
    y = [rank for _, rank in points]
    figure, axis = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    axis.plot(x, y, linewidth=1.2)
    axis.set_xlabel("Model latency (ms)")
    axis.set_ylabel("CDF")
    axis.grid(True, linewidth=0.35, alpha=0.5)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"))
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run_campaign(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    gpus = [item.strip() for item in args.gpu.split(",") if item.strip()]
    gpu_uuids = [item.strip() for item in args.gpu_uuid.split(",") if item.strip()]
    if len(gpus) < args.processes or len(gpu_uuids) < args.processes:
        raise ValueError("one distinct GPU and UUID are required for each independent process")
    protocol = {
        "schema": "vlaforge.cogact.autonomous_native_campaign/1",
        "bundle": str(args.bundle.resolve()),
        "bundle_sha256": sha(args.bundle / "bundle.json"),
        "input_root": str(args.data.resolve()),
        "expected_root": str(args.expected.resolve()),
        "gpus": gpus[:args.processes],
        "gpu_uuids": gpu_uuids[:args.processes],
        "processes": args.processes,
        "warmup": args.warmup,
        "measured": args.measured,
        "seed_schedule": [42, 43, 42],
        "runner": str(args.runner.resolve()),
        "runner_sha256": sha(args.runner),
        "timing_boundary": (
            "RNG preparation + Session model execution + CUDA synchronization; "
            "static input H2D/bind and output copies excluded"
        ),
        "output_archive": "raw-<output-name>.bin, contiguous fixed-size tensor records in invocation order",
        "python_deployment": False,
    }
    write(args.output / "protocol.json", protocol)
    processes = []
    all_latencies = []
    def run_one(repeat: int) -> dict[str, object]:
        gpu = gpus[repeat]
        folder = args.output / "runs" / f"{repeat:02d}"
        command = [str(args.runner), str(args.bundle), str(args.data), str(folder),
                   str(args.warmup), str(args.measured)]
        completed = run_monitored(
            command, folder / "monitor", gpu=gpu,
            environment={"PYTHONHOME": "/no/python/home", "PYTHONPATH": "/no/python/path",
                         "CUDA_VISIBLE_DEVICES": gpu,
                         "VLAFORGE_RNG_SEEDS": "42,43,42",
                         "VLAFORGE_EXPECTED_ROOT": str(args.expected)},
        )
        if completed.returncode != 0:
            raise RuntimeError(f"formal process {repeat} failed with {completed.returncode}")
        rows = parse_rows(folder / "monitor" / "stdout.log", args.warmup, args.measured)
        outputs = hash_outputs(folder, args.warmup + args.measured)
        measured_rows = rows[args.warmup:]
        latencies = [row["latency_ns"] for row in measured_rows]
        all_latencies.extend(latencies)
        maps = folder / "process-maps.txt"
        if not maps.is_file() or any(name in maps.read_text().lower()
                                     for name in ("libpython", "libtorch_python")):
            raise ValueError(f"Python library found in process {repeat}")
        process = {
            "repeat": repeat,
            "pid": json.loads((folder / "monitor" / "monitor.json").read_text())["container_pid"],
            "warmup": args.warmup,
            "measured": args.measured,
            "rows": len(rows),
            "latency_ns": {
                "count": len(latencies), "min": min(latencies), "max": max(latencies),
                "mean": statistics.fmean(latencies), "p50": sorted(latencies)[len(latencies) // 2],
            },
            "complete_output_audit": outputs,
            "process_maps_sha256": sha(maps),
            "runner_ldd_sha256": sha(args.bundle.parent / "runner.ldd.txt")
            if (args.bundle.parent / "runner.ldd.txt").is_file() else None,
        }
        write(folder / "report.json", process)
        return process

    if args.parallel:
        with ThreadPoolExecutor(max_workers=args.processes) as pool:
            processes = list(pool.map(run_one, range(args.processes)))
    else:
        processes = [run_one(repeat) for repeat in range(args.processes)]
    processes.sort(key=lambda item: int(item["repeat"]))

    points = cdf(all_latencies)
    with (args.output / "latency-cdf.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("latency_ns", "cdf"))
        writer.writerows(points)
    plot_cdf(points, args.output / "latency-cdf")
    ordered = sorted(all_latencies)
    summary = {
        "count": len(ordered), "min_ns": ordered[0], "max_ns": ordered[-1],
        "mean_ns": statistics.fmean(ordered), "p50_ns": ordered[len(ordered) // 2],
        "p95_ns": ordered[int(len(ordered) * 0.95) - 1],
        "p99_ns": ordered[int(len(ordered) * 0.99) - 1],
        "std_ns": statistics.pstdev(ordered),
        "calls_per_second": 1e9 / statistics.fmean(ordered),
    }
    report = {"schema": "vlaforge.cogact.autonomous_native_report/1", "status": "completed",
              "protocol": protocol, "processes": processes, "summary": summary,
              "complete_measured_output_tensors": len(all_latencies) * len(OUTPUT_NAMES),
              "all_outputs_exact": True, "no_python_deployment": True,
              "autonomous_cpp_rng": True,
              "cdf_csv_sha256": sha(args.output / "latency-cdf.csv"),
              "cdf_png_sha256": sha(args.output / "latency-cdf.png"),
              "cdf_pdf_sha256": sha(args.output / "latency-cdf.pdf")}
    write(args.output / "report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True,
                        help="comma-separated physical GPU ordinals, one per process")
    parser.add_argument("--gpu-uuid", required=True,
                        help="comma-separated GPU UUIDs, one per process")
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--parallel", action="store_true",
                        help="run independent processes concurrently; default is serial")
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--measured", type=int, default=1024)
    args = parser.parse_args()
    if args.processes < 5 or args.warmup < 128 or args.measured < 1024:
        parser.error("formal campaign requires at least 5 processes, 128 warmups and 1024 measured calls")
    run_campaign(args)


if __name__ == "__main__":
    main()
