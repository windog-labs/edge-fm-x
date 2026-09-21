"""Measure the original CogACT model-and-sampler boundary on H20."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from cogact_gpu_monitor import child_handshake, run_monitored


INPUTS = {
    "tokens": ((1, 31), "int64"),
    "dino": ((1, 3, 224, 224), "float32"),
    "siglip": ((1, 3, 224, 224), "float32"),
}
OUTPUTS = {
    "raw_action_chunk": ((16, 7), "float32"),
    "normalized_action_chunk": ((16, 7), "float32"),
    "native_action_chunk": ((16, 7), "float64"),
    "rng_after": ((16,), "uint8"),
    "draws_consumed": ((1,), "int64"),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_reference(root: Path, sample: int, name: str) -> bytes:
    return (root / f"run-{sample}-{name}.bin").read_bytes()


def plot_cdf(values: list[int], output: Path) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(values)
    x = [value / 1e6 for value in ordered]
    y = [(index + 1) / len(ordered) for index in range(len(ordered))]
    figure, axis = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    axis.plot(x, y, linewidth=1.2)
    axis.set_xlabel("Official model latency (ms)")
    axis.set_ylabel("CDF")
    axis.grid(True, linewidth=0.35, alpha=0.5)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"))
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run_worker(args: argparse.Namespace) -> int:
    child_handshake()
    import numpy as np
    import torch
    from vlaforge.adapters.cogact.cogact_host_pipeline import (
        create,
        official_model_samples,
        official_postprocess_outputs,
    )

    protocol = json.loads(args.protocol.read_text())
    if sha(args.protocol) != args.protocol_sha256 or protocol["schema"] != "edgefm.host_pipeline_protocol/1":
        raise ValueError("host protocol identity changed")
    config = protocol["configurations"][args.configuration]
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "vlaforge.cogact.official_model_worker/1",
        "status": "running",
        "pid": os.getpid(),
        "worker": args.worker,
        "protocol_sha256": sha(args.protocol),
        "boundary": (
            "RNG state restore + original VLM/DDIM model and sampler + CUDA synchronization; "
            "static input H2D, preprocessing and output conversion excluded"
        ),
        "python_deployment": True,
    }
    write(output / "execution.json", report)
    try:
        torch.cuda.set_device(0)
        torch.set_num_threads(2)
        init_start = time.perf_counter_ns()
        pipeline = create(config)
        torch.cuda.synchronize()
        report["initialization_ns"] = time.perf_counter_ns() - init_start
        expected_root = args.expected
        samples = []
        for sample in range(3):
            values = {}
            for name, (shape, dtype) in INPUTS.items():
                array = np.fromfile(args.data / f"sample-{sample}" / f"{name}.bin",
                                    dtype=getattr(np, dtype)).reshape(shape)
                values[name] = torch.from_numpy(array.copy()).to("cuda:0").contiguous()
            rng_before = np.fromfile(args.data / f"sample-{sample}" / "rng_before.bin",
                                     dtype=np.uint8).reshape(16)
            samples.append((values, torch.from_numpy(rng_before.copy())))
        count = args.warmup + args.measured
        archives = {
            name: (output / f"raw-{name}.bin").open("xb")
            for name in OUTPUTS
        }
        columns = ("run", "sample", "measured", "latency_ns", "exact")
        with (output / "samples.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for index in range(count):
                sample = index % 3
                values, rng_before = samples[sample]
                start = time.perf_counter_ns()
                torch.cuda.set_rng_state(rng_before, 0)
                torch.cuda.synchronize()
                model_samples, draws = official_model_samples(pipeline.model, values)
                torch.cuda.synchronize()
                elapsed_ns = time.perf_counter_ns() - start
                complete = official_postprocess_outputs(
                    pipeline.model, model_samples, draws, pipeline.samples[0]["unnorm_key"]
                )
                exact = True
                for name, value in complete.items():
                    actual = value.numpy().tobytes()
                    archives[name].write(actual)
                    exact = exact and actual == read_reference(expected_root, sample, name)
                for stream in archives.values():
                    stream.flush()
                writer.writerow({"run": index, "sample": sample, "measured": int(index >= args.warmup),
                                 "latency_ns": elapsed_ns, "exact": int(exact)})
                stream.flush()
                if not exact:
                    raise ValueError(f"complete output differs at call {index}")
        for stream in archives.values():
            stream.close()
        report.update(status="passed", calls=count, warmup=args.warmup, measured=args.measured)
        write(output / "execution.json", report)
        return 0
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        write(output / "execution.json", report)
        raise


def run_controller(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    if args.processes < 5 or args.warmup < 128 or args.measured < 1024:
        raise ValueError("formal official campaign requires 5 processes and 128+1024 calls")
    gpus = [item.strip() for item in args.gpu.split(",") if item.strip()]
    gpu_uuids = [item.strip() for item in args.gpu_uuid.split(",") if item.strip()]
    if len(gpus) < args.processes or len(gpu_uuids) < args.processes:
        raise ValueError("one distinct GPU and UUID are required for each independent process")
    protocol_hash = sha(args.protocol)
    write(args.output / "protocol.json", {
        "schema": "vlaforge.cogact.official_model_protocol/1",
        "host_protocol": str(args.protocol),
        "host_protocol_sha256": protocol_hash,
        "data": str(args.data),
        "expected": str(args.expected),
        "configuration": args.configuration,
        "gpus": gpus[:args.processes],
        "gpu_uuids": gpu_uuids[:args.processes],
        "processes": args.processes,
        "warmup": args.warmup,
        "measured": args.measured,
        "seed_schedule": [42, 43, 42],
        "timing_boundary": (
            "RNG state restore + original VLM/DDIM model and sampler + CUDA synchronization; "
            "static input H2D, preprocessing and output conversion excluded"
        ),
        "benchmark_script_sha256": sha(Path(__file__).resolve()),
    })
    for worker in range(args.processes):
        gpu = gpus[worker]
        gpu_uuid = gpu_uuids[worker]
        worker_output = args.output / f"campaign-{worker:02d}"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(worker),
                   "--protocol", str(args.protocol), "--protocol-sha256", protocol_hash,
                   "--configuration", args.configuration, "--data", str(args.data),
                   "--expected", str(args.expected),
                   "--output", str(worker_output), "--warmup", str(args.warmup),
                   "--measured", str(args.measured), "--gpu", gpu,
                   "--gpu-uuid", gpu_uuid]
        result = run_monitored(command, args.output / f"monitor-{worker:02d}", gpu=gpu,
                               environment={"CUDA_VISIBLE_DEVICES": gpu, "PYTHONDONTWRITEBYTECODE": "1"})
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:])
    repeat_reports = []
    latencies = []
    for worker in range(args.processes):
        folder = args.output / f"campaign-{worker:02d}"
        with (folder / "samples.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != args.warmup + args.measured:
            raise ValueError(f"worker {worker} row count differs")
        for index, row in enumerate(rows):
            if (int(row["run"]) != index or int(row["sample"]) != index % 3
                    or int(row["measured"]) != int(index >= args.warmup)
                    or int(row["latency_ns"]) <= 0 or int(row["exact"]) != 1):
                raise ValueError(f"worker {worker} timing or exactness differs at {index}")
        worker_latencies = [int(row["latency_ns"]) for row in rows[args.warmup:]]
        latencies.extend(worker_latencies)
        archives = {name: sha(folder / f"raw-{name}.bin") for name in OUTPUTS}
        repeat_reports.append({
            "worker": worker,
            "gpu_uuid": gpu_uuids[worker],
            "measured": len(worker_latencies),
            "mean_ns": statistics.fmean(worker_latencies),
            "p50_ns": sorted(worker_latencies)[len(worker_latencies) // 2],
            "archives": archives,
        })
    with (args.output / "latency-cdf.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("latency_ns", "cdf"))
        for index, value in enumerate(sorted(latencies)):
            writer.writerow((value, (index + 1) / len(latencies)))
    plot_cdf(latencies, args.output / "latency-cdf")
    ordered = sorted(latencies)
    summary = {
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
    report = {
        "schema": "vlaforge.cogact.official_model_report/1",
        "status": "completed",
        "protocol_sha256": sha(args.output / "protocol.json"),
        "workers": repeat_reports,
        "summary": summary,
        "all_outputs_exact": True,
        "python_deployment": True,
        "boundary": (
            "RNG state restore + original VLM/DDIM model and sampler + CUDA synchronization; "
            "static input H2D, preprocessing and output conversion excluded"
        ),
        "cdf_csv_sha256": sha(args.output / "latency-cdf.csv"),
        "cdf_png_sha256": sha(args.output / "latency-cdf.png"),
        "cdf_pdf_sha256": sha(args.output / "latency-cdf.pdf"),
    }
    write(args.output / "report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--configuration", default="official-pytorch")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--measured", type=int, default=1024)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.worker is not None:
        args.protocol_sha256 = args.protocol_sha256 or sha(args.protocol)
        raise SystemExit(run_worker(args))
    run_controller(args)


if __name__ == "__main__":
    main()
