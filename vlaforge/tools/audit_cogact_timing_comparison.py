"""Audit that official and native CogACT campaigns share one timing boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path


NATIVE_BOUNDARY = (
    "RNG preparation + Session model execution + CUDA synchronization; "
    "static input H2D/bind and output copies excluded"
)
OFFICIAL_BOUNDARY = (
    "RNG state restore + original VLM/DDIM model and sampler + CUDA synchronization; "
    "static input H2D, preprocessing and output conversion excluded"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def read_latencies(path: Path) -> list[int]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = [int(row["latency_ns"]) for row in rows]
    if not values or sorted(values) != values:
        raise ValueError(f"{path}: CDF latency rows must be non-empty and ordered")
    return values


def bootstrap_interval(native: list[int], official: list[int],
                       samples: int = 10000) -> tuple[float, float]:
    generator = random.Random(0)
    ratios = []
    for _ in range(samples):
        native_mean = statistics.fmean(generator.choices(native, k=len(native)))
        official_mean = statistics.fmean(generator.choices(official, k=len(official)))
        ratios.append(official_mean / native_mean)
    ratios.sort()
    return ratios[int(samples * 0.025)], ratios[int(samples * 0.975) - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-campaign", type=Path, required=True)
    parser.add_argument("--official-campaign", type=Path, required=True)
    parser.add_argument("--native-audit", type=Path, required=True)
    parser.add_argument("--official-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    native_report = read_json(args.native_campaign / "report.json")
    native_protocol = native_report["protocol"]
    official_report = read_json(args.official_campaign / "report.json")
    official_protocol = read_json(args.official_campaign / "protocol.json")
    native_audit = read_json(args.native_audit)
    official_audit = read_json(args.official_audit)

    if native_protocol.get("timing_boundary") != NATIVE_BOUNDARY:
        raise ValueError("native boundary differs")
    if (official_protocol.get("timing_boundary") != OFFICIAL_BOUNDARY
            or official_report.get("boundary") != OFFICIAL_BOUNDARY):
        raise ValueError("official boundary differs")
    for protocol in (native_protocol, official_protocol):
        if (protocol.get("processes") != 5 or protocol.get("warmup") != 128
                or protocol.get("measured") != 1024
                or protocol.get("seed_schedule") != [42, 43, 42]):
            raise ValueError("campaign protocol differs from the comparison contract")
    if (native_protocol.get("gpus") != official_protocol.get("gpus")
            or native_protocol.get("gpu_uuids") != official_protocol.get("gpu_uuids")):
        raise ValueError("official and native GPU process binding differs")
    if Path(native_protocol["input_root"]).resolve() != Path(official_protocol["data"]).resolve():
        raise ValueError("official and native campaign input roots differ")
    if Path(native_protocol["expected_root"]).resolve() != Path(official_protocol["expected"]).resolve():
        raise ValueError("official and native campaign reference roots differ")
    if native_report.get("all_outputs_exact") is not True or official_report.get("all_outputs_exact") is not True:
        raise ValueError("a campaign does not have exact complete outputs")
    if native_report.get("autonomous_cpp_rng") is not True:
        raise ValueError("native campaign is not autonomous C++ RNG")
    if native_audit.get("status") != "passed" or official_audit.get("status") != "passed":
        raise ValueError("an independent campaign audit did not pass")
    if native_audit.get("campaign_report_sha256") != sha(args.native_campaign / "report.json"):
        raise ValueError("native report identity differs from its independent audit")
    if official_audit.get("campaign_report_sha256") != sha(args.official_campaign / "report.json"):
        raise ValueError("official report identity differs from its independent audit")

    native_values = read_latencies(args.native_campaign / "latency-cdf.csv")
    official_values = read_latencies(args.official_campaign / "latency-cdf.csv")
    if len(native_values) != 5120 or len(official_values) != 5120:
        raise ValueError("expected 5120 measured samples per campaign")
    native_mean = statistics.fmean(native_values)
    official_mean = statistics.fmean(official_values)
    ratio = official_mean / native_mean
    low, high = bootstrap_interval(native_values, official_values)
    native_worker_means = [item["latency_ns"]["mean"] for item in native_report["processes"]]
    official_worker_means = [item["mean_ns"] for item in official_report["workers"]]
    if len(native_worker_means) != 5 or len(official_worker_means) != 5:
        raise ValueError("paired GPU comparison requires five worker means per campaign")
    paired_ratios = [official / native for native, official in
                     zip(native_worker_means, official_worker_means, strict=True)]

    result = {
        "schema": "vlaforge.cogact.timing_comparison_audit/1",
        "status": "passed",
        "same_boundary": True,
        "boundary_contract": {
            "random_work_included": True,
            "model_or_sampler_included": True,
            "cuda_completion_included": True,
            "static_input_h2d_excluded": True,
            "static_input_binding_excluded": True,
            "output_conversion_excluded": True,
            "native": NATIVE_BOUNDARY,
            "official": OFFICIAL_BOUNDARY,
        },
        "same_inputs": True,
        "same_seed_schedule": [42, 43, 42],
        "same_precision": {
            "vlm_generation": "FP32 weights with official BF16 autocast",
            "action_model": "FP32",
            "native_action_output": "F64",
        },
        "native": {
            "campaign_report_sha256": sha(args.native_campaign / "report.json"),
            "independent_audit_sha256": sha(args.native_audit),
            "runner_sha256": native_protocol["runner_sha256"],
            "measured": len(native_values),
            "mean_ns": native_mean,
            "p99_ns": native_values[int(len(native_values) * 0.99) - 1],
        },
        "official": {
            "campaign_report_sha256": sha(args.official_campaign / "report.json"),
            "independent_audit_sha256": sha(args.official_audit),
            "benchmark_script_sha256": official_protocol["benchmark_script_sha256"],
            "measured": len(official_values),
            "mean_ns": official_mean,
            "p99_ns": official_values[int(len(official_values) * 0.99) - 1],
        },
        "official_to_native_mean_ratio": ratio,
        "native_mean_reduction_percent": (1.0 - native_mean / official_mean) * 100.0,
        "official_to_native_mean_ratio_95pct_bootstrap": [low, high],
        "paired_gpu_worker_means": {
            "native_ns": native_worker_means,
            "official_ns": official_worker_means,
            "official_to_native_ratio": paired_ratios,
            "mean_ratio": statistics.fmean(paired_ratios),
            "median_ratio": statistics.median(paired_ratios),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
