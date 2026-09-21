"""Independently audit a completed native multimodal Session campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> object:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def nearest_rank(values: list[int], percentile: float) -> int:
    require(bool(values), "nearest rank requires samples")
    ordered = sorted(values)
    index = max(1, math.ceil(percentile * len(ordered))) - 1
    return ordered[index]


def summarize(values: list[int]) -> dict[str, object]:
    require(bool(values), "latency summary requires samples")
    ordered = sorted(values)
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "mean_ns": mean,
        "min_ns": ordered[0],
        "max_ns": ordered[-1],
        "std_ns": statistics.pstdev(values),
        "sequential_calls_per_second": 1e9 / mean,
        "p50_ns": nearest_rank(values, 0.50),
        "p90_ns": nearest_rank(values, 0.90),
        "p95_ns": nearest_rank(values, 0.95),
        "p99_ns": nearest_rank(values, 0.99),
    }


def close_summary(observed: dict[str, object], expected: dict[str, object]) -> None:
    for key, value in expected.items():
        require(key in observed, f"missing summary field {key}")
        if isinstance(value, int):
            require(observed[key] == value, f"summary integer field differs: {key}")
        else:
            require(
                math.isclose(
                    float(observed[key]),
                    float(value),
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                ),
                f"summary floating field differs: {key}",
            )


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_protocol(protocol: dict, prepared: dict) -> None:
    require(
        protocol.get("schema") == "vlaforge.session_latency_protocol/2",
        "unsupported native Session protocol",
    )
    require(protocol.get("processes") == 5, "formal audit requires five processes")
    require(protocol.get("warmup") == 128, "formal audit requires 128 warmups")
    require(protocol.get("measured") == 1024, "formal audit requires 1024 measured calls")
    require(protocol.get("policies") == ["off"], "audit expects one off policy")
    require(protocol.get("full_paper_acceptance") is False, "campaign must not self-approve paper scope")
    require(prepared.get("status") == "prepared_not_executed", "prepared state differs")
    require(
        len(protocol.get("reference_limitations", [])) >= 3,
        "fixed-profile limitations are not explicit",
    )


def reference_bytes(source: Path, spec: dict) -> bytes:
    import numpy as np

    with np.load(source, allow_pickle=False) as arrays:
        require(spec["name"] in arrays, f"reference lacks {spec['name']}")
        value = arrays[spec["name"]]
        require(
            list(value.shape) == list(spec["shape"]),
            f"reference shape differs for {spec['name']}",
        )
        if spec["dtype"] == "bf16":
            require(
                value.dtype == np.dtype("float32"),
                f"BF16 reference storage differs for {spec['name']}",
            )
            words = value.astype("<f4", copy=False).view("<u4")
            return (words >> np.uint32(16)).astype("<u2").tobytes(order="C")
        return np.ascontiguousarray(value).tobytes(order="C")


def audit_worker(
    campaign: Path,
    repeat: int,
    policy: str,
    protocol: dict,
    references: dict[tuple[int, str], bytes],
    tensor_specs: list[dict],
    seen_pids: set[int],
) -> tuple[dict[str, object], list[int]]:
    folder = campaign / "runs" / f"{repeat:02d}-{policy}"
    report = read_json(folder / "report.json")
    execution = read_json(folder / "execution.json")
    require(report.get("status") == "passed", f"worker {repeat} did not pass")
    require(report.get("pilot") is False and execution.get("pilot") is False, "pilot mixed into formal audit")
    require(report.get("repeat") == repeat and report.get("policy") == policy, "worker identity differs")
    require(execution.get("exit_code") == 0, f"worker {repeat} exited non-zero")
    pid = int(execution["pid"])
    require(pid not in seen_pids, "worker PIDs are not independent")
    seen_pids.add(pid)
    require(execution.get("foreign_compute_owners") is None, "foreign GPU owner observed")
    require(execution.get("monitor_error") is None, "GPU ownership monitor failed")
    require(report.get("no_python_deployment") is True, "worker does not declare no-Python deployment")
    require(report.get("all_same_artifact_bitwise_equal") is True, "not all calls match the artifact")
    require(report.get("all_eager_bitwise_equal") is True, "not all calls match eager output")
    require(report.get("all_paper_numeric_gates_passed") is True, "paper numeric gates failed")
    require(report.get("quality_gate") == "passed", "worker quality gate failed")

    rows = load_csv(folder / "samples.csv")
    calls = protocol["warmup"] + protocol["measured"]
    require(len(rows) == calls, f"worker {repeat} sample count differs")
    require(
        report.get("latency", {}).get("summary", {}).get("count") == protocol["measured"],
        "worker latency count differs",
    )
    steady = []
    for index, row in enumerate(rows):
        require(int(row["run"]) == index, "sample run order differs")
        require(
            int(row["sample"]) == index % len(protocol["samples"]),
            "sample cycling differs",
        )
        require(
            int(row["measured"]) == int(index >= protocol["warmup"]),
            "warmup phase differs",
        )
        require(int(row["revision"]) == index + 1, "input revision differs")
        require(row["finite"] == "1", "non-finite output observed")
        require(row["direct_exact"] == "1", "raw output differs from direct reference")
        require(float(row["direct_mse"]) == 0.0, "direct MSE is non-zero")
        require(float(row["direct_max_abs"]) == 0.0, "direct max abs is non-zero")
        if index >= protocol["warmup"]:
            steady.append(int(row["latency_ns"]))
    close_summary(report["latency"]["summary"], summarize(steady))

    raw_hashes = {}
    complete_tensors = 0
    total_values = 0
    for spec in tensor_specs:
        raw_path = folder / spec["raw_file"]
        record_size = int(spec["size_bytes"])
        require(raw_path.stat().st_size == calls * record_size, "raw output archive size differs")
        raw_hashes[spec["name"]] = sha256(raw_path)
        with raw_path.open("rb") as stream:
            for index, row in enumerate(rows):
                actual = stream.read(record_size)
                require(len(actual) == record_size, "raw output archive is truncated")
                expected = references[int(row["sample"]), spec["name"]]
                require(actual == expected, f"{spec['name']} differs at call {index}")
        complete_tensors += calls
        total_values += calls * int(spec["count"])
    binding = report.get("multi_output_evidence", {}).get("raw_sha256")
    require(binding == raw_hashes, "worker raw-output hash binding differs")
    require(
        report.get("validated_output_tensors") == complete_tensors,
        "complete tensor count differs",
    )

    maps = (folder / "process-maps.txt").read_text().lower()
    require("libpython" not in maps and "libtorch_python" not in maps, "Python is mapped in the worker")
    runtime = report.get("runtime_library_evidence", {})
    require(runtime.get("verified") is True, "runtime library verification failed")
    require(runtime.get("maps_sha256") == report.get("process_maps_sha256"), "maps hash binding differs")
    for item in runtime.get("actual_libraries", []):
        path = Path(item["path"])
        require(path.is_file(), "actual runtime library is missing")
        require(sha256(path) == item["sha256"], "actual runtime library hash differs")
        require(str(path.resolve()) in (folder / "process-maps.txt").read_text(), "runtime library not in maps")

    allocator = report.get("allocator_observation")
    require(isinstance(allocator, dict), "allocator evidence binding is missing")
    require(sha256(folder / "allocator-snapshots.jsonl") == allocator["raw_snapshots_sha256"],
            "allocator raw hash differs")
    require(sha256(folder / "allocator-report.json") == allocator["report_sha256"],
            "allocator report hash differs")
    allocator_report = read_json(folder / "allocator-report.json")
    steady_allocator = allocator_report["steady_interval"]
    require(steady_allocator["ooms"] == 0, "allocator OOM observed")
    require(steady_allocator["allocation_retries"] == 0, "allocator retry observed")
    require(steady_allocator["device_allocation_calls"] == 0, "steady allocator grew")

    telemetry = [json.loads(line) for line in (folder / "telemetry.jsonl").read_text().splitlines()]
    require(bool(telemetry), "GPU telemetry is empty")
    peak_memory = 0
    hardware = None
    for item in telemetry:
        require(item["exit_code"] == 0, "nvidia-smi telemetry query failed")
        require(
            {owner["pid"] for owner in item["compute_owners"]}
            <= {int(execution["monitored_nvml_pid"])},
            "foreign GPU owner observed in telemetry",
        )
        device_rows = list(csv.DictReader(item["stdout"].splitlines(), skipinitialspace=True))
        require(len(device_rows) == 1, "telemetry row count differs")
        device = device_rows[0]
        require(device["uuid"] == protocol["monitor_gpu"], "telemetry GPU UUID differs")
        identity = {
            "name": device["name"],
            "uuid": device["uuid"],
            "driver_version": device["driver_version"],
        }
        require(hardware is None or hardware == identity, "worker mixes GPU hardware identities")
        hardware = identity
        peak_memory = max(peak_memory, int(device["memory.used [MiB]"].split()[0]))

    hashes = {
        str((folder / name).relative_to(campaign)): sha256(folder / name)
        for name in (
            "report.json",
            "execution.json",
            "samples.csv",
            "process-maps.txt",
            "allocator-snapshots.jsonl",
            "allocator-report.json",
            "multi-output-fidelity.json",
            "telemetry.jsonl",
            *(spec["raw_file"] for spec in tensor_specs),
        )
    }
    return (
        {
            "repeat": repeat,
            "policy": policy,
            "pid": pid,
            "nvml_pid": int(execution["monitored_nvml_pid"]),
            "calls": calls,
            "measured": protocol["measured"],
            "complete_output_tensors": complete_tensors,
            "complete_output_values": total_values,
            "all_outputs_exact": True,
            "latency": summarize(steady),
            "peak_sampled_device_used_mib": peak_memory,
            "hardware": hardware,
            "files": hashes,
        },
        steady,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    campaign = args.campaign.resolve(strict=True)
    protocol = read_json(campaign / "protocol.json")
    prepared = read_json(campaign / "prepared.json")
    validate_protocol(protocol, prepared)
    tensor_contract = read_json(campaign / "tensor-contract.json")
    tensor_specs = tensor_contract["outputs"]
    require(
        {item["name"] for item in tensor_specs} == {"tokens", "logits"},
        "unsupported native output contract",
    )
    references = {
        (sample_index, spec["name"]): reference_bytes(
            Path(protocol["samples"][sample_index]["outputs"][spec["name"]]["direct"]),
            spec,
        )
        for sample_index in range(len(protocol["samples"]))
        for spec in tensor_specs
    }

    frozen = read_json(campaign / "frozen-files.json")
    require(len(frozen) >= 100, "frozen file inventory is unexpectedly small")
    for name, expected in frozen.items():
        path = Path(name)
        require(path.is_file(), f"frozen file is missing: {path}")
        require(sha256(path) == expected, f"frozen file changed: {path}")

    seen_pids: set[int] = set()
    workers = []
    combined_samples = []
    for repeat in range(protocol["processes"]):
        worker, samples = audit_worker(
            campaign,
            repeat,
            "off",
            protocol,
            references,
            tensor_specs,
            seen_pids,
        )
        workers.append(worker)
        combined_samples.extend(samples)
    combined = summarize(combined_samples)
    report = read_json(campaign / "report.json")
    require(report.get("status") == "completed", "campaign aggregate is incomplete")
    aggregate = report["policies"]["off"]["combined"]
    close_summary(aggregate["summary"], combined)
    cdf_rows = load_csv(campaign / "off-cdf.csv")
    ordered = sorted(combined_samples)
    require(len(cdf_rows) == len(ordered), "CDF row count differs")
    require(
        [int(row["latency_ns"]) for row in cdf_rows] == ordered,
        "CDF samples differ from raw latency records",
    )
    require(
        all(
            math.isclose(float(row["cdf"]), (index + 1) / len(cdf_rows), rel_tol=0, abs_tol=0)
            for index, row in enumerate(cdf_rows)
        ),
        "CDF ranks differ from raw records",
    )
    hardware = workers[0]["hardware"]
    require(
        all(worker["hardware"] == hardware for worker in workers),
        "campaign mixes hardware identities",
    )
    audit = {
        "schema": "edgefm.native_session_campaign_audit/1",
        "status": "passed",
        "formal_experiment_complete": True,
        "campaign": str(campaign),
        "model": protocol["model"],
        "protocol_sha256": sha256(campaign / "protocol.json"),
        "prepared_sha256": sha256(campaign / "prepared.json"),
        "frozen_files_sha256": sha256(campaign / "frozen-files.json"),
        "aggregate_report_sha256": sha256(campaign / "report.json"),
        "tensor_contract_sha256": sha256(campaign / "tensor-contract.json"),
        "processes": workers,
        "measured_calls": len(combined_samples),
        "complete_output_tensors": sum(
            worker["complete_output_tensors"] for worker in workers
        ),
        "complete_output_values": sum(
            worker["complete_output_values"] for worker in workers
        ),
        "all_outputs_exact": True,
        "latency": combined,
        "peak_sampled_device_used_mib": max(
            worker["peak_sampled_device_used_mib"] for worker in workers
        ),
        "hardware": hardware,
        "boundary": protocol["boundary"],
        "reference_limitations": protocol["reference_limitations"],
        "outliers_removed": False,
        "python_in_worker_maps": False,
        "full_paper_acceptance": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": audit["status"],
        "measured_calls": audit["measured_calls"],
        "mean_ms": audit["latency"]["mean_ns"] / 1e6,
        "p99_ms": audit["latency"]["p99_ns"] / 1e6,
    }, indent=2))


if __name__ == "__main__":
    main()
