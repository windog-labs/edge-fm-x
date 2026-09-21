"""Recompute archived multimodal baseline tokens and timings without a GPU."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root):
    import numpy as np

    bindings = {}

    def bind(path, expected=None):
        actual = digest(path)
        require(expected is None or actual == expected, f"digest mismatch: {path}")
        bindings[str(path)] = actual
        return actual

    def read(path, expected=None):
        bind(path, expected)
        return json.loads(path.read_text())

    campaign = read(root / "campaign.json")
    protocol = read(root / "protocol.json", campaign["protocol_sha256"])
    require(campaign["status"] == "passed", "campaign did not pass")
    require(protocol["schema"] == "vlaforge.multimodal_baseline_protocol/1", "unsupported protocol")
    require(protocol["mode"] == campaign["mode"], "pilot/formal mode differs")
    require(protocol["mode"] in ("formal", "pilot"), "unknown mode")
    if protocol["mode"] == "formal":
        require(protocol["workers"] >= 5 and protocol["warmup"] >= 128 and protocol["measured"] >= 1024,
                "formal sampling threshold not met")
    require(len(campaign["runs"]) == protocol["workers"], "worker coverage differs")
    workers, seen, previous_end = [], set(), 0
    reference_identity = None
    for i, entry in enumerate(campaign["runs"]):
        require(entry["worker"] == i and entry["exit_code"] == 0, "worker failed or order differs")
        require(entry["start_ns"] >= previous_end and entry["end_ns"] > entry["start_ns"], "workers overlapped")
        previous_end = entry["end_ns"]
        folder = root / f"worker-{i:02d}"
        record = read(folder / "execution.json", entry["report_sha256"])
        monitor = read(root / f"monitor-{i:02d}/monitor.json", entry["monitor_sha256"])
        require(record["status"] == "passed" and record["protocol_sha256"] == campaign["protocol_sha256"],
                "worker report failed or protocol differs")
        require(record["pid"] == monitor["container_pid"] and record["pid"] not in seen,
                "worker process identity differs")
        seen.add(record["pid"])
        require(monitor["status"] == "exited" and monitor["exitcode"] == 0, "monitor failed")
        require(monitor["monitored_gpu"] == record["gpu"]["uuid"] == protocol["gpu_uuid"], "GPU UUID differs")
        require(read(root / f"monitor-{i:02d}/preflight-owners.json") == [], "nonempty owner preflight")
        stages = {event["stage"] for event in monitor["observations"]}
        require({"registered-1", "reset", "registered-2", "running"} <= stages, "missing owner handshake")
        for event in monitor["observations"]:
            require({o["pid"] for o in event["owners"]} <= {monitor["nvml_pid"]}, "foreign GPU owner")
            if event.get("ready"):
                require(event["ready"]["pid"] == record["pid"], "owner marker differs")
        for name, expected in record["files"].items():
            require(Path(name).name == name, "unsafe artifact name")
            bind(folder / name, expected)
        required = {"samples.csv", "reference_tokens.npy", "generated_tokens.npy", "prepared_inputs.npz"}
        require(required <= record["files"].keys(), "incomplete output bindings")
        reference = np.load(folder / "reference_tokens.npy", allow_pickle=False)
        tokens = np.load(folder / "generated_tokens.npy", allow_pickle=False)
        count, length = protocol["warmup"] + protocol["measured"], protocol["new_tokens"]
        require(reference.dtype == tokens.dtype == np.dtype("int64"), "non-int64 tokens")
        require(reference.shape == (length,) and tokens.shape == (count, length), "truncated tokens")
        require(bool(np.all(tokens == reference)), "token output differs from reference")
        current = hashlib.sha256(reference.tobytes()).hexdigest()
        require(reference_identity is None or reference_identity == current, "reference differs across workers")
        reference_identity = current
        with (folder / "samples.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == count, "sample count differs")
        for index, row in enumerate(rows):
            require(int(row["run"]) == index and int(row["new_tokens"]) == length, "sample identity differs")
            require(row["phase"] == ("warmup" if index < protocol["warmup"] else "measured"), "warmup mislabeled")
            total, ttft, prepare, resident = (int(row[k]) for k in (
                "input_to_tokens_ns", "ttft_ns", "prepare_h2d_ns", "resident_generate_ns"))
            require(0 < prepare < ttft < total and total == prepare + resident, "inconsistent timings")
            if length > 1:
                require(math.isclose(float(row["decode_tokens_per_s"]), (length - 1) * 1e9 / (total - ttft),
                                     rel_tol=1e-12), "decode rate includes first token or wrong interval")
        measured = rows[protocol["warmup"]:]
        summary = {name + "_mean_ms": statistics.mean(int(r[name + "_ns"]) for r in measured) / 1e6
                   for name in ("ttft", "input_to_tokens", "resident_generate", "prepare_h2d")}
        workers.append({"pid": record["pid"], "nvml_pid": monitor["nvml_pid"], "calls": count,
                        "measured": len(measured), "complete_tokens_exact": True, **summary})
    for path, expected in bindings.items():
        require(digest(Path(path)) == expected, "evidence changed during audit")
    return {"schema": "vlaforge.multimodal_baseline_audit/1", "status": "passed", "mode": protocol["mode"],
            "backend": protocol["backend"], "workers": workers, "inputs_sha256": bindings,
            "reference_token_sha256": reference_identity, "no_python_deployment": False,
            "full_goal_acceptance": False,
            "scope": "all warmup and measured tokens/timings/owner records; same official backend repeatability only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "audit destination already exists")
    result = audit(args.root.resolve())
    result["audit_source_sha256"] = digest(Path(__file__))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "workers": result["workers"]}))


if __name__ == "__main__":
    main()
