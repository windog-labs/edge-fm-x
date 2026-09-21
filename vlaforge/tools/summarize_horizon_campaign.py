"""Audit a completed HBM campaign and summarize its exact measured samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def audit_process(root, run):
    record = read_json(root / "command.json")
    if record["status"] != "completed" or record["exit_code"] != 0:
        raise ValueError("process failed or is incomplete")
    if any(run.get(k) != v for k, v in record.items()):
        raise ValueError("process evidence differs from campaign record")
    if record["finished_ns"] <= record["started_ns"]:
        raise ValueError("invalid process time window")
    resource_status = record.get("resource_status")
    if resource_status is not None:
        if (resource_status != "clear-at-poll-points"
                or record.get("resource_checks", 0) <= 0
                or record.get("foreign_bpu_users")
                or record.get("resource_events")
                or record.get("resource_errors")
                or record.get("contention_stop_requested")):
            raise ValueError("foreign BPU interference or missing resource monitoring")
    if digest(root / "stdout.log") != record["stdout_sha256"]:
        raise ValueError("raw inference log hash mismatch")
    identities = []
    for name in ("before.json", "after.json"):
        snapshot = read_json(root / name)
        files = snapshot.get("files", {})
        serial = files.get("/proc/device-tree/serial-number")
        uname = snapshot.get("uname")
        if not serial or not isinstance(uname, list) or len(uname) != 5:
            raise ValueError("missing board identity snapshot")
        identities.append({"serial_number": serial, "uname": uname})
        users = set()
        for path in ("/sys/devices/system/bpu/users", "/sys/devices/system/bpu/bpu0/users"):
            value = files.get(path)
            if value is None:
                raise ValueError("missing BPU ownership snapshot")
            users.update(int(pid) for pid in re.findall(
                r"^\s*(\d+)\s+\d+(?:\.\d+)?\s*$", value, re.M))
        if users - {record["pid"]}:
            raise ValueError("foreign BPU client in process snapshot")
    if identities[0] != identities[1]:
        raise ValueError("board identity differs between process snapshots")
    return record, identities[0]


def audit(campaign_root, smoke_root, stage_names=None):
    campaign = read_json(campaign_root / "campaign.json")
    smoke = read_json(smoke_root / "campaign.json")
    allowed_statuses = ("completed",) if stage_names is None else ("completed", "running", "failed")
    if campaign.get("status") not in allowed_statuses or campaign.get("phase") != "formal":
        raise ValueError("formal campaign is not completed")
    if smoke.get("status") != "completed" or smoke.get("phase") != "smoke":
        raise ValueError("smoke campaign is not completed")
    if campaign["manifest_sha256"] != smoke["manifest_sha256"]:
        raise ValueError("smoke/formal manifests differ")
    if campaign["manifest"] != smoke["manifest"]:
        raise ValueError("embedded smoke/formal manifests differ")
    if campaign["inputs"] != smoke["inputs"]:
        raise ValueError("smoke/formal input hashes differ")
    manifest = campaign["manifest"]
    all_stage_names = {stage["name"] for stage in manifest["stages"]}
    selected_names = all_stage_names if stage_names is None else set(stage_names)
    if not selected_names or not selected_names <= all_stage_names:
        raise ValueError("unknown or empty stage selection")
    stages = [stage for stage in manifest["stages"] if stage["name"] in selected_names]
    runs = [run for run in campaign["runs"] if run["stage"] in selected_names]
    if stage_names is None and len(runs) != len(campaign["runs"]):
        raise ValueError("campaign contains an unknown stage")
    warmup = manifest.get("warmup", 128)
    measured = manifest.get("measured", 1024)
    processes = manifest.get("processes", 5)
    expected_smoke = {(s["name"], label) for s in manifest["stages"]
                      for label in ("model-info", "infer", "perf")}
    if (len(smoke["runs"]) != len(expected_smoke)
            or {(r["stage"], r["label"]) for r in smoke["runs"]} != expected_smoke):
        raise ValueError("missing or duplicated smoke command")
    board_identity = None
    for run in smoke["runs"]:
        record, identity = audit_process(smoke_root / run["stage"] / run["label"], run)
        if board_identity is not None and identity != board_identity:
            raise ValueError("smoke board identity differs across commands")
        board_identity = identity
        stage = next(s for s in manifest["stages"] if s["name"] == run["stage"])
        command = record["command"]
        mode = "model_info" if run["label"] == "model-info" else run["label"]
        if command[:2] != [manifest["executable"], mode]:
            raise ValueError("unexpected smoke executable or command")
        required = [("--model_file", stage["hbm"])]
        if mode != "model_info":
            required += [("--input_file", ",".join(stage["inputs"])),
                         ("--frame_count", "1" if mode == "infer" else "5")]
        if mode == "perf":
            required += [("--thread_num", "1")]
        for flag, value in required:
            if (command.count(flag) != 1 or command.index(flag) + 1 >= len(command)
                    or command[command.index(flag) + 1] != value):
                raise ValueError(f"unexpected smoke argument: {flag}")
    expected = {(s["name"], f"run-{i + 1}") for s in stages for i in range(processes)}
    if len(runs) != len(expected):
        raise ValueError("wrong number of completed processes")
    if {(r["stage"], r["label"]) for r in runs} != expected:
        raise ValueError("missing or duplicated stage/process label")
    previous_finish = None
    for run in runs:
        root = campaign_root / run["stage"] / run["label"]
        record, identity = audit_process(root, run)
        if identity != board_identity:
            raise ValueError("formal board identity differs from smoke")
        if previous_finish is not None and record["started_ns"] < previous_finish:
            raise ValueError("process execution windows overlap")
        previous_finish = record["finished_ns"]
        if record["infer_records"] != warmup + measured:
            raise ValueError("incorrect inference count")
        source = root / "stdout.log"
        frames = [int(x) for x in re.findall(r"^-+Frame (\d+) begin-+\s*$", source.read_text(), re.M)]
        if frames != list(range(warmup + measured)):
            raise ValueError("raw frame indices are not complete and sequential")
        command = record["command"]
        stage = next(s for s in manifest["stages"] if s["name"] == run["stage"])
        expected_command = [manifest["executable"], "infer", "--model_file", stage["hbm"],
                            "--input_file", ",".join(stage["inputs"]),
                            "--frame_count", str(warmup + measured), "--thread_num", "1"]
        if command != expected_command:
            raise ValueError("unexpected measurement command or arguments")
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", action="append",
                        help="audit only complete named stages; the overall campaign may still be running")
    args = parser.parse_args()
    campaign = audit(args.campaign, args.smoke, args.stage)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = campaign["manifest"]
    stages = [s for s in manifest["stages"] if args.stage is None or s["name"] in args.stage]
    selected_names = {s["name"] for s in stages}
    selected_runs = [r for r in campaign["runs"] if r["stage"] in selected_names]
    reports = []
    for stage in stages:
        name = stage["name"]
        runs = sorted((r for r in campaign["runs"] if r["stage"] == name),
                      key=lambda r: int(r["label"].split("-")[-1]))
        command = [sys.executable, str(Path(__file__).with_name("summarize_hrt_latency.py")),
                   "--output-dir", str(args.output / name), "--warmup", str(manifest.get("warmup", 128))]
        for run in runs:
            command += ["--input", str(args.campaign / name / run["label"] / "stdout.log"),
                        "--label", run["label"]]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        (args.output / name / "summary-command.json").write_text(json.dumps({
            "command": command, "exit_code": completed.returncode}, indent=2) + "\n")
        summary = read_json(args.output / name / "summary.json")
        expected_count = manifest.get("processes", 5) * manifest.get("measured", 1024)
        if summary["pooled"]["samples"] != expected_count:
            raise ValueError("summarized sample count differs from protocol")
        pooled = summary["pooled"]
        reports.append({"stage": name, "scope": stage.get("scope", "single-stage"),
                        "performance_status": "measured", "hbm_sha256": stage["sha256"],
                        "samples": pooled["samples"],
                        **{k.replace("_ns", "_ms"): v / 1e6 for k, v in pooled.items() if k.endswith("_ns")},
                        "inverse_mean_stage_calls_per_s": 1e9 / pooled["mean_ns"],
                        "peak_process_rss_mib": max(r["peak_rss_mib"] for r in runs),
                        "model_load_ms_by_process": [r["model_load_ms"] for r in runs],
                        "copy_sync_timing": "not independently exposed by hrt_model_exec",
                        "summary": str((args.output / name / "summary.json").resolve()),
                        "cdf_csv": summary["cdf_csv"],
                        "fidelity_status": "separate limited check; no accuracy claim from timing"})
    report = {"schema": "vlaforge.hbm_campaign_summary/1", "status": "measured",
              "campaign_sha256": digest(args.campaign / "campaign.json"),
              "manifest_sha256": campaign["manifest_sha256"],
              "source_script_sha256": digest(Path(__file__)),
              "campaign_status_at_audit": campaign["status"],
              "whole_campaign_audited": args.stage is None,
              "audited_stages": sorted(selected_names),
              "stages": reports, "processes_sequential": True,
              "board_identity": {"serial_number": read_json(
                  args.campaign / selected_runs[0]["stage"] / selected_runs[0]["label"]
                  / "before.json")["files"]["/proc/device-tree/serial-number"]},
              "resource_evidence_by_process": [
                  {"stage": r["stage"], "label": r["label"],
                   "status": r.get("resource_status", "before-after-snapshots-only"),
                   "poll_interval_s": r.get("resource_poll_interval_s"),
                   "observations": r.get("resource_checks")}
                  for r in selected_runs],
              "rejected_processes": campaign.get("rejected_reuse_candidates", []),
              "reused_processes": [r["reused_from"] for r in selected_runs if "reused_from" in r],
              "outliers_removed": False, "full_chain_e2e": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output / "stage-performance.csv").open("w", newline="") as stream:
        columns = ["stage", "scope", "samples", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
                   "max_ms", "std_ms", "inverse_mean_stage_calls_per_s", "peak_process_rss_mib"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
