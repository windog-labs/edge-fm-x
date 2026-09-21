#!/usr/bin/env python3
"""Run an existing HBM stage campaign on a Linux Horizon board."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text(path):
    try:
        return Path(path).read_text(errors="replace").strip().strip("\x00")
    except OSError:
        return None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def snapshot():
    paths = [
        "/proc/device-tree/serial-number", "/proc/meminfo", "/proc/loadavg",
        "/sys/devices/system/bpu/ratio", "/sys/devices/system/bpu/users",
        "/sys/devices/system/bpu/bpu0/ratio",
        "/sys/devices/system/bpu/bpu0/users",
    ]
    for pattern in (
        "/sys/class/thermal/thermal_zone*/temp",
        "/sys/class/thermal/thermal_zone*/type",
        "/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq",
        "/sys/devices/system/cpu/cpufreq/policy*/scaling_governor",
        "/sys/class/devfreq/*/cur_freq",
    ):
        paths.extend(str(p) for p in Path("/").glob(pattern.lstrip("/")))
    return {
        "time_ns": time.time_ns(),
        "uname": list(os.uname()),
        "files": {p: read_text(p) for p in paths},
        "processes": subprocess.check_output(
            ["ps", "-eo", "pid,ppid,pcpu,pmem,args"], text=True
        ),
    }


def bpu_users(text):
    return {int(pid) for pid in re.findall(r"^\s*(\d+)\s+\d+(?:\.\d+)?\s*$", text, re.M)}


def registered_bpu_users():
    users = set()
    for path in ("/sys/devices/system/bpu/users", "/sys/devices/system/bpu/bpu0/users"):
        value = read_text(path)
        if value is None:
            raise RuntimeError(f"cannot inspect BPU clients: {path}")
        users.update(bpu_users(value))
    return users


def require_idle():
    ratio = read_text("/sys/devices/system/bpu/ratio")
    if ratio is None or float(ratio) != 0:
        raise RuntimeError(f"BPU idle check failed: ratio={ratio!r}")
    users = registered_bpu_users()
    if users:
        raise RuntimeError(f"other registered BPU clients: {sorted(users)}")
    active = subprocess.run(["pgrep", "-x", "hrt_model_exec"], capture_output=True)
    if active.returncode not in (0, 1):
        raise RuntimeError("cannot check existing hrt_model_exec processes")
    if active.returncode == 0:
        raise RuntimeError("another hrt_model_exec process is running")


def observe(command, directory, *, watch_bpu=False):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "before.json", snapshot())
    started = time.monotonic_ns()
    with (directory / "stdout.log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        record = {"command": command, "shell_command": shlex.join(command),
                  "pid": child.pid, "started_ns": time.time_ns(), "status": "running"}
        write_json(directory / "command.json", record)
        print(json.dumps({"event": "started", "path": str(directory), "pid": child.pid}), flush=True)
        sampled_hwm_kib = 0
        next_resource_check = 0
        resource_checks = 0
        foreign_users = set()
        resource_events = []
        resource_errors = []
        contention_stop_requested = False
        while True:
            if watch_bpu and time.monotonic() >= next_resource_check:
                resource_checks += 1
                try:
                    current_foreign = registered_bpu_users() - {child.pid}
                    if current_foreign - foreign_users:
                        resource_events.append({"time_ns": time.time_ns(),
                                                "foreign_bpu_users": sorted(current_foreign)})
                    foreign_users.update(current_foreign)
                    if current_foreign and not contention_stop_requested:
                        contention_stop_requested = True
                        try:
                            child.terminate()
                        except ProcessLookupError:
                            pass
                except RuntimeError as exc:
                    if str(exc) not in resource_errors:
                        resource_errors.append(str(exc))
                next_resource_check = time.monotonic() + 1
            status = read_text(f"/proc/{child.pid}/status") or ""
            match = re.search(r"^VmHWM:\s+(\d+) kB", status, re.M)
            if match:
                sampled_hwm_kib = max(sampled_hwm_kib, int(match.group(1)))
            # wait4 retains per-child peak RSS; cumulative RUSAGE_CHILDREN does not.
            pid, wait_status, usage = os.wait4(child.pid, os.WNOHANG)
            if pid:
                child.returncode = os.waitstatus_to_exitcode(wait_status)
                break
            time.sleep(0.25)
    record.update({
        "status": "completed" if child.returncode == 0 else "failed",
        "exit_code": child.returncode, "finished_ns": time.time_ns(),
        "command_wall_ns": time.monotonic_ns() - started,
        "peak_rss_kib": usage.ru_maxrss, "peak_rss_mib": usage.ru_maxrss / 1024,
        "sampled_vm_hwm_kib": sampled_hwm_kib,
        "user_cpu_s": usage.ru_utime, "system_cpu_s": usage.ru_stime,
        "wall_time_is_model_latency": False,
    })
    output = read_text(directory / "stdout.log") or ""
    loads = re.findall(r"Load model to DDR cost\s+([0-9.]+)ms", output)
    record["model_load_ms"] = [float(v) for v in loads]
    record["infer_records"] = len(re.findall(r"^Infer time:\s+[0-9.]+ ms", output, re.M))
    record["copy_and_sync_timing"] = "not separately exposed by hrt_model_exec"
    record["stdout_sha256"] = sha256(directory / "stdout.log")
    after = snapshot()
    if watch_bpu:
        for path, value in after.get("files", {}).items():
            if "/bpu/" in path and path.endswith("/users") and value is not None:
                foreign_users.update(bpu_users(value) - {child.pid})
        record.update(resource_status="clear-at-poll-points", resource_checks=resource_checks,
                      resource_poll_interval_s=1, foreign_bpu_users=sorted(foreign_users),
                      resource_events=resource_events, resource_errors=resource_errors,
                      contention_stop_requested=contention_stop_requested)
        if foreign_users or resource_errors:
            record["resource_status"] = "interference-detected" if foreign_users else "monitor-unavailable"
    write_json(directory / "command.json", record)
    write_json(directory / "after.json", after)
    print(json.dumps({"event": "finished", "path": str(directory),
                      "exit_code": child.returncode, "peak_rss_mib": record["peak_rss_mib"]}), flush=True)
    if watch_bpu and (foreign_users or resource_errors):
        raise RuntimeError(f"resource check failed; preserve but exclude this entire run: {directory}")
    if child.returncode:
        raise RuntimeError(f"command failed: {directory}")
    return record


def validate_manifest(manifest):
    if manifest.get("schema") != "vlaforge.hbm_performance_campaign/1":
        raise ValueError("unsupported manifest schema")
    names = [s["name"] for s in manifest["stages"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("stage names must be nonempty and unique")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("invalid stage name")
    if manifest.get("warmup", 128) < 0 or manifest.get("measured", 1024) < 1:
        raise ValueError("invalid sample counts")
    if manifest.get("processes", 5) < 1:
        raise ValueError("invalid process count")
    for stage in manifest["stages"]:
        if sha256(stage["hbm"]) != stage["sha256"]:
            raise ValueError(f"HBM hash mismatch: {stage['name']}")
        if not stage["inputs"]:
            raise ValueError("real input files are required")
        for path in stage["inputs"]:
            if not Path(path).is_file() or not Path(path).stat().st_size:
                raise ValueError(f"missing or empty input: {path}")


def reusable_record(root, count):
    if not (root / "command.json").exists():
        return None, "not executed"
    record = json.loads((root / "command.json").read_text())
    if record.get("status") != "completed" or record.get("exit_code") != 0:
        return None, "incomplete or failed process"
    if record.get("resource_status") in ("interference-detected", "monitor-unavailable"):
        return None, "resource interference or unavailable monitoring"
    if record.get("infer_records") != count:
        return None, "wrong inference count"
    if sha256(root / "stdout.log") != record["stdout_sha256"]:
        raise ValueError("reuse source log hash mismatch")
    for name in ("before.json", "after.json"):
        snapshot_value = json.loads((root / name).read_text())
        for path in ("/sys/devices/system/bpu/users", "/sys/devices/system/bpu/bpu0/users"):
            users = snapshot_value.get("files", {}).get(path)
            if users is None:
                return None, "missing BPU ownership snapshot"
            if bpu_users(users) - {record["pid"]}:
                return None, "foreign BPU client in source snapshot"
    return record, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--smoke-evidence", type=Path)
    parser.add_argument("--reuse-completed-from", type=Path,
                        help="reuse verified individual runs from a terminal formal campaign")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    if args.phase == "formal":
        if args.smoke_evidence is None:
            parser.error("formal sampling requires --smoke-evidence")
        smoke = json.loads((args.smoke_evidence / "campaign.json").read_text())
        if (smoke.get("status") != "completed" or smoke.get("phase") != "smoke"
                or smoke.get("manifest_sha256") != sha256(args.manifest)):
            raise ValueError("matching successful smoke evidence is required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": manifest["schema"], "phase": args.phase, "status": "running",
              "pid": os.getpid(), "started_ns": time.time_ns(),
              "manifest_sha256": sha256(args.manifest), "manifest": manifest,
              "script_sha256": sha256(__file__), "runs": [],
              "inputs": {p: {"sha256": sha256(p), "bytes": Path(p).stat().st_size}
                         for s in manifest["stages"] for p in s["inputs"]}}
    write_json(args.output / "campaign.json", report)
    try:
        reuse = None
        if args.reuse_completed_from:
            reuse = json.loads((args.reuse_completed_from / "campaign.json").read_text())
            if (args.phase != "formal" or reuse.get("phase") != "formal"
                    or reuse.get("status") not in ("completed", "failed")):
                raise ValueError("reuse requires a terminal formal campaign")
            if (reuse["manifest_sha256"] != report["manifest_sha256"]
                    or reuse["manifest"] != manifest or reuse["inputs"] != report["inputs"]):
                raise ValueError("reuse manifest or input hashes differ")
            report["reuse_source"] = {"path": str(args.reuse_completed_from),
                                      "campaign_sha256": sha256(args.reuse_completed_from / "campaign.json")}
            report["rejected_reuse_candidates"] = []
        for stage in manifest["stages"]:
            root = args.output / stage["name"]
            root.mkdir()
            base = [manifest["executable"], "--model_file", stage["hbm"]]
            inputs = ["--input_file", ",".join(stage["inputs"])]
            if args.phase == "smoke":
                commands = [
                    ("model-info", [base[0], "model_info", *base[1:]]),
                    ("infer", [base[0], "infer", *base[1:], *inputs,
                               "--frame_count", "1", "--enable_dump", "true",
                               "--dump_path", str(root / "dumps"),
                               "--remove_padding_process", "true"]),
                    ("perf", [base[0], "perf", *base[1:], *inputs,
                              "--frame_count", "5", "--thread_num", "1",
                              "--profile_path", str(root / "profile")]),
                ]
            else:
                count = manifest.get("warmup", 128) + manifest.get("measured", 1024)
                commands = [(f"run-{i + 1}", [base[0], "infer", *base[1:], *inputs,
                             "--frame_count", str(count), "--thread_num", "1"])
                            for i in range(manifest.get("processes", 5))]
            for label, command in commands:
                if reuse:
                    source_root = args.reuse_completed_from / stage["name"] / label
                    previous, reason = reusable_record(source_root, count)
                    if previous is not None:
                        if previous["command"] != command:
                            raise ValueError("reuse command differs from requested measurement")
                        shutil.copytree(source_root, root / label)
                        report["runs"].append({"stage": stage["name"], "label": label,
                                               "reused_from": str(source_root), **previous})
                        write_json(args.output / "campaign.json", report)
                        print(json.dumps({"event": "reused", "stage": stage["name"], "label": label}), flush=True)
                        continue
                    if reason != "not executed":
                        report["rejected_reuse_candidates"].append({"path": str(source_root), "reason": reason})
                        write_json(args.output / "campaign.json", report)
                time.sleep(2)
                require_idle()
                record = observe(command, root / label, watch_bpu=True)
                report["runs"].append({"stage": stage["name"], "label": label, **record})
                write_json(args.output / "campaign.json", report)
                if args.phase == "formal" and record["infer_records"] != count:
                    raise RuntimeError(f"wrong number of inference records: {stage['name']}/{label}")
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["finished_ns"] = time.time_ns()
        write_json(args.output / "campaign.json", report)


if __name__ == "__main__":
    main()
