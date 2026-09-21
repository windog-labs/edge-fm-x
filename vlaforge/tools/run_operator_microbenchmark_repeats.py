"""Run repeated, fail-closed benchmarks for one captured operator example."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cogact_gpu_monitor import run_monitored


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_snapshot(gpu: str) -> dict:
    owners = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True
    )
    device = subprocess.check_output(
        ["nvidia-smi", "-i", gpu,
         "--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu,clocks.sm",
         "--format=csv,noheader,nounits"], text=True
    )
    devices = list(csv.reader(device.splitlines(), skipinitialspace=True))
    if len(devices) != 1 or devices[0][0] != gpu or not gpu.startswith("GPU-"):
        raise ValueError("requested GPU UUID is not present on this host")
    rows = []
    for row in csv.reader(io.StringIO(owners)):
        if row and row[0].strip() == gpu:
            rows.append({"gpu": row[0].strip(), "pid": int(row[1]),
                         "name": row[2].strip(), "memory_mib": int(row[3])})
    return {"at": datetime.now(timezone.utc).isoformat(),
            "device": device.strip(), "owners": rows}


def own_process(pid: int, ancestor: int) -> bool:
    while pid > 1:
        if pid == ancestor:
            return True
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except (FileNotFoundError, PermissionError):
            return False
        pid = int(next(line.split(":", 1)[1] for line in status.splitlines()
                       if line.startswith("PPid:")))
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--compiled-report", type=Path)
    parser.add_argument("--recipe", choices=("inductor-aten", "inductor-autotune", "inductor-aten-preserving"),
                        default="inductor-aten")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rep-ms", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.rep_ms <= 100:
        parser.error("repeats must be positive and rep-ms must be in [1,100]")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "vlaforge.operator_repeat_campaign/2",
        "status": "running",
        "pid": os.getpid(),
        "gpu_uuid": args.gpu,
        "node": args.node,
        "repeats": args.repeats,
        "rep_ms": args.rep_ms,
        "tool_sha256": sha(args.tool),
        "examples_sha256": sha(args.examples),
        "compiled_report_sha256": sha(args.compiled_report) if args.compiled_report else None,
        "candidate_recipe": args.recipe,
        "driver_sha256": sha(Path(__file__)),
        "source_files": {str(p): sha(p) for p in sorted((args.tool.parent.parent / "python").rglob("*.py"))
                         if "__pycache__" not in p.parts},
        "monitor_source_sha256": sha(Path(__file__).with_name("cogact_gpu_monitor.py")),
        "ownership_mode": "register-reset-register-before-torch",
        "scope": "repeated same-platform operator microbenchmark; no E2E selection",
        "runs": [],
        "selected_for_deployment": False,
        "end_to_end_integrated": False,
    }

    def save() -> None:
        (args.output / "campaign.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    save()
    recipes = ("aten", args.recipe)

    def verify_sources():
        if (sha(Path(__file__)) != report["driver_sha256"]
                or sha(Path(__file__).with_name("cogact_gpu_monitor.py")) != report["monitor_source_sha256"]):
            raise ValueError("controller or monitor implementation changed")
        if sha(args.tool) != report["tool_sha256"] or sha(args.examples) != report["examples_sha256"]:
            raise ValueError("tool or example source changed")
        if any(sha(Path(path)) != digest for path, digest in report["source_files"].items()):
            raise ValueError("operator implementation changed")

    def run_one(recipe, output, *, repeat=None, reuse=None):
        verify_sources()
        before = gpu_snapshot(args.gpu)
        if before["owners"]:
            raise RuntimeError(f"GPU occupied before {recipe}: {before}")
        command = [sys.executable, str(args.tool), "--examples", str(args.examples), "--node", args.node,
                   "--recipe", recipe, "--reference-mode", "revalidate-workload", "--rep-ms", str(args.rep_ms),
                   "--output", str(output)]
        if reuse is not None:
            command.extend(["--reuse-compiled-from", str(reuse)])
        entry = {"recipe": recipe, "repeat": repeat, "command": command, "preflight": before,
                 "disk_free_bytes": shutil.disk_usage(args.output).free, "start_ns": time.time_ns()}
        if repeat is None:
            report["qualification"] = entry
        else:
            report["runs"].append(entry)
        save()
        folder = args.output / ("monitor-" + output.name)
        outcome = run_monitored(command, folder, gpu=args.gpu, environment={
            "CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONDONTWRITEBYTECODE": "1"})
        monitor_path = folder / "monitor.json"
        monitor = json.loads(monitor_path.read_text())
        entry.update(pid=monitor["container_pid"], nvml_pid=monitor["nvml_pid"], exit_code=outcome.returncode,
                     monitor=str(monitor_path), monitor_sha256=sha(monitor_path), end_ns=time.time_ns())
        result = output / "report.json"
        if not result.is_file():
            raise RuntimeError(f"operator report missing: {result}")
        data = json.loads(result.read_text())
        entry.update(status=data.get("status"), report_sha256=sha(result),
                     correctness_passed=data.get("correctness_passed"),
                     source_reference_bitwise_equal=data.get("source_reference_bitwise_equal"),
                     target_reference_identity=data.get("target_reference_identity"), measurement=data.get("measurement"))
        save()
        if (outcome.returncode != 0 or monitor["status"] != "exited" or monitor["exitcode"] != 0
                or entry["status"] != "measured" or not entry["correctness_passed"] or data["recipe"] != recipe):
            raise RuntimeError("operator run failed its numerical or execution gate")
        verify_sources()
        return result, entry

    try:
        compiled, qualified = run_one(args.recipe, args.output / "qualification", reuse=args.compiled_report)
        reuse_source = args.compiled_report or compiled
        for repeat in range(args.repeats):
            ordered = recipes if repeat % 2 == 0 else tuple(reversed(recipes))
            for recipe in ordered:
                output = args.output / f"{recipe}-{repeat:02d}"
                _, entry = run_one(recipe, output, repeat=repeat,
                                   reuse=reuse_source if recipe != "aten" else None)
                if entry["target_reference_identity"] != qualified["target_reference_identity"]:
                    raise ValueError("baseline and candidate target references differ")
        if len({entry["pid"] for entry in report["runs"]}) != 2 * args.repeats:
            raise ValueError("independent process identities were reused")
        report["summary"] = {
            recipe: {
                "process_medians_ms": [
                    run["measurement"]["median_ms"] for run in report["runs"]
                    if run["recipe"] == recipe
                ],
                "mean_process_median_ms": statistics.mean(
                    run["measurement"]["median_ms"] for run in report["runs"]
                    if run["recipe"] == recipe
                ),
            } for recipe in recipes
        }
        report["status"] = "passed"
        save()
        return 0
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        save()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
