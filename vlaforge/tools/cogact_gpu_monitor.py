"""Isolated worker ownership gate, before any tensor or Session construction."""

import csv
import ctypes
import ctypes.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def owners(gpu="0"):
    result = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,process_name,used_gpu_memory",
                                      "--format=csv,noheader,nounits"], text=True)
    return [{"pid": int(row[0]), "name": row[1], "memory_mib": row[2]}
            for row in csv.reader(result.splitlines(), skipinitialspace=True) if row and row[0].strip().isdigit()]


def child_handshake():
    root = os.environ.get("COGACT_GPU_OWNER_FOLDER")
    if root is None:
        return
    if "torch" in sys.modules:
        raise RuntimeError("owner registration/reset must precede Torch import")
    library = ctypes.util.find_library("cudart")
    candidates = [Path(sys.prefix) / "lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"]
    runtime = ctypes.CDLL(library or str(next(path for path in candidates if path.is_file())))
    runtime.cudaFree.argtypes = [ctypes.c_void_p]

    def checked(code):
        if code != 0:
            raise RuntimeError(f"CUDA ownership bootstrap failed: {code}")

    checked(runtime.cudaSetDevice(0))
    for stage in ("registered-1", "reset", "registered-2"):
        checked(runtime.cudaDeviceReset() if stage == "reset" else runtime.cudaFree(None))
        folder = Path(root)
        write(folder / f"owner-{stage}.json", {"pid": os.getpid(), "ordinal": 0})
        deadline = time.monotonic() + 40
        while not (folder / f"continue-{stage}").exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("GPU owner handshake acknowledgement timed out")
            time.sleep(0.1)


def run_monitored(command, folder, *, environment=None, gpu="0"):
    folder.mkdir(parents=True, exist_ok=False)
    initial = owners(gpu)
    write(folder / "preflight-owners.json", initial)
    if initial:
        raise RuntimeError(f"GPU {gpu} compute owner present; refusing worker launch")
    report = {"command": command, "observations": [], "status": "incomplete", "identity_is_nspid_verified": False,
              "monitored_gpu": str(gpu)}
    with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True,
            env={**os.environ, **(environment or {}), "COGACT_GPU_OWNER_FOLDER": str(folder)})
        report["container_pid"] = process.pid
        candidate = None
        try:
            for stage in ("registered-1", "reset", "registered-2"):
                deadline = time.monotonic() + 40
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("worker exited before ownership registration")
                    current = owners(gpu)
                    marker = folder / f"owner-{stage}.json"
                    ready = json.loads(marker.read_text()) if marker.exists() else None
                    report["observations"].append({"stage": stage, "time_ns": time.time_ns(), "owners": current, "ready": ready})
                    if len(current) > 1 or (candidate is not None and any(row["pid"] != candidate for row in current)):
                        raise RuntimeError("foreign GPU owner during registration")
                    if ready is not None:
                        if ready != {"pid": process.pid, "ordinal": 0}:
                            raise RuntimeError("ownership marker does not identify our child/device")
                        if (stage == "reset" and not current) or (stage != "reset" and len(current) == 1):
                            if candidate is None:
                                candidate = current[0]["pid"]
                            (folder / f"continue-{stage}").write_text("continue\n")
                            break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("owner registration visibility timed out")
                    time.sleep(0.1)
            report["nvml_pid"] = candidate
            report["identity_evidence"] = "idle preflight, child context register/reset/re-register before Torch/tensors/Session"
            while process.poll() is None:
                current = owners(gpu)
                report["observations"].append({"stage": "running", "time_ns": time.time_ns(), "owners": current})
                if any(row["pid"] != candidate for row in current):
                    raise RuntimeError("foreign GPU owner appeared; terminating only our worker process group")
                write(folder / "monitor.json", report)
                time.sleep(1)
            report.update(status="exited", exitcode=process.returncode)
        except BaseException as error:
            report.update(status="failed", error=str(error))
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            report["exitcode"] = process.returncode
            raise
        finally:
            write(folder / "monitor.json", report)
    return subprocess.CompletedProcess(command, process.returncode,
        stdout=(folder / "stdout.log").read_text(), stderr=(folder / "stderr.log").read_text())
