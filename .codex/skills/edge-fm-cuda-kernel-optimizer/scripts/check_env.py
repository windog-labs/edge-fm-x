#!/usr/bin/env python3
"""Detect local GPU / CUDA / Triton / CUTLASS environment.

Writes a JSON snapshot consumed by later steps. Safe to run even if some
tools are missing — fields are simply null.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="ignore",
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        return -1, "", str(e)


def _detect_gpus() -> list[dict]:
    gpus: list[dict] = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                gpus.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "compute_capability": f"{major}.{minor}",
                    "sm_arch": f"sm_{major}{minor}",
                    "total_memory_mb": torch.cuda.get_device_properties(i).total_memory // (1024 * 1024),
                })
    except Exception as e:
        return [{"error": f"torch probe failed: {e}"}]
    return gpus


def _detect_nvcc() -> dict:
    path = shutil.which("nvcc")
    if not path:
        return {"available": False, "path": None, "version": None}
    rc, out, _ = _run([path, "--version"])
    version = None
    if rc == 0:
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Cuda compilation tools"):
                version = line
                break
    return {"available": True, "path": path, "version": version or out.strip()}


def _detect_ncu() -> dict:
    path = shutil.which("ncu")
    if not path:
        return {
            "available": False,
            "path": None,
            "version": None,
            "query_metrics_available": None,
            "profiling_admin_only": None,
            "can_read_counters": None,
            "note": None,
        }
    rc, out, _ = _run([path, "--version"])
    version = out.strip().splitlines()[0] if out else None
    # --query-metrics can succeed even when actual profiling counters are
    # restricted by the NVIDIA kernel module. Record both facts separately.
    rc2, _, err2 = _run([path, "--query-metrics"], timeout=5)
    query_ok = rc2 == 0
    profiling_admin_only = _detect_nvidia_profiling_admin_only()
    if profiling_admin_only is None:
        can_read = query_ok
        note = None if query_ok else (
            err2.strip()[:400] or "ncu query failed — perf counters may require elevated permissions"
        )
    elif profiling_admin_only:
        can_read = False
        note = (
            "ncu --query-metrics may succeed, but the active NVIDIA driver has "
            "RmProfilingAdminOnly=1; non-root profiling will fail with ERR_NVGPUCTRPERM"
        )
    else:
        can_read = query_ok
        note = None if query_ok else (
            err2.strip()[:400] or "ncu query failed despite RmProfilingAdminOnly=0"
        )
    return {
        "available": True,
        "path": path,
        "version": version,
        "query_metrics_available": query_ok,
        "profiling_admin_only": profiling_admin_only,
        "can_read_counters": can_read,
        "note": note,
    }


def _detect_nvidia_profiling_admin_only() -> bool | None:
    params_path = Path("/proc/driver/nvidia/params")
    try:
        for line in params_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.startswith("RmProfilingAdminOnly:"):
                continue
            value = line.split(":", 1)[1].strip().lower()
            if value in {"0", "n", "no", "false"}:
                return False
            if value in {"1", "y", "yes", "true"}:
                return True
    except OSError:
        return None
    return None


def _detect_driver() -> dict:
    path = shutil.which("nvidia-smi")
    if not path:
        return {"available": False}
    rc, out, _ = _run([path, "--query-gpu=driver_version,cuda_version", "--format=csv,noheader"])
    if rc != 0:
        return {"available": True, "raw": None}
    return {"available": True, "raw": out.strip()}


def _detect_cutlass() -> dict:
    # Mirrors benchmark.py's find_cutlass_include_dir
    candidates: list[str] = []
    for var in ("CUTLASS_PATH", "CUTLASS_INCLUDE_DIR"):
        v = os.environ.get(var, "").strip()
        if v:
            candidates.append(v)
            candidates.append(os.path.join(v, "include"))
    candidates.extend(sorted(glob.glob("/usr/local/cutlass*/include")))
    candidates.extend(["/usr/local/cutlass/include", "/opt/cutlass/include"])
    seen = set()
    for c in candidates:
        if not c:
            continue
        r = os.path.abspath(c)
        if r in seen:
            continue
        seen.add(r)
        if os.path.isdir(os.path.join(r, "cutlass")) and os.path.isdir(os.path.join(r, "cute")):
            return {"available": True, "include_dir": r}
    return {"available": False, "include_dir": None}


def _detect_python_libs() -> dict:
    libs: dict = {}
    for name in ("torch", "triton", "cutlass"):  # cutlass python if any
        try:
            mod = __import__(name)
            libs[name] = {"available": True, "version": getattr(mod, "__version__", "unknown")}
        except Exception:
            libs[name] = {"available": False, "version": None}
    return libs


def collect_env() -> dict:
    gpus = _detect_gpus()
    primary_arch = None
    for g in gpus:
        if "sm_arch" in g:
            primary_arch = g["sm_arch"]
            break
    return {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "gpus": gpus,
        "primary_sm_arch": primary_arch,
        "nvcc": _detect_nvcc(),
        "ncu": _detect_ncu(),
        "driver": _detect_driver(),
        "cutlass": _detect_cutlass(),
        "libs": _detect_python_libs(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="./env.json", help="Output JSON path")
    args = p.parse_args()

    env = collect_env()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)

    # Also print a compact summary to stdout so Codex / the user can eyeball it
    print(json.dumps({
        "gpu": env["gpus"][0].get("name") if env["gpus"] else None,
        "sm_arch": env["primary_sm_arch"],
        "nvcc": env["nvcc"].get("version"),
        "ncu": env["ncu"].get("available"),
        "ncu_can_read_counters": env["ncu"].get("can_read_counters"),
        "cutlass": env["cutlass"].get("available"),
        "torch": env["libs"].get("torch", {}).get("version"),
        "triton": env["libs"].get("triton", {}).get("version"),
        "out": args.out,
    }, indent=2))

    # Useful for callers: exit 0 regardless — env is informational, not a gate
    sys.exit(0)


if __name__ == "__main__":
    main()
