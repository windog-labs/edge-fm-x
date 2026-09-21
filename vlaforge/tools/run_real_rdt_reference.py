#!/usr/bin/env python3
"""Load the strict three-model RDT pipeline; optionally execute recorded AgileX I/O."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

from vlaforge.adapters.rdt.rdt_assets import file_identity
from vlaforge.adapters.rdt.rdt_reference import RDTConfig, load_rdt, official_agilex_reference, partition_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--numerical-profile", choices=("official-cu121", "torch210-cu128"), default="official-cu121")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--partition", action="store_true")
    parser.add_argument("--capture", action="store_true", help="capture all real online regions after saving reference outputs")
    parser.add_argument("--input-pack", type=Path)
    args = parser.parse_args()
    if not args.load_only and args.input_pack is None:
        parser.error("inference requires an explicitly prepared recorded input pack")
    if args.load_only and args.partition:
        parser.error("partition requires complete reference execution")
    if args.capture and (args.load_only or args.numerical_profile != "torch210-cu128"):
        parser.error("capture requires complete execution in the separate Torch 2.10 profile")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {
        "schema": "vlaforge.rdt_official_reference/1", "pid": os.getpid(),
        "status": "started", "phase": "strict_load", "device": args.device,
        "command": sys.argv, "tool": file_identity(Path(__file__)),
        "full_pipeline_executed": False, "no_python_deployment": "not-run",
        "capture": "not-run", "physical_action_units_verified": False,
        "process_environment": {
            name: os.environ.get(name) for name in (
                "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            )
        },
        "adapter_sources": {
            path.name: file_identity(path)
            for path in (Path(__file__).resolve().parents[1] / "python/vlaforge/adapters").glob("rdt_*.py")
        },
    }
    def save():
        report["elapsed_s"] = time.monotonic() - start
        report["max_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("status", "phase", "elapsed_s")}), flush=True)
    save()
    try:
        def stage(phase, provenance):
            report["phase"] = phase
            report["provenance"] = provenance
            save()
        loaded = load_rdt(RDTConfig(args.source_root, args.assets_root, args.device, args.numerical_profile), on_stage=stage)
        report["provenance"] = loaded.provenance
        report["phase"] = "strict_load_complete"
        report["status"] = "weights_loaded"
        save()
        if args.load_only:
            return
        import numpy as np
        import torch
        from PIL import Image

        from vlaforge.adapters.rdt.rdt_inputs import load_recorded_input
        report["runtime"] = {
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_maps": sorted({
                line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                if "/" in line and any(name in line for name in ("libcuda", "libcudnn", "libcublas"))
            }),
        }
        if args.device.startswith("cuda"):
            props = torch.cuda.get_device_properties(args.device)
            report["runtime"]["selected_gpu"] = {
                "name": props.name, "uuid": str(getattr(props, "uuid", "unavailable")),
                "total_memory": props.total_memory, "capability": [props.major, props.minor],
            }
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=index,uuid,driver_version,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ], capture_output=True, text=True, check=False)
            report["runtime"]["nvidia_smi"] = {
                "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
            }
        data, manifest = load_recorded_input(args.input_pack, source_root=args.source_root)
        manifest_path = args.input_pack / "manifest.json"
        report["input_manifest"] = file_identity(manifest_path)
        report["input_provenance"] = manifest
        images = [Image.fromarray(data[f"image_{index}"]) for index in range(6)]
        proprio = torch.from_numpy(data["proprio"].copy())
        report["phase"] = "official_multimodal_inference"
        report["status"] = "running"
        save()
        traces = official_agilex_reference(
            loaded, instruction=manifest["instruction"], images=images, proprio=proprio,
            control_frequency=manifest["control_frequency"], seed=manifest["seed"],
        )
        arrays = {}
        if args.partition:
            partition = partition_reference(loaded, traces)
            arrays["partitioned_unified_actions"] = partition["actions"].float().cpu().numpy()
            report["partition"] = {name: value for name, value in partition.items()
                                   if name not in ("actions", "samples", "model_outputs")}
            for index, sample in enumerate(partition["samples"]):
                arrays[f"partitioned_sample_{index}"] = sample.float().cpu().numpy()
        for name, value in traces.items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.bfloat16:
                    value = value.float()
                arrays[name] = value.detach().cpu().numpy()
            elif name in ("state_adaptor_inputs", "model_outputs"):
                for index, tensor in enumerate(value):
                    arrays[f"{name}_{index}"] = tensor.float().detach().cpu().numpy()
        np.savez_compressed(output / "full_reference.npz", **arrays)
        report["full_reference"] = file_identity(output / "full_reference.npz")
        report["tensor_shapes"] = {name: list(value.shape) for name, value in arrays.items()}
        report["reference_wall_seconds_including_capture_hooks"] = traces["reference_wall_seconds"]
        report["timing_is_benchmark"] = False
        report["full_pipeline_executed"] = True
        report["phase"] = "complete_official_reference"
        report["status"] = "passed_reference_execution_only"
        if args.partition and not report["partition"]["complete_chunk_exact"]:
            report["status"] = "failed_partition_parity"
            raise ValueError("partitioned full chunk differs from official reference; raw outputs preserved")
        save()
        if args.capture:
            from capture_real_rdt_regions import capture_real_regions
            report["phase"] = "capture_real_regions"
            save()
            report["capture_result"] = capture_real_regions(loaded, traces, output / "capture")
            report["capture"] = report["capture_result"]["status"]
            report["phase"] = "capture_complete"
            if report["capture"] != "captured_all_regions":
                report["status"] = "reference_passed_capture_incomplete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
