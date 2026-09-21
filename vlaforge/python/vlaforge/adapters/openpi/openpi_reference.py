"""Real-checkpoint OpenPI reference with frozen observations and explicit scope."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import re
import resource
import time
from dataclasses import asdict
from pathlib import Path

from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.adapters.openpi.openpi_frontend import (
    OpenPIConfig,
    build_openpi_frontend,
    load_openpi,
    prepare_openpi_inputs,
)
from vlaforge.adapters.openpi.openpi_inputs import load_openpi_input_pack
from vlaforge.numerical_context import snapshot
from vlaforge.validation.contracts import NumericContract
from vlaforge.validation.deployment_metrics import compare_action_chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--partition", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--capture-save-only", action="store_true")
    parser.add_argument("--absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--relative-tolerance", type=float, default=0.0)
    args = parser.parse_args()
    if re.fullmatch(r"cpu|cuda:[0-9]+", args.device) is None:
        parser.error("device must be cpu or an explicit cuda:N visible-device index")
    if args.prepare_only and args.partition:
        parser.error("prepare-only cannot run the partitioned model")
    if (args.capture or args.capture_save_only) and not args.partition:
        parser.error("capture requires the full partition parity gate")
    if args.capture and args.capture_save_only:
        parser.error("select either same-worker capture/reload or capture-save-only")
    config = OpenPIConfig(
        source_root=args.source_root,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_sha256=args.checkpoint_sha256,
        config_name=args.config_name,
        device=args.device,
        num_steps=args.num_steps,
    )
    contract = NumericContract(
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    # Validate tolerances before allocating a checkpoint or running any model.
    if any(
        not math.isfinite(value) or value < 0
        for value in (args.absolute_tolerance, args.relative_tolerance)
    ):
        parser.error("numeric tolerances must be finite and non-negative")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "vlaforge.openpi_reference/2",
        "pid": os.getpid(),
        "status": "started",
        "device": args.device,
        "config_name": args.config_name,
        "num_steps": args.num_steps,
        "prepare_only": args.prepare_only,
        "partition_requested": args.partition,
        "tolerances": asdict(contract),
        "no_python_deployment": "not-run",
        "capture": "not-run",
        "action_space_definition": "official output transform, native ALOHA dataset action scale; not robot-calibrated physical validation",
        "physical_units_verified": False,
        "robot_calibration_verified": False,
        "adapter_sources": {
            path.name: file_digest(path)
            for path in Path(__file__).parent.glob("openpi*.py")
        },
        "execution_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "jax", "numpy", "safetensors")
        },
        "processor_config": {
            "source_root": str(args.source_root.resolve()),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        },
    }
    from vlaforge.adapters.openpi.openpi_phased import process_identity

    report["process_identity"] = process_identity()
    start = time.monotonic()
    device = None

    def save():
        report["wall_time_seconds"] = time.monotonic() - start
        report["max_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if device is not None and device.type == "cuda":
            report["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            report["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(
                device
            )
            report["cuda_memory_peak_scope"] = (
                "complete worker through current phase, including capture when requested"
            )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        import numpy as np
        import torch

        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            report["cuda_device"] = {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "build_version": torch.version.cuda,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }

        observation, noise, manifest = load_openpi_input_pack(
            args.input_manifest, source_root=args.source_root
        )
        report["input_manifest_digest"] = file_digest(args.input_manifest)
        report["input_provenance"] = manifest
        report["phase"] = "strict_policy_load"
        save()
        loaded = load_openpi(config)
        numerical_context = snapshot()
        report["numerical_context"] = numerical_context.to_dict()
        report["numerical_context_scope"] = (
            "observed and required in Python reference; not C++ enforcement"
        )
        if any(parameter.device != device for parameter in loaded.model.parameters()):
            raise ValueError(
                "reference allocated a parameter outside the selected device"
            )
        report["checkpoint_provenance"] = loaded.provenance
        prepared = prepare_openpi_inputs(loaded, observation, noise=noise)
        prepared_arrays = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in prepared.tensors.items()
        }
        np.savez(output / "prepared_inputs.npz", **prepared_arrays)
        report["prepared_inputs"] = file_digest(output / "prepared_inputs.npz")
        report["official_image_memory_formats"] = list(prepared.image_memory_formats)
        report["prepared_shapes"] = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in prepared_arrays.items()
        }
        if args.prepare_only:
            report["status"] = "prepared"
            report["phase"] = "prepared_inputs"
            report["evidence_level"] = (
                "real-checkpoint-and-recorded-input-preparation; no action inference"
            )
            save()
            return
        report["phase"] = (
            "official_and_partitioned_actions" if args.partition else "official_actions"
        )
        save()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_start = time.monotonic()
        with torch.inference_mode():
            if args.partition:
                frontend = build_openpi_frontend(
                    loaded,
                    prepared,
                    absolute_tolerance=args.absolute_tolerance,
                    relative_tolerance=args.relative_tolerance,
                )
                normalized = frontend.normalized_reference
                candidate = frontend.normalized_partitioned
            else:
                normalized = loaded.model.sample_actions(
                    args.device,
                    prepared.observation,
                    noise=prepared.tensors["noise"].clone(),
                    num_steps=args.num_steps,
                )
                candidate = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            report["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            report["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(
                device
            )
        report["action_audit_wall_seconds"] = time.monotonic() - inference_start
        numerical_context.require_current()
        if not torch.isfinite(normalized).all():
            raise ValueError("official full action chunk contains non-finite values")

        def physical(value):
            return loaded.policy._output_transform(
                {
                    "state": prepared.observation.state[0].detach().cpu().numpy(),
                    "actions": value[0].detach().cpu().numpy(),
                }
            )["actions"]

        physical_reference = physical(normalized)
        arrays = {
            "normalized_reference": normalized.detach().cpu().numpy(),
            "physical_reference": physical_reference,
        }
        if not np.isfinite(physical_reference).all():
            raise ValueError("physical action chunk contains non-finite values")
        if candidate is not None:
            physical_candidate = physical(candidate)
            arrays.update(
                normalized_partitioned=candidate.detach().cpu().numpy(),
                physical_partitioned=physical_candidate,
            )
            report["fidelity"] = [
                compare_action_chunk(
                    normalized,
                    candidate,
                    sample_id="recorded-frame",
                    space="normalized",
                    contract=contract,
                ),
                compare_action_chunk(
                    physical_reference,
                    physical_candidate,
                    sample_id="recorded-frame",
                    space="native-aloha-action-scale",
                    contract=contract,
                ),
            ]
            if not all(
                item["metrics"]["within_tolerance"] for item in report["fidelity"]
            ):
                raise ValueError(
                    "full action chunk failed the frozen tolerance contract"
                )
        np.savez(output / "actions.npz", **arrays)
        report["actions"] = file_digest(output / "actions.npz")
        if args.capture_save_only:
            from vlaforge.adapters.openpi.openpi_phased import capture_and_persist_openpi

            report["phase"] = "capture_save_only"

            def persist_progress(value):
                report["capture"] = value
                save()

            captured = capture_and_persist_openpi(
                frontend,
                output / "exported_regions",
                contract=contract,
                on_progress=persist_progress,
            )
            if captured["status"] != "persisted_awaiting_independent_reload":
                raise ValueError("actual region capture/persistence did not pass")
            report["status"] = "persisted"
            report["evidence_level"] = (
                "real-reference-and-partition-passed; saved-region-independent-reload-pending"
            )
            save()
            return
        if args.capture:
            from vlaforge.adapters.openpi.openpi_capture import capture_and_reload_openpi

            report["phase"] = "capture_save_reload"

            def capture_progress(value):
                report["capture"] = value
                save()

            captured, reloaded = capture_and_reload_openpi(
                frontend,
                prepared,
                output / "exported_regions",
                contract=contract,
                on_progress=capture_progress,
            )
            if captured["status"] != "passed":
                raise ValueError(
                    "actual region capture/save/reload invocation did not pass"
                )
            arrays["normalized_reloaded"] = reloaded.detach().cpu().numpy()
            arrays["native_action_scale_reloaded"] = physical(reloaded)
            report["reloaded_native_scale_fidelity"] = compare_action_chunk(
                physical_reference,
                arrays["native_action_scale_reloaded"],
                sample_id="recorded-frame",
                space="native-aloha-action-scale",
                contract=contract,
            )
            np.savez(output / "actions.npz", **arrays)
            report["actions"] = file_digest(output / "actions.npz")
            if not report["reloaded_native_scale_fidelity"]["metrics"][
                "within_tolerance"
            ]:
                raise ValueError("saved-reloaded native-scale full action chunk failed")
        report["status"] = "passed"
        report["evidence_level"] = (
            f"real-checkpoint-recorded-input-{device.type}-partition-parity"
            if args.partition
            else f"real-checkpoint-recorded-input-{device.type}-reference-only"
        )
        if args.capture:
            report["evidence_level"] = (
                f"real-checkpoint-recorded-input-{device.type}-export-save-reload-full-invocation-parity"
            )
        save()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise


if __name__ == "__main__":
    main()
