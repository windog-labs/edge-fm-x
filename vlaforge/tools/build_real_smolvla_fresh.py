#!/usr/bin/env python3
"""Audit real explicit-input fresh chunks through capture, AOTI and C++ Session."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

SOURCE = Path(__file__).resolve().parents[1]
REPOSITORY = SOURCE.parent
sys.path.insert(0, str(SOURCE / "python"))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_input_pack(path: Path, *, processing_mode: str) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("schema") != "vlaforge.smolvla_observations/2":
        raise ValueError(
            "fresh deployment requires the explicit processing contract in input pack v2"
        )
    if manifest.get("status") != "materialized_tensor_boundary_only":
        raise ValueError("input pack must declare its tensor-only measurement scope")
    normalization = manifest.get("normalization", {})
    if normalization.get("mode") != processing_mode:
        raise ValueError("requested processing mode differs from the input pack")
    verified = normalization.get("normalization_statistics_verified")
    if processing_mode == "published-compatibility" and verified is not False:
        raise ValueError("published compatibility cannot imply verified normalization")
    if processing_mode == "strict-statistics" and verified is not True:
        raise ValueError(
            "strict statistics requires an explicit successful consumption gate"
        )
    if processing_mode not in {"published-compatibility", "strict-statistics"}:
        raise ValueError("unknown explicit processing mode")
    if (
        not manifest.get("records")
        or manifest.get("noise_semantics") != "saved_float32_tensor_is_authoritative"
    ):
        raise ValueError(
            "input pack requires actual records and authoritative saved noise"
        )
    root = path.parent.resolve()
    for record in manifest["records"]:
        if (
            processing_mode == "strict-statistics"
            and (record.get("state_normalization") or {}).get("exact_formula_match")
            is not True
        ):
            raise ValueError(
                "strict input record lacks an actual state-transform check"
            )
        source = (root / record["path"]).resolve(strict=True)
        if not source.is_relative_to(root) or digest(source) != record["sha256"]:
            raise ValueError("input record path or digest does not match its pack")
    return manifest


def load_observation(path: Path, record: dict[str, Any]):
    import numpy as np

    with np.load(path, allow_pickle=False) as pack:
        values = {name: pack[name].copy() for name in pack.files}
    if set(values) != set(record["tensors"]):
        raise ValueError(
            "saved observation keys differ from the declared tensor profile"
        )
    for name, value in values.items():
        metadata = record["tensors"][name]
        if (
            list(value.shape) != metadata["shape"]
            or str(value.dtype) != metadata["dtype"]
        ):
            raise ValueError(f"saved tensor profile changed: {name}")
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"saved tensor must be finite and nonempty: {name}")
    if hashlib.sha256(values["noise"].tobytes()).hexdigest() != record["noise_sha256"]:
        raise ValueError("authoritative noise digest differs")
    return values


def fidelity(
    reference: Any,
    actual: Any,
    *,
    sample_id: str,
    space: str,
    max_abs: float,
    mean_abs: float,
    tensor_tolerances: bool = True,
) -> dict[str, Any]:
    import numpy as np
    from vlaforge.validation.contracts import NumericContract
    from vlaforge.validation.deployment_metrics import compare_action_chunk

    report = compare_action_chunk(
        reference, actual, sample_id=sample_id, space=space, contract=NumericContract()
    )
    metrics = report["metrics"]
    cosine = metrics["cosine_similarity"]
    report["gates"] = {
        "technical_tolerance": {
            "maximum_absolute_error_limit": max_abs,
            "mean_absolute_error_limit": mean_abs,
            "passed": metrics["maximum_absolute_error"] <= max_abs
            and metrics["mean_absolute_error"] <= mean_abs,
        },
        "paper_numerics": {
            "mean_squared_error_limit": 1e-5,
            "cosine_similarity_minimum": 0.9999,
            "passed": metrics["mean_squared_error"] <= 1e-5
            and cosine is not None
            and cosine >= 0.9999,
        },
        "bitwise_equal": reference.dtype == actual.dtype
        and reference.shape == actual.shape
        and np.array_equal(reference.view(np.uint8), actual.view(np.uint8)),
        "full_paper_acceptance": False,
    }
    if not tensor_tolerances:
        for gate in ("technical_tolerance", "paper_numerics"):
            report["gates"][gate] = {
                "applicable": False,
                "passed": None,
                "reason": "tensor-output tolerances do not transfer to rescaled action units",
            }
    return report


def log_stage(root: Path, stage: str, **values):
    path = root / "report.json"
    report = (
        read_json(path)
        if path.is_file()
        else {"schema": "vlaforge.smolvla_fresh_deployment/1", "stages": []}
    )
    report["stages"].append({"stage": stage, "time_ns": time.time_ns(), **values})
    report["full_paper_acceptance"] = False
    report["physical_action_units_verified"] = False
    write_json(path, report)
    print(stage, json.dumps(values, allow_nan=False), flush=True)


def save_inventory(program, root: Path, name: str):
    path = root / "exports" / f"{name}.operators.json"
    try:
        from vlaforge.analysis.operator_inventory import exported_operator_inventory

        payload = exported_operator_inventory(program, region_name=name)
        write_json(path, payload)
        return {"status": "recorded_static_graph_only", "sha256": digest(path)}
    except Exception as error:
        log_stage(root, "operator_inventory_failed", region=name, message=str(error))
        return {"status": "failed", "error": str(error)}


def verify_policy_provenance(policy: Path, manifest: dict[str, Any]) -> None:
    recovery = manifest.get("statistics_recovery")
    if recovery is not None:
        from vlaforge.adapters.smolvla.smolvla_migration import verify_recovered_profile

        if (
            verify_recovered_profile(policy, robot_type=recovery["robot_type"])
            != recovery
        ):
            raise ValueError(
                "recovered policy differs from input statistics provenance"
            )
    if digest(policy / "model.safetensors") != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from the input protocol")
    for name, sha in manifest["processor_files"].items():
        path = (policy / name).resolve(strict=True)
        if not path.is_relative_to(policy.resolve()) or digest(path) != sha:
            raise ValueError("processor files differ from the input protocol")


def capture(args):
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from vlaforge.adapters.smolvla.smolvla_fresh import build_smolvla_fresh_program
    from vlaforge.frontend import capture_region, save_exported_region
    from vlaforge.ir.serializer import canonical_json

    root = args.output
    if (root / "capture.json").exists() or (root / "exports").exists():
        raise ValueError(
            "capture output must be fresh; partial captures are not reusable"
        )
    manifest = verify_input_pack(args.input_pack, processing_mode=args.processing_mode)
    verify_policy_provenance(args.policy_path, manifest)
    selected = manifest["records"][args.first : args.first + args.count]
    if len(selected) != args.count:
        raise ValueError("requested observations exceed the verified input pack")
    config = PreTrainedConfig.from_pretrained(args.policy_path, local_files_only=True)
    config.vlm_model_name = str(args.vlm_path.resolve())
    config.device = args.device
    config.num_steps = args.num_steps
    policy = SmolVLAPolicy.from_pretrained(
        args.policy_path, config=config, local_files_only=True, strict=True
    ).eval()
    loaded = []
    for record in selected:
        arrays = load_observation(args.input_pack.parent / record["path"], record)
        values = {
            name: torch.from_numpy(value).to(args.device)
            for name, value in arrays.items()
        }
        noise = values.pop("noise")
        with torch.inference_mode():
            reference = policy.predict_action_chunk(
                {key: value.clone() for key, value in values.items()},
                noise=noise.clone(),
            )
        if not bool(torch.isfinite(reference).all()):
            raise ValueError(
                "official complete reference chunk contains nonfinite values"
            )
        loaded.append((values, noise, reference.detach().cpu().contiguous().numpy()))
    built = build_smolvla_fresh_program(
        policy, loaded[0][0], loaded[0][1], specialize_fixed_vision=True
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "module.json").write_text(
        canonical_json(built.program.module, indent=2) + "\n"
    )
    records = []
    for index, ((batch, noise, reference), source) in enumerate(
        zip(loaded, selected, strict=True)
    ):
        built.bind_inputs(batch, noise)
        folder = root / "inputs" / f"sample-{index:06d}"
        folder.mkdir(parents=True)
        tensors = {name: batch[key] for name, key in built.batch_keys.items()}
        tensors["noise"] = noise
        raw = {
            name: value.detach().cpu().contiguous().numpy()
            for name, value in tensors.items()
        }
        np.savez(folder / "inputs.npz", **raw)
        np.save(folder / "reference.npy", reference, allow_pickle=False)
        for name, value in raw.items():
            (folder / f"{name}.bin").write_bytes(value.tobytes())
        records.append(
            {
                "index": index,
                "source_record": source,
                "path": str(folder.relative_to(root)),
                "inputs_sha256": digest(folder / "inputs.npz"),
                "reference_sha256": digest(folder / "reference.npy"),
            }
        )
    exports = root / "exports"
    exports.mkdir()
    regions = []
    for region in built.program.module.regions:
        log_stage(root, "capture_start", region=region.name)
        outcome = capture_region(
            region,
            built.program.regions[region.name],
            built.region_examples[region.name],
            strict=False,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
        ).require_supported()
        path = exports / f"{region.name}.pt2e"
        evidence = exports / f"{region.name}.capture.json"
        save_exported_region(outcome, program_path=path, evidence_path=evidence)
        example_path = exports / f"{region.name}.capture-inputs.pt"
        torch.save(
            tuple(
                value.detach().cpu().clone()
                for value in built.region_examples[region.name]
            ),
            example_path,
        )
        regions.append(
            {
                "name": region.name,
                "export_sha256": digest(path),
                "capture_sha256": digest(evidence),
                "example_args_sha256": digest(example_path),
                "operator_inventory": save_inventory(
                    outcome.exported_program, root, region.name
                ),
            }
        )
        log_stage(
            root,
            "capture_pass",
            region=region.name,
            maximum_absolute_error=outcome.evidence.maximum_absolute_error,
        )
    record = {
        "schema": "vlaforge.smolvla_fresh_capture/1",
        "status": "captured",
        "processing_mode": args.processing_mode,
        "normalization": manifest["normalization"],
        "statistics_recovery": manifest.get("statistics_recovery"),
        "input_pack": str(args.input_pack.resolve()),
        "input_pack_sha256": digest(args.input_pack),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_strict_load": True,
        "policy_path": str(args.policy_path.resolve()),
        "num_steps": args.num_steps,
        "upstream_revision": subprocess.check_output(
            ["git", "-C", str(args.lerobot_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "module_sha256": digest(root / "module.json"),
        "regions": regions,
        "records": records,
        "output_shape": list(loaded[0][2].shape),
        "dtype": "float32",
        "device": args.device,
        "target": "sm_%d%d"
        % torch.cuda.get_device_capability(torch.device(args.device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "camera_keys": list(built.camera_keys),
        "source_adapter_sha256": digest(
            SOURCE / "python/vlaforge/adapters/smolvla/smolvla_fresh.py"
        ),
        "source_tool_sha256": digest(Path(__file__)),
        "full_paper_acceptance": False,
    }
    write_json(root / "capture.json", record)
    log_stage(
        root,
        "capture_completed",
        samples=len(records),
        regions=len(regions),
        processing_mode=args.processing_mode,
    )


def load_capture(root: Path):
    from vlaforge.ir.serializer import module_from_data

    record = read_json(root / "capture.json")
    if (
        record.get("schema") != "vlaforge.smolvla_fresh_capture/1"
        or record.get("status") != "captured"
    ):
        raise ValueError("a completed fresh capture is required")
    if digest(root / "module.json") != record["module_sha256"]:
        raise ValueError("captured semantic IR digest changed")
    if digest(Path(record["input_pack"])) != record["input_pack_sha256"]:
        raise ValueError("captured input processing protocol changed")
    for region in record["regions"]:
        for suffix, key in (
            (".pt2e", "export_sha256"),
            (".capture.json", "capture_sha256"),
        ):
            if digest(root / "exports" / f"{region['name']}{suffix}") != region[key]:
                raise ValueError("captured Region evidence or export changed")
    for sample in record["records"]:
        for name, key in (
            ("inputs.npz", "inputs_sha256"),
            ("reference.npy", "reference_sha256"),
        ):
            if digest(root / sample["path"] / name) != sample[key]:
                raise ValueError("captured input or full reference changed")
    return record, module_from_data(read_json(root / "module.json"))


def compile_artifacts(args):
    record, _ = load_capture(args.output)
    folder = args.output / "artifacts"
    folder.mkdir(exist_ok=True)
    for region in record["regions"]:
        name = region["name"]
        artifact, manifest = folder / f"{name}.pt2", folder / f"{name}.compile.json"
        if artifact.exists() or manifest.exists():
            verify_artifact(
                args.output, region, record["target"], args.inductor_profile
            )
            continue
        command = [
            sys.executable,
            "-m",
            "vlaforge.cli",
            "compile-artifact",
            str(args.output / "exports" / f"{name}.pt2e"),
            "--output",
            str(artifact),
            "--manifest",
            str(manifest),
            "--target",
            record["target"],
            "--inductor-profile",
            args.inductor_profile,
        ]
        log_stage(args.output, "compile_start", region=name, command=command)
        with (folder / f"{name}.compile.log").open("w") as log:
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(SOURCE / "python"),
                    "TORCHINDUCTOR_COMPILE_THREADS": "1",
                },
            )
        verify_artifact(args.output, region, record["target"], args.inductor_profile)
        log_stage(
            args.output, "compile_pass", region=name, profile=args.inductor_profile
        )


def verify_artifact(root: Path, region, target: str, profile: str):
    from vlaforge.deployment.aoti_export import (
        backend_pass_records,
        backend_program_pass_records,
    )
    from vlaforge.deployment.aoti_package import package_pass_records, verify_package_audit
    from vlaforge.deployment.aoti_profile import aoti_configs

    path = root / "artifacts" / f"{region['name']}.pt2"
    result = read_json(root / "artifacts" / f"{region['name']}.compile.json")
    configs = aoti_configs(profile)
    program_audit = result.get("backend_program_audit", {})
    expected_program_passes = backend_program_pass_records(configs)
    artifact_sha = digest(path)
    if (
        result.get("status") != "passed"
        or result.get("target") != target
        or result.get("inductor_profile") != profile
        or result.get("inductor_configs") != configs
        or result.get("backend_graph_passes", []) != backend_pass_records(configs)
        or program_audit.get("passes", []) != expected_program_passes
        or (expected_program_passes and not isinstance(program_audit.get("rewrites"), list))
        or result.get("backend_package_audit", {}).get("passes", [])
        != package_pass_records(configs)
        or result.get("exported_program", {}).get("sha256") != region["export_sha256"]
        or result.get("artifact", {}).get("sha256") != artifact_sha
        or result["artifact"].get("size_bytes") != path.stat().st_size
    ):
        raise ValueError(
            "AOTI package is incomplete or differs from its captured source/profile"
        )
    verify_package_audit(path, configs, result.get("backend_package_audit", {}),
                         artifact_sha256=artifact_sha)
    return result


def candidate_scale(args, captured, reference, actual, folder):
    if args.statistics_namespace is None:
        return {"status": "not_requested", "physical_action_units_verified": False}
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.smolvla import processor_smolvla  # noqa: F401
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from vlaforge.adapters.smolvla.smolvla_processing import resolve_smolvla_statistics

    if args.dataset_info is None:
        raise ValueError(
            "statistics selection requires the verified dataset meta/info.json"
        )
    source_pack = read_json(Path(captured["input_pack"]))
    if digest(args.dataset_info) != source_pack["dataset_files"]["meta/info.json"]:
        raise ValueError(
            "dataset robot metadata differs from the captured input protocol"
        )
    robot_type = read_json(args.dataset_info)["robot_type"]
    policy = Path(captured["policy_path"])
    verify_policy_provenance(policy, source_pack)
    processor_config = read_json(policy / "policy_postprocessor.json")
    steps = [
        step
        for step in processor_config["steps"]
        if step["registry_name"] == "unnormalizer_processor"
    ]
    if len(steps) != 1:
        raise ValueError("one declared action unnormalizer is required")
    state_path = (policy / steps[0]["state_file"]).resolve(strict=True)
    if not state_path.is_relative_to(policy.resolve()):
        raise ValueError("processor statistics must remain inside the policy profile")
    expected_stats = source_pack["processor_files"].get(state_path.name)
    if expected_stats is None:
        expected_stats = (
            (source_pack.get("statistics_recovery") or {})
            .get("output_files", {})
            .get(state_path.name)
        )
    if digest(state_path) != expected_stats:
        raise ValueError("postprocessor source changed")
    profile = resolve_smolvla_statistics(
        load_file(str(state_path)),
        namespace=None
        if args.statistics_namespace == "unscoped"
        else args.statistics_namespace,
        robot_type=robot_type,
        feature_shapes={"action": (reference.shape[-1],)},
        source_sha256=digest(state_path),
    )
    post = PolicyProcessorPipeline.from_pretrained(
        str(policy),
        config_filename="policy_postprocessor.json",
        overrides={"unnormalizer_processor": {"stats": profile.stats}},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    profile.require_consumed(post.steps[0])
    before, candidate = (
        torch.from_numpy(reference.copy()),
        torch.from_numpy(actual.copy()),
    )
    native, native_candidate = post(before.clone()), post(candidate.clone())
    verified = profile.verify_transform("action", before, native, inverse=True)
    verified_candidate = profile.verify_transform(
        "action", candidate, native_candidate, inverse=True
    )
    np.savez(
        folder / "action_scale_candidate.npz",
        reference=native.numpy(),
        candidate=native_candidate.numpy(),
    )
    write_json(
        folder / "action_scale_fidelity.json",
        fidelity(
            native.numpy(),
            native_candidate.numpy(),
            sample_id=folder.name,
            space=f"explicit_{args.statistics_namespace}_action_scale_candidate_not_calibrated",
            max_abs=args.max_abs,
            mean_abs=args.mean_abs,
            tensor_tolerances=False,
        ),
    )
    return {
        "status": "explicit_scale_candidate_verified",
        "selection": profile.to_dict(),
        "transform": verified,
        "candidate_transform": verified_candidate,
        "physical_action_units_verified": False,
    }


def verify_action_scale(args):
    import numpy as np

    if args.statistics_namespace is None:
        raise ValueError(
            "action scale verification requires an explicit statistics selection"
        )
    record, _ = load_capture(args.output)
    root = args.output / "action-scale-audit"
    root.mkdir(exist_ok=False)
    direct_samples = read_json(args.output / "direct/report.json")["samples"]
    results = []
    for sample in record["records"]:
        index = sample["index"]
        candidate_path = args.output / "direct" / f"sample-{index:06d}/action_chunk.npy"
        if digest(candidate_path) != direct_samples[index]["output_sha256"]:
            raise ValueError("direct output differs from its original execution report")
        folder = root / f"sample-{index:06d}"
        folder.mkdir()
        reference = np.load(
            args.output / sample["path"] / "reference.npy", allow_pickle=False
        )
        candidate = np.load(candidate_path, allow_pickle=False)
        results.append(
            {
                "index": index,
                **candidate_scale(args, record, reference, candidate, folder),
            }
        )
    write_json(
        root / "report.json",
        {
            "status": "executed",
            "samples": results,
            "physical_action_units_verified": False,
            "full_paper_acceptance": False,
        },
    )
    log_stage(
        args.output,
        "action_scale_verified",
        count=len(results),
        report=str(root / "report.json"),
    )


def verify_export(args):
    import numpy as np
    import torch
    from vlaforge.frontend import InvocationProgram
    from vlaforge.interpreter import InputBinding, InputStamp, Interpreter, TensorView

    record, module = load_capture(args.output)
    implementations, inventories, examples = {}, {}, {}

    def with_examples(name, implementation):
        def run(*arguments):
            if name not in examples:
                path = args.output / "exports" / f"{name}.runtime-inputs.pt"
                torch.save(
                    tuple(value.detach().cpu().clone() for value in arguments), path
                )
                examples[name] = {
                    "path": str(path),
                    "sha256": digest(path),
                    "sample_index": 0,
                }
            return implementation(*arguments)

        return run

    for region in record["regions"]:
        name = region["name"]
        program = torch.export.load(args.output / "exports" / f"{name}.pt2e")
        inventories[name] = save_inventory(program, args.output, name)
        implementations[name] = with_examples(name, program.module())
    executor = Interpreter(
        module,
        regions=implementations,
        validators=InvocationProgram(module, {}).validators,
    )
    results = []
    for sample in record["records"]:
        input_folder = args.output / sample["path"]
        folder = args.output / "export-verification" / f"sample-{sample['index']:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        with np.load(input_folder / "inputs.npz", allow_pickle=False) as pack:
            values = {
                name: torch.from_numpy(pack[name].copy()).to(record["device"])
                for name in pack.files
            }
        bindings = {
            port.name: InputBinding(
                TensorView(
                    values[port.name],
                    port.payload.shape,
                    port.payload.dtype,
                    device=port.device,
                ),
                InputStamp(revision=sample["index"] + 1),
            )
            for port in module.inputs
        }
        with torch.inference_mode():
            actual = executor.run(inputs=bindings).committed_outputs.output(
                "action_chunk"
            )
        actual = actual.detach().cpu().contiguous().numpy()
        reference = np.load(input_folder / "reference.npy", allow_pickle=False)
        np.save(folder / "action_chunk.npy", actual, allow_pickle=False)
        report = fidelity(
            reference,
            actual,
            sample_id=folder.name,
            space="exported_vs_official_model_tensor_output",
            max_abs=0,
            mean_abs=0,
        )
        write_json(folder / "fidelity.json", report)
        results.append(
            {
                "index": sample["index"],
                "gates": report["gates"],
                "metrics": report["metrics"],
            }
        )
    write_json(
        args.output / "export-verification/report.json",
        {
            "status": "executed",
            "samples": results,
            "operator_inventory": inventories,
            "runtime_examples": examples,
        },
    )
    log_stage(
        args.output,
        "complete_export_reference_verified",
        all_bitwise_equal=all(item["gates"]["bitwise_equal"] for item in results),
    )


def direct(args):
    import numpy as np
    import torch
    import torch._inductor
    import torch._inductor.codecache  # noqa: F401
    from vlaforge.frontend import InvocationProgram
    from vlaforge.interpreter import InputBinding, InputStamp, Interpreter, TensorView

    record, module = load_capture(args.output)
    implementations = {}
    for region in record["regions"]:
        verify_artifact(args.output, region, record["target"], args.inductor_profile)
        implementations[region["name"]] = torch._inductor.aoti_load_package(
            str(args.output / "artifacts" / f"{region['name']}.pt2")
        )
    executor = Interpreter(
        module,
        regions=implementations,
        validators=InvocationProgram(module, {}).validators,
    )
    results = []
    for sample in record["records"]:
        folder = args.output / "direct" / f"sample-{sample['index']:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        input_folder = args.output / sample["path"]
        with np.load(input_folder / "inputs.npz", allow_pickle=False) as pack:
            values = {
                name: torch.from_numpy(pack[name].copy()).to(record["device"])
                for name in pack.files
            }
        bindings = {
            port.name: InputBinding(
                TensorView(
                    values[port.name],
                    port.payload.shape,
                    port.payload.dtype,
                    device=port.device,
                ),
                InputStamp(revision=sample["index"] + 1),
            )
            for port in module.inputs
        }
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        with torch.inference_mode():
            actual = executor.run(inputs=bindings).committed_outputs.output(
                "action_chunk"
            )
        torch.cuda.synchronize()
        duration = time.perf_counter_ns() - start
        actual = actual.detach().cpu().contiguous().numpy()
        reference = np.load(input_folder / "reference.npy", allow_pickle=False)
        np.save(folder / "action_chunk.npy", actual, allow_pickle=False)
        comparison = fidelity(
            reference,
            actual,
            sample_id=folder.name,
            space="model_tensor_output_with_explicit_" + record["processing_mode"],
            max_abs=args.max_abs,
            mean_abs=args.mean_abs,
        )
        write_json(folder / "fidelity.json", comparison)
        scaling = candidate_scale(args, record, reference, actual, folder)
        item = {
            "index": sample["index"],
            "latency_ns": duration,
            "gates": comparison["gates"],
            "metrics": comparison["metrics"],
            "action_scale_candidate": scaling,
            "output_sha256": digest(folder / "action_chunk.npy"),
        }
        results.append(item)
        log_stage(args.output, "direct_sample_completed", **item)
    write_json(
        args.output / "direct/report.json",
        {
            "status": "executed",
            "samples": results,
            "measurement_boundary": "single cold fresh Interpreter invocation with resident CUDA tensors; not a benchmark distribution",
            "full_paper_acceptance": False,
        },
    )


def session_loop_rows(module, device: str, policy: str):
    from vlaforge.compiler import compile_module

    if not getattr(module, "invocations", None) and policy == "source":
        return []
    selected = compile_module(
        module, loop_execution=policy, default_device=device, state_device=device
    )
    return [
        {
            "task_id": task.id,
            "policy": task.attributes["replay"],
            "steps": len(
                range(
                    task.attributes["lower"],
                    task.attributes["upper"],
                    task.attributes["step"],
                )
            ),
        }
        for task in selected.plan.tasks
        if task.opcode == "vla.for" and task.attributes.get("replay", "off") != "off"
    ]


def replay_runner_fragments(rows):
    checks, failures = [], []
    for row in rows:
        task, policy, steps = row["task_id"], row["policy"], row["steps"]
        condition = "true"
        if policy == "required":
            condition = f"info.state == VLAFORGE_REPLAY_READY && info.captured_steps == {steps}u && info.replay_count == run + 1u && info.ordinary_count == 0u"
        elif policy == "batch-only":
            condition = "info.state == VLAFORGE_REPLAY_UNPREPARED && info.captured_steps == 0u && info.replay_count == 0u && info.ordinary_count == run + 1u"
        checks.append(f"""    {{
      VLAForgeBoundedReplayInfo info{{}}; info.struct_size = sizeof(info);
      if (vlaforge_model_session_get_replay_info(session, {task}u, &info).code != VLAFORGE_STATUS_OK || !({condition})) {{
        std::fprintf(stderr, "loop {task} execution contract failed\\n"); result = 16; break;
      }}
      std::printf("REPLAY,%zu,{task},{policy},%u,%u,%llu,%llu\\n", run,
          static_cast<unsigned>(info.state), info.captured_steps,
          static_cast<unsigned long long>(info.replay_count),
          static_cast<unsigned long long>(info.ordinary_count));
    }}
""")
        failures.append(f"""      {{
        VLAForgeBoundedReplayInfo info{{}}; info.struct_size = sizeof(info);
        if (vlaforge_model_session_get_replay_info(session, {task}u, &info).code == VLAFORGE_STATUS_OK && info.state == VLAFORGE_REPLAY_POISONED) {{
          std::fprintf(stderr, "fatal capture/context; worker exit: %s\\n", info.reason);
          api->destroy(session); std::fflush(stderr); std::_Exit(10);
        }}
      }}
""")
    return "".join(checks), "".join(failures)


def session_selection(root: Path, record, policy: str):
    return {
        "schema": "vlaforge.fresh_session_selection/1",
        "loop_execution": policy,
        "capture_sha256": digest(root / "capture.json"),
        "input_module_sha256": record["module_sha256"],
        "input_pack_sha256": record["input_pack_sha256"],
        "regions": [
            {
                "name": region["name"],
                "export_sha256": region["export_sha256"],
                "capture_sha256": region["capture_sha256"],
                "compile_manifest_sha256": digest(
                    root / "artifacts" / f"{region['name']}.compile.json"
                ),
                "artifact_sha256": digest(root / "artifacts" / f"{region['name']}.pt2"),
            }
            for region in record["regions"]
        ],
    }


def verify_session_selection(bundle: Path, expected):
    path = bundle / "evidence/session-selection.json"
    if path.exists():
        if read_json(path) != expected:
            raise ValueError(
                "existing Session execution policy or captured artifacts changed"
            )
    elif expected["loop_execution"] != "source":
        raise ValueError(
            "existing Session has no compiler policy evidence; choose a new label"
        )


def runner_source(
    module, sample_count: int, device: str, *, loop_execution="source"
) -> str:
    dtype_bytes = {"f32": 4, "f16": 2, "bf16": 2, "i64": 8, "i32": 4, "bool": 1}
    ordinal = int(device.removeprefix("cuda:"))
    declarations = []
    for port in module.inputs:
        if port.device != device or port.payload.dtype not in dtype_bytes:
            raise ValueError("fresh harness requires fixed CUDA tensor input profiles")
        shape = tuple(port.payload.shape)
        size = math.prod(shape) * dtype_bytes[port.payload.dtype]
        declarations.append(
            '{"%s", VLAFORGE_DTYPE_%s, {%s}, %du}'
            % (port.name, port.payload.dtype.upper(), ",".join(map(str, shape)), size)
        )
    output = module.outputs[0]
    if (
        len(module.outputs) != 1
        or output.payload.dtype != "f32"
        or output.device != device
    ):
        raise ValueError("fresh harness expects one complete FP32 CUDA output")
    source = Path(__file__).with_name("smolvla_fresh_runner.cpp.in").read_text()
    audit, failure = replay_runner_fragments(
        session_loop_rows(module, device, loop_execution)
    )
    return (
        source.replace("@INPUTS@", ",\n".join(declarations))
        .replace("@ORDINAL@", str(ordinal))
        .replace("@SAMPLES@", str(sample_count))
        .replace("@OUTPUT_COUNT@", str(math.prod(output.payload.shape)))
        .replace("@REPLAY_AUDIT@", audit)
        .replace("@REPLAY_FAILURE@", failure)
    )


def session_paths(root: Path, label: str | None) -> tuple[Path, Path]:
    if label is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", label):
            raise ValueError("Session label must be a simple 1-80 character identifier")
        root = root / "session-attempts" / label
    return root / "bundle", root / "session"


def verify_cpp_input_bytes(root: Path, records) -> None:
    import numpy as np

    for sample in records:
        folder = root / sample["path"]
        with np.load(folder / "inputs.npz", allow_pickle=False) as pack:
            for name in pack.files:
                if (folder / f"{name}.bin").read_bytes() != pack[name].tobytes():
                    raise ValueError(
                        "C++ input bytes differ from the verified capture inputs"
                    )


def session(args):
    import numpy as np
    import torch
    from vlaforge.deployment import (
        ArtifactIdentity,
        ArtifactKind,
        EffectAudit,
        RegionArtifactContract,
        ValueContract,
        WorkspaceContract,
        build_artifact_compile_bundle,
        load_bundle_manifest,
    )
    from vlaforge.deployment.capabilities import aoti_backend_capability
    from vlaforge.frontend import InvocationProgram
    from vlaforge.ir.serializer import io_schema_digest
    from vlaforge.validation.deployment_metrics import latency_report

    bundle, folder = session_paths(args.output, args.session_label)
    if (folder / "report.json").exists() or (
        folder.exists() and any(folder.glob("run-*.bin"))
    ):
        raise ValueError("Session outputs already exist; choose a new --session-label")
    record, module = load_capture(args.output)
    policy = getattr(args, "loop_execution", "source")
    contracts, sources = {}, {}
    for index, region in enumerate(record["regions"]):
        verify_artifact(args.output, region, record["target"], args.inductor_profile)
        name = region["name"]
        path = args.output / "artifacts" / f"{name}.pt2"
        evidence = read_json(args.output / "exports" / f"{name}.capture.json")
        contracts[name] = RegionArtifactContract(
            region_id=index,
            region_name=name,
            inputs=tuple(
                ValueContract.from_dict(value) for value in evidence["inputs"]
            ),
            outputs=tuple(
                ValueContract.from_dict(value) for value in evidence["outputs"]
            ),
            io_schema_digest=io_schema_digest(module),
            identity=ArtifactIdentity(
                model_name="SmolVLA-Base-fresh",
                upstream_revision=record["upstream_revision"],
                checkpoint_identity="sha256:" + record["checkpoint_sha256"],
                graph_sha256=evidence["graph_digest"],
            ),
            artifact_kind=ArtifactKind.AOTI_PACKAGE,
            artifact_path=f"artifacts/{name}.pt2",
            artifact_sha256=digest(path),
            artifact_size_bytes=path.stat().st_size,
            workspace=WorkspaceContract(device=record["device"]),
            capability=aoti_backend_capability(
                record["target"], ("bf16", "bool", "f32", "i32", "i64")
            ),
            effect_audit=EffectAudit.from_dict(evidence["effect_audit"]),
            backend_variant=f"torch-{torch.__version__}",
        )
        sources[name] = path
    verify_cpp_input_bytes(args.output, record["records"])
    selection = session_selection(args.output, record, policy)
    source = runner_source(
        module, len(record["records"]), record["device"], loop_execution=policy
    )
    if (bundle / "bundle.json").is_file():
        manifest = load_bundle_manifest(bundle / "bundle.json")
        manifest.verify_files(bundle)
        verify_session_selection(bundle, selection)
        if (
            bundle / "generated/runner.cpp"
        ).read_text() != source or manifest.io_schema_digest != io_schema_digest(
            module
        ):
            raise ValueError(
                "existing Session bundle differs from the captured fresh program"
            )
    else:
        log_stage(args.output, "session_build_start")
        selection_source = bundle.parent / "session-selection.json"
        write_json(selection_source, selection)
        version = subprocess.check_output(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(REPOSITORY), "status", "--porcelain"], text=True
            )
        )
        major, minor = torch.cuda.get_device_capability(torch.device(record["device"]))
        try:
            build_artifact_compile_bundle(
                module,
                bundle,
                region_artifacts=contracts,
                artifact_sources=sources,
                validators=InvocationProgram(module, {}).cpp_validators(),
                runner_source=source,
                runtime_root=SOURCE,
                cmake_prefix_path=torch.utils.cmake_prefix_path,
                backend_versions={
                    "aoti": torch.__version__,
                    "cuda": str(torch.version.cuda),
                },
                profile="verified",
                loop_execution=policy,
                auxiliary_files={
                    "evidence/session-selection.json": selection_source,
                    "evidence/capture.json": args.output / "capture.json",
                },
                source_revision=version,
                source_dirty=dirty,
                environment={
                    "TORCH_CUDA_ARCH_LIST": f"{major}.{minor}",
                    "CMAKE_BUILD_PARALLEL_LEVEL": "2",
                },
                default_device=record["device"],
                state_device=record["device"],
            )
        except subprocess.CalledProcessError as error:
            (bundle.parent / "session-build-failure.log").write_text(
                str(error.stdout or "") + str(error.stderr or "")
            )
            raise
        log_stage(args.output, "session_build_pass")
    folder.mkdir(parents=True, exist_ok=True)
    runner = bundle / "bin/vlaforge_generated_runner"
    dependencies = subprocess.check_output(["ldd", str(runner)], text=True)
    (folder / "runner.ldd.txt").write_text(dependencies)
    if "libpython" in dependencies.lower():
        raise ValueError("C++ runner unexpectedly links Python")
    command = [
        str(runner),
        str(bundle),
        str(args.output / "inputs"),
        str(folder),
        str(args.session_repetitions),
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONHOME": "/no/python/home",
            "PYTHONPATH": "/no/python/path",
        },
    )
    (folder / "stdout.log").write_text(completed.stdout)
    (folder / "stderr.log").write_text(completed.stderr)
    write_json(
        folder / "execution.json",
        {
            "command": command,
            "exit_code": completed.returncode,
            "loop_execution": policy,
        },
    )
    completed.check_returncode()
    timings, comparisons, replay_records = [], [], []
    for line in completed.stdout.splitlines():
        if line.startswith("REPLAY,"):
            _, run, task, mode, state, captured, replays, ordinary = line.split(",")
            replay_records.append(
                {
                    "run": int(run),
                    "task_id": int(task),
                    "policy": mode,
                    "state": int(state),
                    "captured_steps": int(captured),
                    "replay_count": int(replays),
                    "ordinary_count": int(ordinary),
                }
            )
            continue
        if not line.startswith("FRESH,"):
            continue
        _, run_id, sample_id, duration = line.split(",")
        run_id, sample_id = int(run_id), int(sample_id)
        actual = np.fromfile(folder / f"run-{run_id}.bin", dtype=np.float32).reshape(
            record["output_shape"]
        )
        sample = record["records"][sample_id]
        reference = np.load(
            args.output / sample["path"] / "reference.npy", allow_pickle=False
        )
        direct_output = np.load(
            args.output / "direct" / f"sample-{sample_id:06d}/action_chunk.npy",
            allow_pickle=False,
        )
        direct_comparison = fidelity(
            direct_output,
            actual,
            sample_id=f"run-{run_id}",
            space="cpp_vs_same_aoti_direct",
            max_abs=0.0,
            mean_abs=0.0,
        )
        reference_comparison = fidelity(
            reference,
            actual,
            sample_id=f"run-{run_id}",
            space="cpp_vs_official_model_tensor_output",
            max_abs=args.max_abs,
            mean_abs=args.mean_abs,
        )
        write_json(folder / f"run-{run_id}.direct-fidelity.json", direct_comparison)
        write_json(
            folder / f"run-{run_id}.reference-fidelity.json", reference_comparison
        )
        comparisons.append(
            {
                "index": run_id,
                "sample": sample_id,
                "direct_gates": direct_comparison["gates"],
                "reference_gates": reference_comparison["gates"],
            }
        )
        timings.append(
            {
                "index": run_id,
                "latency_ns": int(duration),
                "repeat_id": run_id // len(record["records"]),
            }
        )
    expected = args.session_repetitions * len(record["records"])
    if len(timings) != expected or [item["index"] for item in timings] != list(
        range(expected)
    ):
        raise ValueError("C++ runner omitted or duplicated a complete fresh output")
    result = {
        "status": "executed",
        "session_label": args.session_label,
        "loop_execution": policy,
        "execution_selection": selection,
        "loop_runtime_records": replay_records,
        "first_call_includes_loop_warmup_and_capture": policy == "required",
        "bundle_manifest_sha256": digest(bundle / "bundle.json"),
        "command": command,
        "samples": comparisons,
        "latency": latency_report(timings),
        "warmup_count": 0,
        "measurement_boundary": "resident CUDA tensors; fresh Session run plus synchronization; H2D/D2H excluded",
        "every_observation_revision_changed": True,
        "python_linked": False,
        "python_environment_disabled": True,
        "full_paper_acceptance": False,
        "direct_cpp_all_bitwise_equal": all(
            item["direct_gates"]["bitwise_equal"] for item in comparisons
        ),
    }
    write_json(folder / "report.json", result)
    log_stage(
        args.output,
        "session_completed",
        session_label=args.session_label,
        report=str(folder / "report.json"),
        count=len(timings),
        direct_cpp_all_bitwise_equal=result["direct_cpp_all_bitwise_equal"],
    )
    if policy != "source" and not result["direct_cpp_all_bitwise_equal"]:
        raise ValueError(
            "generated loop execution differs from the saved same-artifact direct output"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "capture",
            "compile",
            "verify-export",
            "verify-action-scale",
            "direct",
            "session",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-pack", type=Path)
    parser.add_argument(
        "--processing-mode",
        choices=("published-compatibility", "strict-statistics"),
        required=True,
    )
    parser.add_argument(
        "--policy-path", type=Path, default=REPOSITORY / "examples/smolvla/SmolVLA-Base"
    )
    parser.add_argument(
        "--vlm-path",
        type=Path,
        default=REPOSITORY / "examples/smolvla/SmolVLM2-500M-Video-Instruct",
    )
    parser.add_argument(
        "--lerobot-source", type=Path, default=REPOSITORY / "third_party/lerobot"
    )
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--inductor-profile",
        choices=("default", "conservative", "eager-numerics", "aten-preserving"),
        default="conservative",
    )
    parser.add_argument("--max-abs", type=float, default=0.05)
    parser.add_argument("--mean-abs", type=float, default=0.01)
    parser.add_argument("--statistics-namespace")
    parser.add_argument("--dataset-info", type=Path)
    parser.add_argument("--session-repetitions", type=int, default=1)
    parser.add_argument("--session-label")
    parser.add_argument(
        "--loop-execution",
        choices=("source", "off", "batch-only", "prefer", "required"),
        default="source",
    )
    args = parser.parse_args(argv)
    if args.first < 0 or args.count < 1 or args.num_steps < 1:
        parser.error("observation interval and iteration count must be positive")
    if not 1 <= args.session_repetitions <= 100000:
        parser.error("session repetitions must be in [1,100000]")
    if any(
        not math.isfinite(value) or value < 0 for value in (args.max_abs, args.mean_abs)
    ):
        parser.error("numeric tolerances must be finite and nonnegative")
    if args.stage in ("capture", "all") and args.input_pack is None:
        parser.error("capture requires --input-pack")
    args.output = args.output.resolve()
    sys.path.insert(0, str(args.lerobot_source / "src"))
    stages = (
        ("capture", "compile", "verify-export", "direct", "session")
        if args.stage == "all"
        else (args.stage,)
    )
    try:
        for stage in stages:
            if stage != "capture":
                captured, _ = load_capture(args.output)
                if captured["processing_mode"] != args.processing_mode:
                    raise ValueError(
                        "requested mode differs from captured input semantics"
                    )
            {
                "capture": capture,
                "compile": compile_artifacts,
                "verify-export": verify_export,
                "verify-action-scale": verify_action_scale,
                "direct": direct,
                "session": session,
            }[stage](args)
    except Exception as error:
        log_stage(
            args.output, "failure", error_type=type(error).__name__, message=str(error)
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
