"""Independent same-artifact runtime context/layout diagnostics, never a benchmark."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path

from vlaforge.adapters.openpi.openpi_aoti import _capture_source, _region_result, _tuple
from vlaforge.adapters.openpi.openpi_capture import _run_invocation, _saved_device
from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.adapters.openpi.openpi_operator import _fingerprint
from vlaforge.numerical_context import offline_restore, snapshot


@contextmanager
def record_matmul_setters(torch, events):
    """Observe one public setter in this isolated worker; not thread safe."""
    original = torch.set_float32_matmul_precision

    def recorded(value):
        event = {"setter": "torch.set_float32_matmul_precision", "value": value}
        events.append(event)
        original(value)
        event["returned"] = True

    torch.set_float32_matmul_precision = recorded
    try:
        yield
    finally:
        torch.set_float32_matmul_precision = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--numeric-policy", choices=("captured", "default"), required=True
    )
    parser.add_argument(
        "--loader", choices=("single-threaded", "pooled"), required=True
    )
    parser.add_argument("--stream", choices=("default", "dedicated"), required=True)
    parser.add_argument(
        "--outputs", choices=("direct", "contiguous-copy"), required=True
    )
    parser.add_argument(
        "--inference-mode", choices=("captured", "disabled"), required=True
    )
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 3:
        parser.error("diagnostic repetitions must be in [1,3]")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "vlaforge.openpi_same_artifact_runtime_probe/1",
        "status": "started",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "tool": file_digest(Path(__file__)),
        "profile": {
            key: getattr(args, key)
            for key in (
                "numeric_policy",
                "loader",
                "stream",
                "outputs",
                "inference_mode",
                "repetitions",
            )
        },
        "performance_measured": False,
        "discarded_warmup": 0,
        "native_policy_enforcement": False,
        "calls": [],
        "runs": [],
        "matmul_setter_calls": [],
        "setter_trace_scope": "Only public set_float32_matmul_precision calls are intercepted; other domains use complete observed snapshots",
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        import numpy as np
        import torch
        import torch._inductor.codecache  # noqa: F401
        import vlaforge.numerical_context
        from vlaforge.deployment import load_bundle_manifest
        from vlaforge.frontend import InvocationProgram
        from vlaforge.validation.contracts import NumericContract
        from vlaforge.validation.deployment_metrics import compare_action_chunk

        path, source, context, module, regions = _capture_source(
            args.capture_report, args.capture_sha256
        )
        if file_digest(args.native_report)["sha256"] != args.native_sha256:
            raise ValueError("native report SHA mismatch")
        native = json.loads(args.native_report.read_text())
        if (
            native.get("native_exit_code") != 0
            or native.get("python_linked") is not False
            or native.get("python_environment_disabled") is not True
            or native.get("selection", {}).get("capture", {}).get("sha256")
            != args.capture_sha256
        ):
            raise ValueError("requires complete real same-capture native execution")
        bundle = args.native_report.parent / "bundle"
        if file_digest(bundle / "bundle.json") != native["bundle"]:
            raise ValueError("native bundle manifest changed")
        manifest = load_bundle_manifest(bundle / "bundle.json")
        manifest.verify_files(bundle)
        artifacts = {
            item.region_name: bundle / item.artifact_path
            for item in manifest.region_artifacts
        }
        if set(artifacts) != set(regions):
            raise ValueError("native artifact set differs from captured Regions")
        native_actions = args.native_report.parent / "runs/run-0.bin"
        if file_digest(native_actions) != native["runs"][0]["output"]:
            raise ValueError("original native action bytes changed")
        with np.load(path.parent / "actions.npz", allow_pickle=False) as pack:
            official = pack["normalized_reference"]
        native_reference = np.fromfile(native_actions, dtype=np.float32).reshape(
            official.shape
        )
        device = _saved_device(source)
        report.update(
            capture={"path": str(path), **file_digest(path)},
            native={"path": str(args.native_report), **file_digest(args.native_report)},
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(device),
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            context_before=snapshot().to_dict(),
            numerical_context_implementation=file_digest(
                Path(vlaforge.numerical_context.__file__)
            ),
        )
        guard = (
            offline_restore(context, acknowledge_process_global=True)
            if args.numeric_policy == "captured"
            else nullcontext()
        )
        with record_matmul_setters(torch, report["matmul_setter_calls"]), guard:
            report["execution_context"] = snapshot().to_dict()
            candidates = {
                name: torch._inductor.aoti_load_package(
                    str(artifact),
                    run_single_threaded=args.loader == "single-threaded",
                    device_index=device.index,
                )
                for name, artifact in artifacts.items()
            }
            with np.load(
                path.parent / "prepared_inputs.npz", allow_pickle=False
            ) as pack:
                tensors = {
                    name: torch.from_numpy(pack[name]).to(device) for name in pack.files
                }
            stream = (
                torch.cuda.Stream(device=device)
                if args.stream == "dedicated"
                else torch.cuda.default_stream(device)
            )
            trace = output / "region-trace"
            trace.mkdir()

            def dump(call, direction, values):
                records = []
                for index, value in enumerate(values):
                    metadata = _fingerprint(value)
                    file = trace / f"call-{call}-{direction}-{index}.bin"
                    payload = (
                        value.detach()
                        .contiguous()
                        .reshape(-1)
                        .view(torch.uint8)
                        .cpu()
                        .numpy()
                        .tobytes()
                    )
                    file.write_bytes(payload)
                    records.append(
                        {
                            "index": index,
                            "file": file.name,
                            **metadata,
                            "address_mod_256": value.data_ptr() % 256,
                            "size_bytes": len(payload),
                        }
                    )
                return records

            def wrap(name):
                def run(*values):
                    call = len(report["calls"])
                    inputs = dump(call, "input", values)
                    with torch.inference_mode(args.inference_mode == "captured"):
                        returned = _tuple(candidates[name](*values))
                        before = [_fingerprint(value) for value in returned]
                        actual = (
                            tuple(value.contiguous().clone() for value in returned)
                            if args.outputs == "contiguous-copy"
                            else returned
                        )
                    report["calls"].append(
                        {
                            "call": call,
                            "region": name,
                            "inputs": inputs,
                            "raw_returned_metadata": before,
                            "outputs": dump(call, "output", actual),
                        }
                    )
                    save()
                    return _region_result(actual)

                return run

            implementations = {name: wrap(name) for name in regions}
            validators = InvocationProgram(module, {}).validators
            for index in range(args.repetitions):
                with torch.cuda.stream(stream):
                    actual, _ = _run_invocation(
                        module, implementations, validators, tensors
                    )
                    actual = actual.detach().cpu().numpy()
                np.save(output / f"actions-{index}.npy", actual, allow_pickle=False)
                compare = lambda reference, label: compare_action_chunk(
                    reference,
                    actual,
                    sample_id=f"run-{index}",
                    space=label,
                    contract=NumericContract(
                        absolute_tolerance=0, relative_tolerance=0
                    ),
                )
                report["runs"].append(
                    {
                        "index": index,
                        "vs_native": compare(
                            native_reference, "normalized_probe_vs_native"
                        ),
                        "vs_official": compare(
                            official, "normalized_probe_vs_official"
                        ),
                        "actions": file_digest(output / f"actions-{index}.npy"),
                    }
                )
                save()
        report.update(status="diagnostic-complete", context_after=snapshot().to_dict())
        save()
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
