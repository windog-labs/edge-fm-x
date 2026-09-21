"""Benchmark actual exported operator examples in an isolated CUDA process."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import statistics
import sys
import time
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def recipe_configs(recipe):
    from vlaforge.deployment.aoti_profile import aoti_configs

    if recipe == "aten":
        return None
    if recipe == "inductor-aten-preserving":
        return aoti_configs("aten-preserving")
    if recipe not in ("inductor-aten", "inductor-autotune"):
        raise ValueError("unknown operator candidate recipe")
    result = aoti_configs("eager-numerics")
    if recipe == "inductor-autotune":
        result.update(
            max_autotune=True,
            max_autotune_gemm=True,
            max_autotune_gemm_backends="ATEN,TRITON",
        )
    return result


def verify_example(manifest_path, node):
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema")
        not in (
            "vlaforge.actual_operator_examples/1",
            "vlaforge.actual_subgraph_examples/1",
        )
        or manifest.get("status") != "extracted_and_eager_verified"
        or not node
        or Path(node).name != node
        or node in (".", "..")
    ):
        raise ValueError("a complete actual operator example manifest is required")
    matching = [item for item in manifest["examples"] if item["node"] == node]
    if len(matching) != 1:
        raise ValueError("operator selection is missing or ambiguous")
    record = matching[0]
    folder = manifest_path.parent / node
    for filename, key in (
        ("inputs.pt", "inputs_sha256"),
        ("operator.pt2", "export_sha256"),
    ):
        path = (folder / filename).resolve(strict=True)
        if (
            not path.is_relative_to(manifest_path.parent.resolve())
            or digest(path) != record[key]
        ):
            raise ValueError("operator example file differs from its capture hash")
    return record


def output_metrics(reference, candidate):
    from torch.utils._pytree import tree_flatten
    from vlaforge.analysis.numerical_probe import tensor_difference

    expected, expected_tree = tree_flatten(reference)
    actual, actual_tree = tree_flatten(candidate)
    if expected_tree != actual_tree:
        raise ValueError("candidate changed output structure")
    return [tensor_difference(a, b) for a, b in zip(expected, actual, strict=True)]


def output_identity(value):
    import torch
    from torch.utils._pytree import tree_flatten
    from vlaforge.analysis.numerical_probe import tensor_value_bytes

    values, structure = tree_flatten(value)
    if not values or not all(isinstance(item, torch.Tensor) for item in values):
        raise ValueError("operator reference requires nonempty tensor outputs")
    return {
        "structure": str(structure),
        "tensors": [
            {
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "sha256": hashlib.sha256(tensor_value_bytes(item)).hexdigest(),
            }
            for item in values
        ],
    }


def validate_reference_platform(manifest, gpu, mode):
    if mode not in ("captured", "revalidate-workload"):
        raise ValueError("unknown operator reference mode")
    if mode == "captured" and manifest.get("gpu") != gpu:
        raise ValueError("benchmark must use the example's actual GPU platform")


def load_example_values(path):
    import torch

    # CPU scalar tensors participate in different opmath from CUDA scalars.
    # An unavailable recorded CUDA ordinal must fail, not silently remap devices.
    return torch.load(path, weights_only=True)


def verified_reuse(previous_path, current):
    from vlaforge.deployment.aoti_export import (
        backend_pass_records,
        backend_program_pass_records,
    )
    from vlaforge.deployment.aoti_package import (
        package_pass_records,
        verify_package_audit,
    )

    previous = json.loads(previous_path.read_text())
    for name in (
        "schema",
        "recipe",
        "inductor_configs",
        "source_manifest_sha256",
        "example",
        "torch",
        "cuda",
        "gpu",
        "compute_capability",
        "matmul_precision",
    ):
        if name not in previous or previous[name] != current[name]:
            raise ValueError(f"reuse provenance differs: {name}")
    if current["schema"] == "vlaforge.operator_microbenchmark/2":
        for name in (
            "reference_mode",
            "device_loading",
            "target_numerical_context",
            "target_reference_identity",
            "effect_audit_source_sha256",
        ):
            if name not in previous or previous[name] != current[name]:
                raise ValueError(f"reuse provenance differs: {name}")
    if previous.get("backend_graph_audit", {}).get(
        "passes", []
    ) != backend_pass_records(current["inductor_configs"]):
        raise ValueError("reuse provenance differs: backend graph passes")
    program_audit = previous.get("backend_program_audit", {})
    expected_program_passes = backend_program_pass_records(current["inductor_configs"])
    if program_audit.get("passes", []) != expected_program_passes or (
        expected_program_passes and not isinstance(program_audit.get("rewrites"), list)
    ):
        raise ValueError("reuse provenance differs: backend program preparation")
    if previous.get("backend_package_audit", {}).get(
        "passes", []
    ) != package_pass_records(current["inductor_configs"]):
        raise ValueError("reuse provenance differs: backend package passes")
    artifact = previous_path.parent / "compiled.pt2"
    artifact_sha = digest(artifact)
    if artifact_sha != previous.get("artifact_sha256"):
        raise ValueError("reuse artifact differs from its recorded hash")
    verify_package_audit(
        artifact,
        current["inductor_configs"],
        previous.get("backend_package_audit", {}),
        artifact_sha256=artifact_sha,
    )
    return artifact


def load_aoti_candidate(artifact):
    import importlib
    import torch

    # Standalone Torch 2.10 package loading needs the device metadata helper.
    importlib.import_module("torch._inductor.codecache")
    # The normal model pool queries completion events, which is capture-unsafe.
    return torch._inductor.aoti_load_package(
        str(artifact), run_single_threaded=True, device_index=0
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument(
        "--recipe",
        choices=(
            "aten",
            "inductor-aten",
            "inductor-autotune",
            "inductor-aten-preserving",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rep-ms", type=int, default=20)
    parser.add_argument("--reuse-compiled-from", type=Path)
    parser.add_argument(
        "--reference-mode",
        choices=("captured", "revalidate-workload"),
        default="captured",
        help="Explicit workload transport remeasures a local eager reference; source parity remains separate",
    )
    args = parser.parse_args()
    if not 1 <= args.rep_ms <= 100:
        parser.error("rep-ms must be in [1,100]")
    if os.environ.get("COGACT_GPU_OWNER_FOLDER") is not None:
        from cogact_gpu_monitor import child_handshake

        child_handshake()
    record = verify_example(args.examples, args.node)
    configs = recipe_configs(args.recipe)
    if args.reuse_compiled_from and configs is None:
        parser.error("ATen baseline does not load a compiled candidate")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "vlaforge.operator_microbenchmark/2",
        "status": "started",
        "recipe": args.recipe,
        "inductor_configs": configs,
        "example": record,
        "source_manifest_sha256": digest(args.examples),
        "tool_sha256": digest(Path(__file__)),
        "command": [sys.executable, *sys.argv],
        "candidate_accepted": False,
        "microbenchmark_numeric_eligible": False,
        "selected_for_deployment": False,
        "end_to_end_integrated": False,
        "reference_mode": args.reference_mode,
        "device_loading": "preserve-recorded-devices/1",
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        import torch
        import torch._inductor.codecache
        import triton
        from triton.testing import do_bench_cudagraph
        from vlaforge.frontend import effect_audit
        from vlaforge.numerical_context import snapshot

        if not torch.cuda.is_available():
            raise ValueError("actual CUDA execution is required")
        report.update(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            triton=triton.__version__,
            gpu=torch.cuda.get_device_name(),
            compute_capability=list(torch.cuda.get_device_capability()),
            matmul_precision=torch.get_float32_matmul_precision(),
        )
        source_manifest = json.loads(args.examples.read_text())
        validate_reference_platform(source_manifest, report["gpu"], args.reference_mode)
        report["source_gpu"] = source_manifest.get("gpu")
        report["source_numerical_context"] = source_manifest.get(
            "source_numerical_context"
        )
        report["historical_context_restored"] = False
        context = snapshot()
        report["target_numerical_context"] = context.to_dict()
        folder = args.examples.parent / args.node
        values = load_example_values(folder / "inputs.pt")
        program = torch.export.load(folder / "operator.pt2")
        report["effect_audit_source_sha256"] = digest(Path(effect_audit.__file__))
        audit = effect_audit.audit_exported_program(program)
        report["effect_audit"] = audit.to_dict()
        if not audit.passed:
            raise ValueError("operator exported program failed recursive effect audit")
        source_reference = values["reference"]
        call_args, call_kwargs = values["args"], values["kwargs"]
        eager = program.module()
        with torch.inference_mode():
            first = eager(*call_args, **call_kwargs)
            report["source_reference_comparison"] = output_metrics(
                source_reference, first
            )
            report["source_reference_bitwise_equal"] = all(
                item["bitwise_equal"] for item in report["source_reference_comparison"]
            )
            second = eager(*call_args, **call_kwargs)
            report["target_reference_repeat"] = output_metrics(first, second)
        context.require_current()
        if not all(item["bitwise_equal"] for item in report["target_reference_repeat"]):
            raise ValueError("local eager reference is not repeatable")
        if (
            args.reference_mode == "captured"
            and not report["source_reference_bitwise_equal"]
        ):
            raise ValueError("local eager differs from captured reference")
        reference = (
            first if args.reference_mode == "revalidate-workload" else source_reference
        )
        report["target_reference_identity"] = output_identity(reference)
        torch.save(
            {"source_reference": source_reference, "target_reference": first},
            args.output / "reference-comparison.pt",
        )
        report["reference_comparison_sha256"] = digest(
            args.output / "reference-comparison.pt"
        )
        save()
        if configs is None:
            implementation = eager
            report["compile_seconds"] = 0.0
        else:
            if args.reuse_compiled_from:
                artifact = verified_reuse(args.reuse_compiled_from, report)
                report["compile_seconds"] = None
                report["backend_graph_audit"] = json.loads(
                    args.reuse_compiled_from.read_text()
                ).get("backend_graph_audit", {"passes": [], "rewrites": []})
                report["backend_program_audit"] = json.loads(
                    args.reuse_compiled_from.read_text()
                ).get("backend_program_audit", {"passes": [], "rewrites": []})
                report["backend_package_audit"] = json.loads(
                    args.reuse_compiled_from.read_text()
                ).get("backend_package_audit", {"passes": [], "translation": None})
                report["reuse"] = {
                    "report_path": str(args.reuse_compiled_from.resolve()),
                    "report_sha256": digest(args.reuse_compiled_from),
                    "artifact_path": str(artifact.resolve()),
                    "numeric_and_timing_validation_skipped": False,
                }
            else:
                report["status"] = "compiling"
                save()
                from vlaforge.deployment.aoti_export import (
                    prepare_backend_options,
                    prepare_backend_program,
                )

                backend_program, program_audit = prepare_backend_program(
                    program, configs
                )
                backend_options, backend_audit = prepare_backend_options(configs)
                start = time.perf_counter()
                artifact = torch._inductor.aoti_compile_and_package(
                    backend_program,
                    package_path=str(args.output / "compiled.pt2"),
                    inductor_configs=backend_options,
                )
                report["compile_seconds"] = time.perf_counter() - start
                report["backend_graph_audit"] = backend_audit
                report["backend_program_audit"] = program_audit
                from vlaforge.deployment.aoti_package import finalize_aoti_package

                report["backend_package_audit"] = finalize_aoti_package(
                    artifact, configs
                )
            report["artifact_sha256"] = digest(Path(artifact))
            report["loader"] = {"run_single_threaded": True, "device_index": 0}
            implementation = load_aoti_candidate(artifact)
        context.require_current()
        with torch.inference_mode():
            actual = implementation(*call_args, **call_kwargs)
            metrics = output_metrics(reference, actual)
            report["metrics"] = metrics
            report["correctness_gate"] = "bitwise-equal-full-output"
            report["correctness_passed"] = all(
                item["bitwise_equal"] for item in metrics
            )
            torch.save(
                {"reference": reference, "candidate": actual},
                args.output / "outputs.pt",
            )
            report["outputs_sha256"] = digest(args.output / "outputs.pt")
            if not report["correctness_passed"]:
                report["status"] = "rejected_numerics"
                save()
                return 3
            torch.cuda.synchronize()
            report["status"] = "benchmarking"
            save()
            benchmark_source = inspect.getsource(do_bench_cudagraph)
            (args.output / "benchmark-library-source.py").write_text(benchmark_source)
            samples = do_bench_cudagraph(
                lambda: implementation(*call_args, **call_kwargs),
                rep=args.rep_ms,
                return_mode="all",
            )
        context.require_current()
        if not samples or any(not (0 < float(item) < float("inf")) for item in samples):
            raise ValueError(
                "benchmark did not return finite positive measured samples"
            )
        report.update(
            status="measured",
            microbenchmark_numeric_eligible=True,
            measurement={
                "method": "triton.testing.do_bench_cudagraph",
                "library_source_sha256": digest(
                    args.output / "benchmark-library-source.py"
                ),
                "rep_ms": args.rep_ms,
                "raw_gpu_batch_means_ms": [float(item) for item in samples],
                "median_ms": statistics.median(samples),
                "mean_ms": statistics.mean(samples),
                "sample_semantics": "each value is an actually timed CUDA graph batch divided by its operation count; not a single-invocation latency CDF",
                "boundary": "resident fixed real input tensors, graph-batched device work, warm cache; excludes compilation, input transfer, Python dispatch and C++ Session scheduling",
            },
        )
        save()
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:  # noqa: BLE001 - terminal process boundary; never fallback.
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        save()
        # A failed graph capture can poison process-global CUDA allocator state.
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    sys.exit(main())
