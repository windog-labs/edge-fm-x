#!/usr/bin/env python3
"""Isolate the Torch 2.10 proxy empty-list bug without changing shared passes."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import traceback


@contextmanager
def native_stderr(path):
    saved = os.dup(2)
    try:
        with Path(path).open("wb") as stream:
            os.dup2(stream.fileno(), 2)
            try:
                yield
            finally:
                os.dup2(saved, 2)
    finally:
        os.close(saved)


def mark_empty_proxy_lists(graph):
    import torch

    changes = []
    for node in graph.nodes:
        if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
            continue
        affected = []
        for index, argument in enumerate(node.target._schema.arguments):
            value = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
            if not isinstance(value, (tuple, list)) or len(value):
                continue
            kind = argument.type
            if kind.kind() == "OptionalType":
                kind = kind.getElementType()
            if kind.kind() != "ListType":
                continue
            element = kind.getElementType()
            dynamic = element.kind() in ("IntType", "SymIntType", "TensorType", "NumberType")
            if element.kind() == "OptionalType":
                dynamic = element.getElementType().kind() == "TensorType"
            if dynamic:
                affected.append(argument.name)
        if affected:
            custom = dict(node.meta.get("custom", {}))
            custom.setdefault("compile_with_inductor", {})
            node.meta["custom"] = custom
            changes.append({"node": node.name, "target": str(node.target), "arguments": affected,
                            "kind": "native_lowering_empty_dynamic_proxy_list"})
    return changes


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--export-sha", required=True)
    parser.add_argument("--package-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.export) != args.export_sha or sha(args.package) != args.package_sha:
        raise ValueError("original export/package digest mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import torch._inductor.codecache  # noqa: F401
    from torch._inductor.custom_graph_pass import CustomGraphPass
    from vlaforge.deployment.aoti_profile import aoti_configs
    from vlaforge.deployment.aoti_export import prepare_backend_options
    from vlaforge.deployment.aoti_package import finalize_aoti_package

    report = {"status": "started", "export_sha256": args.export_sha, "original_package_sha256": args.package_sha,
              "tool_sha256": sha(__file__), "torch": torch.__version__, "cuda": torch.version.cuda,
              "semantic_change": False, "full_model_acceptance": False}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        exported = torch.export.load(args.export)
        positional, keywords = exported.example_inputs
        report["input_shapes"] = [list(value.shape) for value in positional]
        report["input_dtypes"] = [str(value.dtype) for value in positional]
        torch.save(positional, args.output / "actual-example-inputs.pt")
        report["example_inputs_sha256"] = sha(args.output / "actual-example-inputs.pt")
        with torch.inference_mode():
            expected = exported.module()(*positional, **keywords)
        np.save(args.output / "expected.npy", expected.float().cpu().numpy(), allow_pickle=False)
        try:
            original = torch._inductor.aoti_load_package(str(args.package))
            with native_stderr(args.output / "original-native-stderr.log"):
                with torch.inference_mode():
                    result = original(*positional, **keywords)
                torch.cuda.synchronize()
            report["original"] = (
                {"status": "unexpected_success", "exact": torch.equal(expected, result)}
                if isinstance(result, torch.Tensor)
                else {"status": "returned_non_tensor", "output_type": type(result).__name__}
            )
        except Exception as error:
            report["original"] = {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}
        save()
        original_stderr = args.output / "original-native-stderr.log"
        native_error = original_stderr.read_text() if original_stderr.is_file() else ""
        report["original"]["native_stderr_sha256"] = sha(original_stderr) if original_stderr.is_file() else None
        if "Expected SymIntList or IntList but got None" not in report["original"].get("error", "") + native_error:
            raise ValueError("original package did not reproduce the expected empty-list failure")
        configs = aoti_configs("aten-preserving")
        # Factory operators such as ones need selective decomposition before
        # lowering; a post-grad-only marker is too late for those operators.
        report["pre_aot_rewrites"] = mark_empty_proxy_lists(exported.graph)
        save()
        options, audit = prepare_backend_options(configs)
        original_pass = options["post_grad_custom_post_pass"]
        rewrites = []

        class DiagnosticPass(CustomGraphPass):
            def __call__(self, graph):
                original_pass(graph)
                rewrites.extend(mark_empty_proxy_lists(graph))

            def uuid(self):
                return None

        options["post_grad_custom_post_pass"] = DiagnosticPass()
        candidate_path = args.output / "candidate.pt2"
        torch._inductor.aoti_compile_and_package(exported, package_path=str(candidate_path), inductor_configs=options)
        report["shared_pass_audit"], report["diagnostic_rewrites"] = audit, rewrites
        report["package_audit"] = finalize_aoti_package(candidate_path, configs)
        report["candidate_sha256"] = sha(candidate_path)
        candidate = torch._inductor.aoti_load_package(str(candidate_path))
        with torch.inference_mode():
            actual = candidate(*positional, **keywords)
        torch.cuda.synchronize()
        np.save(args.output / "actual.npy", actual.float().cpu().numpy(), allow_pickle=False)
        report["candidate"] = {"shape": list(actual.shape), "dtype": str(actual.dtype), "device": str(actual.device),
                               "exact": actual.shape == expected.shape and actual.dtype == expected.dtype and torch.equal(expected, actual)}
        report["status"] = "isolated_region_exact" if report["candidate"]["exact"] else "isolated_region_mismatch"
        save()
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
