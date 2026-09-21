"""Extract a declared, read-only higher-order region from real OpenPI capture.

This offline diagnostic preserves the original device and layout. It is not a
latency measurement, and a cold/warm disagreement is retained, not discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from vlaforge.adapters.openpi.openpi_aoti import _capture_source, _tensor_metrics
from vlaforge.adapters.openpi.openpi_capture import _saved_device
from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.analysis.operator_capture import _snapshot_arguments
from vlaforge.analysis.operator_inventory import _metadata
from vlaforge.numerical_context import offline_restore, snapshot


def _fingerprint(value):
    import torch
    from torch.utils._pytree import tree_map

    def tensor(item):
        if not isinstance(item, torch.Tensor):
            return item
        raw = item.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        return {
            **_metadata(item),
            "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        }

    return tree_map(tensor, value)


def _read_only_subgraph(module):
    import torch

    records = []
    for path, child in module.named_modules():
        if not isinstance(child, torch.fx.GraphModule):
            continue
        for node in child.graph.nodes:
            if node.op in {"placeholder", "get_attr", "output"}:
                continue
            if node.op != "call_function":
                raise ValueError("subgraph must contain explicit function calls")
            target = node.target
            if isinstance(target, torch._ops.OpOverload):
                if target._schema.is_mutable:
                    raise ValueError("mutable operator in extracted subgraph")
                if torch.Tag.nondeterministic_seeded in target.tags:
                    raise ValueError("implicit RNG in extracted subgraph")
            elif isinstance(target, torch._ops.HigherOrderOperator):
                if str(target) not in {
                    "wrap_with_set_grad_enabled",
                    "wrap_with_autocast",
                }:
                    raise ValueError("unsupported nested higher-order operator")
            elif target.__module__ != "_operator" or target.__name__ != "getitem":
                raise ValueError("unsupported non-ATen function in subgraph")
        records.append(
            {
                "path": path,
                "graph": child.code,
                "sha256": hashlib.sha256(child.code.encode()).hexdigest(),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-getter-diagnostic", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "vlaforge.actual_subgraph_examples/1",
        "status": "extracting",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "tool": file_digest(Path(__file__)),
        "performance_measured": False,
        "original_storage_addresses_preserved": False,
        "discarded_warmup": 0,
        "cold_and_second_reference_retained": True,
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        import numpy as np
        import torch
        from torch.utils._pytree import tree_flatten

        path, source, context, _, regions = _capture_source(
            args.capture_report, args.capture_sha256
        )
        device = _saved_device(source)
        if args.cpu_getter_diagnostic and device.type != "cpu":
            raise ValueError("getter diagnostic is restricted to original CPU input")
        entry = regions[args.region]
        if not entry["capture_evidence"]["effect_audit"]["passed"]:
            raise ValueError("source Region failed effect audit")
        report.update(
            source_report={"path": str(path), **file_digest(path)},
            source_archive=entry["archive"],
            source_input={
                "path": str(path.parent / "prepared_inputs.npz"),
                **source["prepared_inputs"],
            },
            source_invocation_ir=source["capture"]["invocation_ir"],
            source_region=args.region,
            source_node=args.node,
            source_numerical_context=context.to_dict(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            device=str(device),
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            threads=torch.get_num_threads(),
            cpu_getter_diagnostic=args.cpu_getter_diagnostic,
            scope="actual prefix execution stopped immediately after selected node",
        )
        save()

        class NodeReached(Exception):
            pass

        records = []
        extracted = {}

        def extract():
            program = torch.export.load(entry["archive"]["path"])
            module = program.module()
            names = [item["name"] for item in entry["capture_evidence"]["inputs"]]
            with np.load(
                path.parent / "prepared_inputs.npz", allow_pickle=False
            ) as pack:
                inputs = tuple(
                    torch.from_numpy(pack[name]).to(device) for name in names
                )
            selected = next(
                (n for n in module.graph.nodes if n.name == args.node), None
            )
            if selected is None or str(selected.target) != "wrap_with_set_grad_enabled":
                raise ValueError("selected node is not the explicit grad-state wrapper")

            class Recorder(torch.fx.Interpreter):
                def run_node(self, node):
                    if node.name != args_node:
                        return super().run_node(node)
                    positional, keywords = self.fetch_args_kwargs_from_env(node)
                    if keywords or len(positional) < 3 or positional[0] is not False:
                        raise ValueError("expected read-only no-grad wrapper arguments")
                    body, values = positional[1], positional[2:]
                    if not all(isinstance(value, torch.Tensor) for value in values):
                        raise ValueError("subgraph requires explicit tensor arguments")
                    before = _fingerprint(values)
                    actual = super().run_node(node)
                    if before != _fingerprint(values):
                        raise ValueError("selected subgraph mutated its input tensors")
                    copied_values, _ = _snapshot_arguments(values, {})
                    copied_outputs, _ = _snapshot_arguments(tuple(actual), {})
                    records.append({"inputs": before, "outputs": _fingerprint(actual)})
                    torch.save(
                        {
                            "args": copied_values,
                            "kwargs": {},
                            "reference": copied_outputs,
                        },
                        output / f"invocation-{len(records):02d}.pt",
                    )
                    extracted.update(
                        body=body,
                        args=copied_values,
                        reference=copied_outputs,
                        target=node.target,
                    )
                    raise NodeReached()

            args_node = args.node
            with torch.inference_mode():
                for _ in range(2):
                    try:
                        Recorder(module).run(*inputs)
                    except NodeReached:
                        pass
            if len(records) != 2 or records[0]["inputs"] != records[1]["inputs"]:
                raise ValueError("first and second invocation inputs differ")

        if args.cpu_getter_diagnostic:
            # Deliberately reproduces the prior fresh-worker diagnostic sequence.
            torch.set_float32_matmul_precision(context.float32_matmul_precision)
            torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
            extract()
        else:
            with offline_restore(context, acknowledge_process_global=True):
                extract()

        report["invocations"] = records
        report["cold_equals_second"] = records[0]["outputs"] == records[1]["outputs"]
        report["subgraphs"] = _read_only_subgraph(extracted["body"])
        report["observed_context_after_extraction"] = snapshot().to_dict()
        save()

        class Subgraph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.body = extracted["body"]

            def forward(self, *values):
                return extracted["target"](False, self.body, *values)

        independent = Subgraph()
        folder = output / args.node
        folder.mkdir()
        with (
            offline_restore(context, acknowledge_process_global=True),
            torch.inference_mode(),
        ):
            expected, expected_tree = tree_flatten(extracted["reference"])
            actual, actual_tree = tree_flatten(independent(*extracted["args"]))
            if expected_tree != actual_tree:
                raise ValueError("independent subgraph output tree differs")
            metrics = [
                _tensor_metrics(a, b) for a, b in zip(expected, actual, strict=True)
            ]
            report["independent_vs_second_metrics"] = metrics
            save()
            if not all(item["exact"] for item in metrics):
                raise ValueError(
                    "independent subgraph differs from second actual invocation"
                )
            torch.save(
                {
                    "args": extracted["args"],
                    "kwargs": {},
                    "reference": extracted["reference"],
                },
                folder / "inputs.pt",
            )
            exported = torch.export.export(independent, extracted["args"], strict=True)
            torch.export.save(exported, folder / "operator.pt2")
            reloaded, reloaded_tree = tree_flatten(
                torch.export.load(folder / "operator.pt2").module()(*extracted["args"])
            )
            if reloaded_tree != expected_tree:
                raise ValueError("reloaded output tree differs")
            reloaded_metrics = [
                _tensor_metrics(a, b) for a, b in zip(expected, reloaded, strict=True)
            ]
            report["reloaded_vs_second_metrics"] = reloaded_metrics
            if not all(item["exact"] for item in reloaded_metrics):
                raise ValueError(
                    "reloaded subgraph differs from actual second invocation"
                )
        report["files"] = {
            str(p.relative_to(output)): file_digest(p)
            for p in output.rglob("*")
            if p.is_file() and p.name != "report.json"
        }
        report["status"] = "extracted_and_eager_verified"
        report["benchmark_reference"] = (
            "second actual invocation; cold reference separately retained"
        )
        report["examples"] = [
            {
                "node": args.node,
                "signature": {
                    "target": str(extracted["target"]),
                    "arguments": records[1]["inputs"],
                    "outputs": records[1]["outputs"],
                },
                "metrics": metrics,
                "inputs_sha256": file_digest(folder / "inputs.pt")["sha256"],
                "export_sha256": file_digest(folder / "operator.pt2")["sha256"],
            }
        ]
        save()
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
