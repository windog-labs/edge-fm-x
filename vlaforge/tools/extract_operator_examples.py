"""Save actual ATen workloads as independent exported programs and tensor packs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from diagnose_exported_numerics import digest, verify_capture_sources


def load_runtime_inputs(path, evidence, program):
    import numpy as np
    import torch
    from torch.export.graph_signature import InputKind, TensorArgument

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as pack:
            values = tuple(
                torch.from_numpy(pack[item["name"]].copy())
                for item in evidence["inputs"]
            )
    else:
        values = torch.load(path, map_location="cpu", weights_only=True)
    specs = [
        item
        for item in program.graph_signature.input_specs
        if item.kind == InputKind.USER_INPUT
    ]
    if (
        not isinstance(values, tuple)
        or len(values) != len(evidence["inputs"])
        or len(specs) != len(values)
        or any(not isinstance(item.arg, TensorArgument) for item in specs)
    ):
        raise ValueError("runtime inputs must match the flat captured tensor tuple")
    nodes = {node.name: node for node in program.graph.nodes}
    inputs = []
    for value, contract, spec in zip(values, evidence["inputs"], specs, strict=True):
        expected = nodes[spec.arg.name].meta["val"]
        if (
            not isinstance(value, torch.Tensor)
            or list(value.shape) != contract["type"]["shape"]
            or value.dtype != expected.dtype
            or str(expected.device) != contract["device"]
        ):
            raise ValueError("runtime input shape/dtype/device differs from capture")
        inputs.append(value.to(contract["device"]))
    return tuple(inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exported-program", type=Path, required=True)
    parser.add_argument("--capture-evidence", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--runtime-inputs", type=Path, required=True)
    parser.add_argument("--runtime-inputs-sha256", required=True)
    parser.add_argument("--node", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = verify_capture_sources(
        args.capture_manifest, args.exported_program, args.capture_evidence
    )
    if digest(args.runtime_inputs) != args.runtime_inputs_sha256:
        raise ValueError("runtime inputs differ from the bound example SHA256")
    if args.output.exists():
        raise ValueError("operator extraction output must be new")

    import torch
    from torch.utils._pytree import tree_flatten
    from vlaforge.analysis import operator_capture
    from vlaforge.analysis.numerical_probe import tensor_difference

    if not torch.cuda.is_available():
        raise ValueError(
            "this CLI extracts the real CUDA workload on its target device"
        )
    evidence = json.loads(args.capture_evidence.read_text())
    if not evidence["effect_audit"]["passed"]:
        raise ValueError("source graph did not pass the actual effect audit")
    program = torch.export.load(args.exported_program)
    inputs = load_runtime_inputs(args.runtime_inputs, evidence, program)
    examples = operator_capture.capture_operator_examples(
        program, tuple(inputs), node_names=tuple(args.node)
    )
    args.output.mkdir(parents=True)
    report = {
        "schema": "vlaforge.actual_operator_examples/1",
        "status": "extracting",
        "source_sha256": {
            **source,
            "runtime_inputs": args.runtime_inputs_sha256,
            "tool": digest(Path(__file__)),
            "implementation": digest(Path(operator_capture.__file__)),
        },
        "command": [sys.executable, *sys.argv],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "performance_measured": False,
        "original_storage_addresses_preserved": False,
        "examples": [],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with torch.inference_mode():
        for example in examples:
            output = example.module()(*example.args, **example.kwargs)
            expected, expected_tree = tree_flatten(example.reference)
            actual, actual_tree = tree_flatten(output)
            if expected_tree != actual_tree:
                raise ValueError("independent operator output structure differs")
            metrics = [
                tensor_difference(left, right)
                for left, right in zip(expected, actual, strict=True)
            ]
            if not all(item["bitwise_equal"] for item in metrics):
                raise ValueError(
                    "independent operator did not reproduce its actual reference"
                )
            folder = args.output / example.node
            folder.mkdir()
            torch.save(
                {
                    "args": example.args,
                    "kwargs": example.kwargs,
                    "reference": example.reference,
                },
                folder / "inputs.pt",
            )
            exported = torch.export.export(
                example.module(), example.args, example.kwargs, strict=True
            )
            torch.export.save(exported, folder / "operator.pt2")
            report["examples"].append(
                {
                    "node": example.node,
                    "signature": example.signature,
                    "metrics": metrics,
                    "inputs_sha256": digest(folder / "inputs.pt"),
                    "export_sha256": digest(folder / "operator.pt2"),
                }
            )
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "extracted_and_eager_verified"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
