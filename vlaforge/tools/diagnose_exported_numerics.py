"""Compile independent intermediate-output probes with explicit source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_capture_sources(manifest_path, program_path, evidence_path):
    manifest = json.loads(manifest_path.read_text())
    program_sha = digest(program_path)
    evidence_sha = digest(evidence_path)
    matches = [
        region
        for region in manifest["regions"]
        if region["export_sha256"] == program_sha
        and region["capture_sha256"] == evidence_sha
    ]
    if len(matches) != 1:
        raise ValueError("capture manifest must bind exactly this program/evidence pair")
    return {
        "capture_manifest": digest(manifest_path),
        "exported_program": program_sha,
        "capture_evidence": evidence_sha,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exported-program", type=Path, required=True)
    parser.add_argument("--capture-evidence", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--node", action="append", required=True)
    parser.add_argument("--inductor-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("probe output must be new")
    started_at = datetime.now(timezone.utc).isoformat()
    source = verify_capture_sources(
        args.capture_manifest, args.exported_program, args.capture_evidence
    )

    import numpy as np
    import torch
    import torch._inductor.codecache
    from torch.export.graph_signature import InputKind, TensorArgument
    from vlaforge.analysis import numerical_probe
    from vlaforge.analysis.numerical_probe import tensor_difference, tensor_probe_module
    from vlaforge.frontend.region_capture import _graph_digest

    if not torch.cuda.is_available():
        raise ValueError("CUDA numerical diagnosis requires the actual GPU")
    args.output.mkdir(parents=True)
    evidence = json.loads(args.capture_evidence.read_text())
    program = torch.export.load(args.exported_program)
    # Export serialization may rename graph values; file hashes bind provenance.
    loaded_graph_digest = _graph_digest(program)
    user_specs = [
        spec
        for spec in program.graph_signature.input_specs
        if spec.kind == InputKind.USER_INPUT
    ]
    if len(user_specs) != len(evidence["inputs"]) or any(
        not isinstance(spec.arg, TensorArgument) for spec in user_specs
    ):
        raise ValueError("probe requires flat, captured tensor-only inputs")
    nodes = {node.name: node for node in program.graph.nodes}
    with np.load(args.input_npz, allow_pickle=False) as inputs:
        values = []
        for contract, spec in zip(evidence["inputs"], user_specs, strict=True):
            value = torch.from_numpy(inputs[contract["name"]].copy()).to(
                contract["device"]
            )
            expected = nodes[spec.arg.name].meta["val"]
            if (
                list(value.shape) != contract["type"]["shape"]
                or value.dtype != expected.dtype
                or value.device != expected.device
            ):
                raise ValueError(
                    "probe input shape/dtype/device differs from captured profile"
                )
            values.append(value)
    configs = json.loads(args.inductor_config.read_text())
    source.update({
        "input_npz": digest(args.input_npz),
        "inductor_config": digest(args.inductor_config),
        "tool": digest(Path(__file__)),
        "probe_implementation": digest(Path(numerical_probe.__file__)),
    })
    module = tensor_probe_module(program, tuple(args.node))
    (args.output / "probe_graph.py").write_text(module.code)
    with torch.inference_mode():
        reference = module(*values)
        prepared = torch.export.export(module, tuple(values), strict=True)
    torch.export.save(prepared, args.output / "probe.pt2")
    start = time.perf_counter()
    artifact = torch._inductor.aoti_compile_and_package(
        prepared,
        package_path=str(args.output / "compiled.pt2"),
        inductor_configs=configs,
    )
    compile_seconds = time.perf_counter() - start
    runner = torch._inductor.aoti_load_package(artifact)
    with torch.inference_mode():
        candidate = runner(*values)
    torch.cuda.synchronize()
    report = {
        "schema": "vlaforge.numerical_probe/1",
        "status": "diagnostic_completed",
        "source_sha256": source,
        "started_at": started_at,
        "command": [sys.executable, *sys.argv],
        "probe_outputs": "immediate_tensor_snapshots_before_later_alias_mutation",
        "capture_graph_digest": evidence["graph_digest"],
        "loaded_graph_digest": loaded_graph_digest,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "inductor_configs": configs,
        "compile_seconds": compile_seconds,
        "artifact_sha256": digest(Path(artifact)),
        "nodes": {
            name: tensor_difference(left, right)
            for name, left, right in zip(args.node, reference, candidate, strict=True)
        },
        "limitation": "extra outputs and dead-code elimination change fusion; this is not original-artifact intermediate evidence or a performance result",
    }
    torch.save(
        {
            "reference": [item.cpu() for item in reference],
            "candidate": [item.cpu() for item in candidate],
        },
        args.output / "raw_tensors.pt",
    )
    report["raw_tensors_sha256"] = digest(args.output / "raw_tensors.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
