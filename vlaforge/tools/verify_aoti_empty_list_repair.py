"""Verify a public-CLI scalar-factory candidate on its captured real examples."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--export-sha", required=True)
    parser.add_argument("--compile-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import torch._inductor.codecache  # noqa: F401 - Torch package loader lazy import.
    from vlaforge.deployment.aoti_export import backend_pass_records, backend_program_pass_records
    from vlaforge.deployment.aoti_package import verify_package_audit
    from vlaforge.deployment.aoti_profile import aoti_configs

    record = json.loads(args.compile_manifest.read_text())
    configs = aoti_configs("aten-preserving")
    artifact = Path(record["artifact"]["path"])
    if (
        sha(args.export) != args.export_sha
        or record["exported_program"]["sha256"] != args.export_sha
        or record["status"] != "passed"
        or record["inductor_configs"] != configs
        or record["backend_program_audit"]["passes"] != backend_program_pass_records(configs)
        or record["backend_graph_passes"] != backend_pass_records(configs)
        or record["artifact"]["sha256"] != sha(artifact)
    ):
        raise ValueError("candidate is not bound to the public compiler and captured export")
    verify_package_audit(artifact, configs, record["backend_package_audit"])
    program = torch.export.load(args.export)
    positional, keywords = program.example_inputs
    torch.save({"args": positional, "kwargs": keywords}, args.output / "actual-inputs.pt")
    with torch.inference_mode():
        expected = program.module()(*positional, **keywords)
        implementation = torch._inductor.aoti_load_package(
            str(artifact), run_single_threaded=True, device_index=0
        )
        outputs = [implementation(*positional, **keywords) for _ in range(3)]
        torch.cuda.synchronize()
    if expected.shape != torch.Size([]) or expected.dtype != torch.float32:
        raise ValueError("this regression expects the original zero-dimensional FP32 factory")
    np.save(args.output / "expected.npy", expected.cpu().numpy(), allow_pickle=False)
    results = []
    for index, actual in enumerate(outputs):
        np.save(args.output / f"actual-{index}.npy", actual.cpu().numpy(), allow_pickle=False)
        results.append({
            "shape": list(actual.shape), "dtype": str(actual.dtype), "device": str(actual.device),
            "bitwise_equal": actual.shape == expected.shape and actual.dtype == expected.dtype
            and actual.device == expected.device and torch.equal(expected, actual),
        })
    report = {
        "status": "isolated_public_cli_region_passed" if all(row["bitwise_equal"] for row in results) else "numerics_failed",
        "full_model_acceptance": False,
        "command": [sys.executable, *sys.argv], "tool_sha256": sha(__file__),
        "compile_manifest_sha256": sha(args.compile_manifest),
        "export_sha256": sha(args.export), "artifact_sha256": sha(artifact),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "expected": {"shape": list(expected.shape), "dtype": str(expected.dtype), "value": expected.item()},
        "calls": results,
        "files": {path.name: sha(path) for path in args.output.iterdir() if path.is_file()},
        "backend_program_audit": record["backend_program_audit"],
        "backend_graph_rewrites": record["backend_graph_rewrites"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "isolated_public_cli_region_passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
