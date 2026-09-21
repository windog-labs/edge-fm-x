"""Build a formal Session benchmark protocol from a validated native bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DTYPES = {
    "f32": ("<f4", 4),
    "f64": ("<f8", 8),
    "f16": ("<f2", 2),
    "bf16": ("<u2", 2),
    "i64": ("<i8", 8),
    "i32": ("<i4", 4),
    "u64": ("<u8", 8),
    "u8": ("u1", 1),
    "bool": ("?", 1),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def decode(path: Path, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype not in DTYPES:
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    code, itemsize = DTYPES[dtype]
    raw = np.frombuffer(path.read_bytes(), dtype=code)
    expected = int(np.prod(shape))
    if raw.size != expected:
        raise ValueError(f"tensor size differs from schema: {path}")
    if dtype == "bf16":
        raw = (raw.astype("<u4") << 16).view("<f4")
    return np.ascontiguousarray(raw.reshape(shape))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--allocator-observation", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    validation = read_json(args.validation_dir / "report.json")
    if validation.get("status") != "passed" or validation.get("all_tokens_equal_official") is not True:
        raise ValueError("validation report is not accepted")
    input_schema = read_json(args.bundle / "metadata/input_schema.json")
    output_schema = read_json(args.bundle / "metadata/output_schema.json")
    artifact_outputs = validation.get("artifact_outputs")
    expected_outputs = [
        {
            "index": row["output_id"],
            "name": row["name"],
            "dtype": row["payload"]["dtype"],
            "shape": row["payload"]["shape"],
        }
        for row in output_schema["outputs"]
    ]
    if artifact_outputs != expected_outputs:
        raise ValueError("validation output order differs from the bundle schema")

    samples = []
    data_root = args.validation_dir / "data"
    for index in range(len(validation["samples"])):
        source = data_root / str(index)
        inputs = {}
        for input_port in input_schema["inputs"]:
            path = source / f"{input_port['input_id']}.bin"
            decode(path, input_port["payload"]["dtype"], input_port["payload"]["shape"])
            inputs[input_port["name"]] = str(path.resolve())

        references = {}
        for output in output_schema["outputs"]:
            references[output["name"]] = decode(
                source / f"direct-{output['output_id']}.bin"
                if output["output_id"] != 1
                else source / "direct.bin",
                output["payload"]["dtype"],
                output["payload"]["shape"],
            )
        reference_path = args.output / f"sample-{index:02d}.npz"
        np.savez(reference_path, **references)
        outputs = {
            output["name"]: {
                "direct": str(reference_path.resolve()),
                "eager": str(reference_path.resolve()),
            }
            for output in output_schema["outputs"]
        }
        samples.append(
            {"sample_id": f"validation-{index:02d}", "inputs": inputs, "outputs": outputs}
        )

    declarations = []
    for output in output_schema["outputs"]:
        if output["name"] == "tokens" and output["payload"]["dtype"] in ("i64", "i32", "u64", "u8", "bool"):
            role = "exact"
        elif output["name"] == "logits":
            role = "primary-action"
        else:
            raise ValueError(f"unsupported native multimodal output role: {output['name']}")
        declarations.append({"name": output["name"], "role": role})

    evidence = [
        str((args.validation_dir / "report.json").resolve()),
        str((args.bundle / "bundle.json").resolve()),
        str((args.bundle / "metadata/input_schema.json").resolve()),
        str((args.bundle / "metadata/output_schema.json").resolve()),
    ]
    protocol = {
        "schema": "vlaforge.session_latency_protocol/2",
        "boundary": "host-model-tensor: H2D, input binding, Session run, completion and complete D2H",
        "boundary_includes": [
            "host input staging",
            "H2D",
            "input binding",
            "Session run",
            "completion",
            "complete typed D2H",
        ],
        "boundary_excludes": [
            "image/text preprocessing",
            "checkpoint and artifact initialization",
            "validation and logging IO",
            "detokenization",
        ],
        "separate_wall_boundary": "resident tensor runner host-call timing; not sensor or file IO",
        "warmup": 128,
        "measured": 1024,
        "processes": 5,
        "policies": ["off"],
        "bundles": {"off": str(args.bundle.resolve())},
        "quality_gate": "passed",
        "eager_validation": "bitwise",
        "outputs": declarations,
        "paper_gates": {"mse_max": 0.0, "cosine_min": 1.0},
        "full_paper_acceptance": False,
        "gpu_ordinal": 0,
        "monitor_gpu": args.gpu_uuid,
        "cuda_visible_devices": args.gpu_uuid,
        "cuda_arch": "9.0",
        "cmake_prefix_path": str(Path(__import__("torch").utils.cmake_prefix_path).resolve()),
        "bundle_metadata_mode": "selection-manifest",
        "allocator_observation": "libtorch-native/1" if args.allocator_observation else "off",
        "owner_identity_mode": "cuda-registration-handshake",
        "sample_order": "validation-data-order",
        "policy_order": ["off"],
        "model": validation["model"],
        "reference_identity": "artifact direct output; official token validation is bound separately",
        "reference_limitations": [
            "fixed prompt length 327",
            "fixed image tensor shape 1200x1536",
            "fixed generation length",
            "16 frozen validation images",
        ],
        "evidence": evidence,
        "samples": samples,
    }
    protocol_path = args.output / "protocol.json"
    write_json(protocol_path, protocol)
    print(
        json.dumps(
            {
                "status": "protocol_built",
                "protocol": str(protocol_path),
                "protocol_sha256": sha256(protocol_path),
                "samples": len(samples),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
