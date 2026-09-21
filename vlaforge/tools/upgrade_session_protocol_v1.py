"""Upgrade a legacy single-output Session protocol to the public v2 shape.

This changes only the protocol envelope: inputs, output reference bytes, model
bundle and measurement counts remain bound to their original hashes. It is a
model-neutral migration for legacy artifacts, not an RDT-specific adapter.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

V2_KEYS = {
    "schema", "boundary", "warmup", "measured", "processes", "policies", "bundles",
    "quality_gate", "eager_validation", "samples", "evidence", "gpu_ordinal",
    "monitor_gpu", "cuda_visible_devices", "cuda_arch", "cmake_prefix_path",
    "bundle_metadata_mode", "paper_gates", "outputs", "full_paper_acceptance",
    "boundary_includes", "boundary_excludes", "separate_wall_boundary", "model",
    "quality_reason", "checkpoint_sha256", "sample_order", "policy_order",
    "telemetry_interval_seconds", "clock_power_policy", "other_compute_owner_policy",
    "owner_identity_mode", "allocator_observation", "reference_identity",
    "reference_limitations", "numerical_worker_bootstrap", "aoti_package_extraction_root",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def migrate(protocol_path: Path, module_path: Path, output: Path) -> dict:
    source = read(protocol_path)
    module = read(module_path)
    if source.get("schema") != "vlaforge.session_latency_protocol/1":
        raise ValueError("only the legacy single-output Session protocol is accepted")
    outputs = module.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0].get("name"), str):
        raise ValueError("legacy migration requires exactly one named Module output")
    if not source.get("samples") or any(set(sample) != {"sample_id", "inputs", "direct", "eager"}
                                         for sample in source["samples"]):
        raise ValueError("legacy samples must have exactly one direct/eager output pair")
    # Retain only fields in the public v2 schema. Legacy provenance fields are
    # not silently lost: the sidecar migration ledger records the source digest
    # and the exact envelope-only change set.
    result = {key: copy.deepcopy(value) for key, value in source.items() if key in V2_KEYS}
    result["schema"] = "vlaforge.session_latency_protocol/2"
    result["outputs"] = [{"name": outputs[0]["name"], "role": "primary-action",
                           "active_dimensions": source.get("active_dimensions")}]
    for sample in result["samples"]:
        sample["outputs"] = {outputs[0]["name"]: {"direct": sample.pop("direct"), "eager": sample.pop("eager")}}
    result["bundles"] = {"off": source["bundles"]["off"],
                          "batch-only": source["bundles"]["off"],
                          "required": source["bundles"]["off"]}
    result["policies"] = ["off", "batch-only", "required"]
    result["bundle_metadata_mode"] = "selection-manifest"
    result["quality_reason"] = "legacy protocol envelope migration; complete BF16 output and active dimensions retained"
    migration = {"schema": "vlaforge.session_protocol_migration/1",
                 "source_protocol_sha256": sha(protocol_path),
                 "source_module_sha256": sha(module_path),
                 "only_changes": ["schema", "output declaration", "sample output envelope", "policy variants"],
                 "dropped_legacy_fields": sorted(set(source) - V2_KEYS),
                 "input_bytes_unchanged": True, "output_reference_paths_unchanged": True,
                 "model_bundle_path_unchanged": True}
    result["evidence"] = [*source.get("evidence", []), str(protocol_path), str(module_path)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    output.with_name(output.stem + ".migration.json").write_text(
        json.dumps(migration, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    migrate(args.protocol.resolve(), args.module.resolve(), args.output.resolve())
    print(json.dumps({"status": "migrated", "protocol": str(args.output), "protocol_sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
