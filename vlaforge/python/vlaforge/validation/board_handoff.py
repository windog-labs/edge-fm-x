"""Portable real-input handoffs, deliberately separate from board acceptance.

Sources can be measured CUDA protocols or explicit data-only contracts. Source
binaries, latencies and device assignments never establish board deployment.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath

from vlaforge.ir.serializer import module_from_data
from vlaforge.validation.session_benchmark import (
    MULTI_SCHEMA,
    benchmark_output_contract,
    decode_tensor,
    encode_output_reference,
    multi_output_declarations,
    numeric_metrics,
    tensor_bytes,
    validate_protocol,
)

SCHEMA = "vlaforge.board_input_handoff/1"
DATA_SCHEMA = "vlaforge.board_data_protocol/1"
TARGET_SCHEMA = "vlaforge.board_target/1"
DRIVER_SCHEMA = "vlaforge.board_target_driver/1"
STAGES = ("preflight", "build", "run", "collect")
TARGETS = {
    "orin-cuda": {"accelerator": "cuda", "provider_contract": "cuda-region-session"},
    "horizon-bpu": {"accelerator": "bpu", "provider_contract": "bpu-region-session"},
}


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA256")
    return value


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("missing or unknown fields; expected " + repr(sorted(expected)))


def safe_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("expected relative POSIX path")
    path = PurePosixPath(relative)
    if path.is_absolute() or path.as_posix() != relative or any(p in (".", "..") for p in path.parts):
        raise ValueError("unsafe or noncanonical relative path")
    candidate = Path(root).joinpath(*path.parts)
    current = Path(root)
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("handoff cannot contain symlinks")
    if not candidate.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("path escapes handoff")
    return candidate


def file_record(root, relative):
    path = safe_path(root, relative)
    if not path.is_file():
        raise ValueError("missing regular file: " + relative)
    return {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def check_file(root, record):
    _keys(record, ("path", "size_bytes", "sha256"))
    _digest(record["sha256"])
    if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
        raise ValueError("invalid file size")
    if file_record(root, record["path"]) != record:
        raise ValueError("file size or SHA256 mismatch: " + record["path"])
    return safe_path(root, record["path"])


def target_template(family):
    if family not in TARGETS:
        raise ValueError("unknown target family")
    return {"schema": TARGET_SCHEMA, "family": family, "architecture": "aarch64",
            **TARGETS[family], "sku": None, "sdk_versions": None,
            "memory_bytes": None, "provider": None, "driver": None,
            "device_access": None, "execution_partition": None,
            "clock_power_thermal_policy": None}


def target_pending(target):
    expected = target_template(target.get("family"))
    _keys(target, expected)
    for key in ("schema", "family", "architecture", "accelerator", "provider_contract"):
        if target[key] != expected[key]:
            raise ValueError("target family/architecture/provider contract mismatch")
    pending = []
    for key in ("sku", "clock_power_thermal_policy"):
        if target[key] is None:
            pending.append(key)
        elif not isinstance(target[key], str) or not target[key].strip():
            raise ValueError("invalid target " + key)
    for key in ("sdk_versions", "execution_partition"):
        value = target[key]
        if value is None:
            pending.append(key)
        elif (not isinstance(value, dict) or not value or
              any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in value.items())):
            raise ValueError("expected explicit nonempty target mapping: " + key)
    memory = target["memory_bytes"]
    if memory is None:
        pending.append("memory_bytes")
    elif type(memory) is not int or memory < 1:
        raise ValueError("invalid target memory")
    for key in ("provider", "driver"):
        value = target[key]
        if value is None:
            pending.append(key)
            continue
        _keys(value, ("contract", "path", "sha256"))
        if value["contract"] != (DRIVER_SCHEMA if key == "driver" else target["provider_contract"]):
            raise ValueError("provider or driver contract differs")
        if not isinstance(value["path"], str) or not Path(value["path"]).is_absolute():
            raise ValueError("driver/provider must be explicitly installed absolute paths")
        _digest(value["sha256"])
    access = target["device_access"]
    if access is None:
        pending.append("device_access")
    elif (not isinstance(access, list) or not access or len(set(access)) != len(access)
          or any(not isinstance(p, str) or not p.startswith("/dev/") for p in access)):
        raise ValueError("device access requires explicit unique /dev paths")
    return pending


def _validate_source_protocol(protocol):
    if protocol.get("schema") != DATA_SCHEMA:
        validate_protocol(protocol)
        return
    _keys(protocol, ("schema", "boundary", "checkpoint_sha256", "outputs", "samples"))
    _digest(protocol["checkpoint_sha256"])
    if not isinstance(protocol["boundary"], str) or not protocol["boundary"].strip():
        raise ValueError("data protocol requires an explicit input/output boundary")
    names = {item["name"] for item in multi_output_declarations(protocol)}
    samples = protocol["samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("data protocol requires nonempty ordered samples")
    identities = set()
    for sample in samples:
        _keys(sample, ("sample_id", "inputs", "outputs"))
        identity = sample["sample_id"]
        if (type(identity) not in (int, str) or identity == ""
                or (type(identity) is int and identity < 0) or identity in identities):
            raise ValueError("data protocol requires unique nonempty string or nonnegative integer sample IDs")
        identities.add(identity)
        inputs = sample["inputs"]
        if (not isinstance(inputs, dict) or not inputs
                or any(not isinstance(name, str) or not name for name in inputs)
                or any(not isinstance(path, str) or not path for path in inputs.values())):
            raise ValueError("data sample requires named input file paths")
        _keys(sample["outputs"], names)
        for reference in sample["outputs"].values():
            _keys(reference, ("direct", "eager"))
            if any(not isinstance(path, str) or not path for path in reference.values()):
                raise ValueError("data output requires direct and eager reference file paths")


def _contracts(module, protocol):
    output = benchmark_output_contract(module, protocol)
    if protocol["schema"] in (MULTI_SCHEMA, DATA_SCHEMA):
        outputs = output["outputs"]
    else:
        outputs = [dict(output, name=module.outputs[0].name, role="primary-action", index=0)]
    inputs = []
    for port in module.inputs:
        payload = port.payload
        if payload.layout != "contiguous":
            raise ValueError("handoff v1 only supports complete contiguous tensor storage")
        inputs.append({"name": port.name, "dtype": payload.dtype,
                       "shape": list(payload.shape), "layout": "contiguous",
                       "byte_order": "little", "size_bytes": tensor_bytes(payload.dtype, payload.shape)})
    return inputs, outputs


def prepare(protocol_path, module_path, provenance_path, destination):
    """Copy bytes once; a manifest is published only after independent validation."""
    import numpy as np

    protocol, module_data = read_json(protocol_path), read_json(module_path)
    _validate_source_protocol(protocol)
    module = module_from_data(module_data)
    inputs, outputs = _contracts(module, protocol)
    provenance = read_json(provenance_path)
    _keys(provenance, ("schema", "identity", "limitations", "files"))
    if provenance["schema"] != "vlaforge.board_source_selection/1":
        raise ValueError("unsupported provenance selection")
    if not isinstance(provenance["identity"], dict) or not provenance["identity"]:
        raise ValueError("explicit reference identity required")
    if not isinstance(provenance["limitations"], list) or not provenance["limitations"]:
        raise ValueError("explicit reference limitations required")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    files = []

    def copy(source, relative, expected=None):
        source = Path(source)
        if not source.is_file() or source.is_symlink():
            raise ValueError("source must be a regular, non-symlink file")
        before = sha256(source)
        if expected is not None and before != _digest(expected):
            raise ValueError("source SHA256 differs from selection")
        target = safe_path(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as left, target.open("xb") as right:
            shutil.copyfileobj(left, right, 8 * 1024 * 1024)
        record = file_record(root, relative)
        if record["sha256"] != before or sha256(source) != before:
            raise ValueError("source changed during copy")
        files.append(record)
        return relative

    copy(protocol_path, "source/protocol.json")
    copy(module_path, "source/module.json")
    copy(provenance_path, "source/selection.json")
    for index, item in enumerate(provenance["files"]):
        _keys(item, ("path", "sha256", "role"))
        if not isinstance(item["role"], str) or not item["role"]:
            raise ValueError("source role required")
        copy(item["path"], f"provenance/{index:04d}/{Path(item['path']).name}", item["sha256"])
    samples = []
    for index, sample in enumerate(protocol["samples"]):
        if set(sample["inputs"]) != {item["name"] for item in inputs}:
            raise ValueError("sample must contain every input and no unknown input")
        row = {"sample_id": index, "inputs": {}, "outputs": {}}
        for number, spec in enumerate(inputs):
            relative = f"samples/{index:04d}/input-{number:03d}.bin"
            row["inputs"][spec["name"]] = copy(sample["inputs"][spec["name"]], relative)
        for number, spec in enumerate(outputs):
            reference = sample["outputs"][spec["name"]] if protocol["schema"] in (MULTI_SCHEMA, DATA_SCHEMA) else sample
            entry = {}
            for kind in ("eager", "direct"):
                original = copy(reference[kind], f"samples/{index:04d}/output-{number:03d}-{kind}.npy")
                raw = encode_output_reference(np.load(root / original, allow_pickle=False), spec)
                relative = f"samples/{index:04d}/output-{number:03d}-{kind}.bin"
                with (root / relative).open("xb") as stream:
                    stream.write(raw)
                files.append(file_record(root, relative))
                entry[kind] = {"storage": relative, "original_npy": original}
            row["outputs"][spec["name"]] = entry
        samples.append(row)
    for family in TARGETS:
        relative = f"targets/{family}.json"
        write_new(root / relative, target_template(family))
        files.append(file_record(root, relative))
    manifest = {"schema": SCHEMA, "scope": "data-and-reference-only",
                "board_executed": False, "board_verified": False,
                "physical_units_verified": False, "robot_calibration_verified": False,
                "source_protocol_sha256": sha256(protocol_path),
                "checkpoint_sha256": _digest(protocol["checkpoint_sha256"]),
                "identity": provenance["identity"], "limitations": provenance["limitations"],
                "source_boundary": protocol["boundary"], "inputs": inputs,
                "outputs": outputs, "samples": samples, "files": files}
    result = validate(root, manifest=manifest)
    # Publish a fully flushed completion record atomically, without replacement.
    with tempfile.NamedTemporaryFile(mode="w", dir=root.parent, prefix=".board-manifest-") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, root / "manifest.json")
    return result


def validate(root, *, manifest=None):
    import numpy as np

    root = Path(root)
    if root.is_symlink() or (root / "manifest.json").is_symlink():
        raise ValueError("handoff root and manifest cannot be symlinks")
    manifest = read_json(root / "manifest.json") if manifest is None else manifest
    _keys(manifest, ("schema", "scope", "board_executed", "board_verified",
                    "physical_units_verified", "robot_calibration_verified",
                    "source_protocol_sha256", "checkpoint_sha256", "identity",
                    "limitations", "source_boundary", "inputs", "outputs", "samples", "files"))
    if manifest["schema"] != SCHEMA or manifest["scope"] != "data-and-reference-only":
        raise ValueError("unsupported board handoff")
    for name in ("board_executed", "board_verified", "physical_units_verified", "robot_calibration_verified"):
        if manifest[name] is not False:
            raise ValueError("offline data validation cannot assert " + name)
    records = {}
    for item in manifest["files"]:
        path = check_file(root, item)
        if item["path"] in records:
            raise ValueError("duplicate file record")
        records[item["path"]] = path
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual_files - {"manifest.json"} != set(records):
        raise ValueError("unlisted or missing handoff files")
    protocol = read_json(records["source/protocol.json"])
    _validate_source_protocol(protocol)
    if sha256(records["source/protocol.json"]) != manifest["source_protocol_sha256"]:
        raise ValueError("source protocol binding differs")
    if manifest["checkpoint_sha256"] != _digest(protocol["checkpoint_sha256"]):
        raise ValueError("checkpoint identity differs")
    inputs, outputs = _contracts(module_from_data(read_json(records["source/module.json"])), protocol)
    if inputs != manifest["inputs"] or outputs != manifest["outputs"] or manifest["source_boundary"] != protocol["boundary"]:
        raise ValueError("typed contracts or boundary differ from original IR/protocol")
    selection = read_json(records["source/selection.json"])
    if manifest["identity"] != selection["identity"] or manifest["limitations"] != selection["limitations"]:
        raise ValueError("source identity/limitations changed")
    for index, item in enumerate(selection["files"]):
        relative = f"provenance/{index:04d}/{Path(item['path']).name}"
        if sha256(records[relative]) != item["sha256"]:
            raise ValueError("provenance differs from fixed source selection")
    if len(manifest["samples"]) != len(protocol["samples"]):
        raise ValueError("missing complete source samples")
    comparisons = []
    for index, row in enumerate(manifest["samples"]):
        _keys(row, ("sample_id", "inputs", "outputs"))
        if type(row["sample_id"]) is not int or row["sample_id"] != index:
            raise ValueError("source sample order changed")
        if set(row["inputs"]) != {item["name"] for item in inputs} or set(row["outputs"]) != {item["name"] for item in outputs}:
            raise ValueError("sample must contain every ABI input and output")
        for number, spec in enumerate(inputs):
            if row["inputs"][spec["name"]] != f"samples/{index:04d}/input-{number:03d}.bin":
                raise ValueError("input/sample storage order changed")
            decode_tensor(records[row["inputs"][spec["name"]]].read_bytes(), spec["dtype"], spec["shape"])
        for number, spec in enumerate(outputs):
            entry = row["outputs"][spec["name"]]
            _keys(entry, ("direct", "eager"))
            values, storage = {}, {}
            for kind in ("direct", "eager"):
                _keys(entry[kind], ("storage", "original_npy"))
                if entry[kind] != {"storage": f"samples/{index:04d}/output-{number:03d}-{kind}.bin",
                                   "original_npy": f"samples/{index:04d}/output-{number:03d}-{kind}.npy"}:
                    raise ValueError("output/sample storage order changed")
                raw = records[entry[kind]["storage"]].read_bytes()
                value = np.load(records[entry[kind]["original_npy"]], allow_pickle=False)
                if raw != encode_output_reference(value, spec):
                    raise ValueError("complete reference storage differs from original NPY")
                values[kind] = decode_tensor(raw, spec["dtype"], spec["shape"])
                storage[kind] = raw
            exact = storage["eager"] == storage["direct"]
            if spec["role"] == "exact" and not exact:
                raise ValueError("exact output reference differs")
            comparison = {"sample_id": index, "output": spec["name"], "complete_bitwise_equal": exact}
            if spec["role"] != "exact":
                comparison["metrics"] = numeric_metrics(values["eager"], values["direct"])
            comparisons.append(comparison)
    pending = {family: target_pending(read_json(records[f"targets/{family}.json"])) for family in TARGETS}
    return {"schema": "vlaforge.board_handoff_validation/1", "status": "passed",
            "scope": "offline-data-integrity-only", "sample_count": len(manifest["samples"]),
            "file_count": len(records), "total_bytes": sum(item["size_bytes"] for item in manifest["files"]),
            "board_executed": False, "board_verified": False,
            "physical_units_verified": False, "robot_calibration_verified": False,
            "target_pending": pending, "comparisons": comparisons}


def stage(pack, descriptor, phase, output):
    """Invoke a separately supplied target driver; driver success is not acceptance.

    SDK/provider checks, no-Python build/run, owner supervision, complete metrics,
    clocks/thermals and mixed-device accounting belong to the pinned target driver.
    """
    if phase not in STAGES:
        raise ValueError("unsupported board stage")
    validate(pack)
    target = read_json(descriptor)
    pending = target_pending(target)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    report = {"schema": "vlaforge.board_stage_dispatch/1", "stage": phase,
              "descriptor_sha256": sha256(descriptor), "manifest_sha256": sha256(Path(pack) / "manifest.json"),
              "board_executed": False, "board_verified": False,
              "status": "pending", "pending": pending,
              "host": {"machine": platform.machine(), "system": platform.system()},
              "started_unix_ns": time.time_ns()}
    if pending:
        write_new(root / "dispatch.json", report)
        return report
    if phase in ("preflight", "run") and platform.machine() != target["architecture"]:
        report["pending"] = ["actual-board-architecture"]
        write_new(root / "dispatch.json", report)
        return report
    for kind in ("provider", "driver"):
        record = target[kind]
        if not Path(record["path"]).is_file() or sha256(record["path"]) != record["sha256"]:
            raise ValueError("installed " + kind + " differs from pinned descriptor")
    if phase in ("preflight", "run") and any(not os.access(p, os.R_OK | os.W_OK) for p in target["device_access"]):
        raise ValueError("required board devices unavailable")
    command = [target["driver"]["path"], phase, "--pack", str(Path(pack).resolve()),
               "--target-descriptor", str(Path(descriptor).resolve()), "--output", str(root.resolve() / "driver")]
    report.update(status="started", command=command)
    write_new(root / "started.json", report)
    with (root / "driver.log").open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        report["pid"] = process.pid
        write_new(root / "process.json", report)
        report["returncode"] = process.wait()
    report.update(status="driver-returned" if report["returncode"] == 0 else "failed",
                  ended_unix_ns=time.time_ns(), acceptance="requires-independent-board-evidence-audit")
    write_new(root / "dispatch.json", report)
    return report
