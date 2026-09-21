"""Join a build-only bundle to independent typed native execution evidence.

The original build-only and diagnostic records are immutable. This projection
does not relabel either record or claim native-scale robot outputs.
"""

import argparse
import csv
import json
import os
from pathlib import Path

from vlaforge.adapters.openpi.openpi_checkpoint import file_digest

SCHEMA = "vlaforge.openpi_typed_native_join/1"


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), **file_digest(path)}


def checked(item):
    path = Path(item["path"])
    if record(path) != item:
        raise ValueError("typed native source file identity changed")
    return path, json.loads(path.read_text())


def validate_complete_bytes(raw, direct, eager, *, size, calls):
    if type(size) is not int or type(calls) is not int or size <= 0 or calls <= 0:
        raise ValueError("complete native tensor size/call count must be positive integers")
    if len(raw) != size * calls or len(direct) != size or len(eager) != size:
        raise ValueError("complete native output archive length differs")
    if direct != eager or any(raw[index * size:(index + 1) * size] != eager for index in range(calls)):
        raise ValueError("complete native output bytes differ from official or same artifact")


def validate_typed_native_join(join):
    from vlaforge.deployment.numerical import RegionNumericalBinding
    from vlaforge.validation.session_benchmark import validate_rows

    if set(join) != {"schema", "build_report", "diagnostic_report", "prepared", "protocol"} or join["schema"] != SCHEMA:
        raise ValueError("unknown or incomplete typed native join schema")
    build_path, build = checked(join["build_report"])
    diagnostic_path, diagnostic = checked(join["diagnostic_report"])
    prepared_path, prepared = checked(join["prepared"])
    protocol_path, protocol = checked(join["protocol"])
    if (build.get("schema") != "vlaforge.openpi_native_session_audit/1"
            or build.get("status") != "built-unexecuted" or build.get("build_only") is not True
            or build.get("native_executed") is not False):
        raise ValueError("typed native source requires unchanged build-only evidence")
    for key in ("native_executed", "no_python_process_verified", "numerical_provider_enforcement_verified",
                "complete_normalized_output_verified", "complete_bitwise_equal"):
        if diagnostic.get(key) is not True:
            raise ValueError("typed native complete execution gate not passed: " + key)
    if diagnostic.get("status") != "passed" or type(diagnostic.get("exitcode")) is not int or diagnostic["exitcode"] != 0:
        raise ValueError("typed native process did not exit successfully")
    if diagnostic["prepared_sha256"] != join["prepared"]["sha256"] or diagnostic["protocol_sha256"] != join["protocol"]["sha256"]:
        raise ValueError("typed native preparation identity differs")
    root = prepared_path.parent
    bundle = build_path.parent / "bundle"
    if protocol["policies"] != ["off"] or protocol["bundles"] != {"off": str(bundle)} or len(protocol["samples"]) != 1:
        raise ValueError("typed native join currently requires one complete off-policy sample")
    frozen_path = root / "frozen-files.json"
    if file_digest(frozen_path)["sha256"] != prepared["frozen_files_sha256"]:
        raise ValueError("typed native frozen file manifest changed")
    frozen = json.loads(frozen_path.read_text())
    for path, digest in frozen.items():
        if file_digest(Path(path))["sha256"] != digest:
            raise ValueError("typed native frozen source/bundle/input changed: " + path)
    folder = diagnostic_path.parent / "native"
    executable = root / "build/off/vlaforge_generated_runner"
    warmup, measured = diagnostic["warmup_calls"], diagnostic["measured_diagnostic_calls"]
    if any(type(value) is not int or value < 0 for value in (warmup, measured)) or measured == 0:
        raise ValueError("typed native call counts are invalid")
    count = warmup + measured
    command = [str(executable), str(bundle), str(root / "data"), str(folder), str(warmup), str(measured)]
    if diagnostic["command"] != command or frozen.get(str(executable)) != diagnostic["executable_sha256"]:
        raise ValueError("typed native executable/command identity differs")
    monitor_path = folder / "monitor.json"
    if file_digest(monitor_path)["sha256"] != diagnostic["monitor_sha256"]:
        raise ValueError("typed native ownership evidence changed")
    monitor = json.loads(monitor_path.read_text())
    if monitor["status"] != "exited" or type(monitor["exitcode"]) is not int or monitor["exitcode"] != 0 or monitor["command"] != command:
        raise ValueError("typed native ownership monitored process did not succeed")
    maps = diagnostic.get("actual_maps")
    if not maps:
        raise ValueError("typed native actual DSO maps missing")
    for item in maps:
        path = Path(item["path"])
        if file_digest(path)["sha256"] != item["sha256"] or item["pid"] != monitor["container_pid"] or item["executable"] != str(executable):
            raise ValueError("typed native actual process maps identity differs")
        content = path.read_text()
        if "libpython" in content.lower() or "libvlaforge_libtorch_numerical_backend" not in content:
            raise ValueError("typed native maps contain Python or lack numerical provider")
    worker = prepared["numerical_worker_initialization"]["off"]
    if worker != json.loads((root / "source/numerical-workers.json").read_text())["off"]:
        raise ValueError("typed native numerical worker binding changed")
    bindings = tuple(RegionNumericalBinding.from_dict(item) for item in build["selection"]["numerical_bindings"])
    if not bindings or any(item.requirement.policy.digest() != worker["policy_sha256"] for item in bindings):
        raise ValueError("typed native numerical policy differs from compiled model")
    observed = [line for line in (folder / "stderr.log").read_text().splitlines() if line.startswith("NUMERICAL_WORKER_BOOTSTRAP_OK,")]
    if observed != ["NUMERICAL_WORKER_BOOTSTRAP_OK," + worker["policy_sha256"]]:
        raise ValueError("typed native explicit worker bootstrap marker missing")
    with (folder / "samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    validate_rows(rows, warmup=warmup, measured=measured, samples=1)
    contract = json.loads((root / "tensor-contract.json").read_text())
    raw_path = folder / contract["raw_file"]
    if file_digest(raw_path)["sha256"] != diagnostic["output_sha256"]:
        raise ValueError("typed native raw output identity changed")
    direct, eager = (root / "data/0/direct.bin").read_bytes(), (root / "data/0/eager.bin").read_bytes()
    validate_complete_bytes(raw_path.read_bytes(), direct, eager, size=contract["size_bytes"], calls=count)
    capture_path, capture = checked(build["selection"]["capture"])
    import numpy as np
    if file_digest(capture_path.parent / "actions.npz") != capture["actions"]:
        raise ValueError("captured official complete actions changed")
    with np.load(capture_path.parent / "actions.npz", allow_pickle=False) as values:
        if values["normalized_reference"].tobytes() != eager or list(values["normalized_reference"].shape) != contract["shape"]:
            raise ValueError("typed native reference is not the complete captured official action")
    if file_digest(bundle / "bundle.json") != build["bundle"]:
        raise ValueError("typed native original bundle changed")
    return {"selection": build["selection"], "bundle": build["bundle"],
            "evidence_schema": SCHEMA, "diagnostic_report": join["diagnostic_report"]}, bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("build-report", "diagnostic-report", "prepared", "protocol", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    join = {"schema": SCHEMA, **{key: record(getattr(args, key)) for key in ("build_report", "diagnostic_report", "prepared", "protocol")}}
    validate_typed_native_join(join)
    with args.output.open("x") as stream:
        stream.write(json.dumps(join, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
