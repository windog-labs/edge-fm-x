"""Build and measure frozen, no-Python Sessions at explicit tensor boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "python"))

from vlaforge.deployment import load_bundle_manifest
from vlaforge.ir.serializer import module_from_data
from vlaforge.validation.deployment_metrics import latency_report
from vlaforge.validation.session_benchmark import (
    MULTI_SCHEMA,
    benchmark_output_contract,
    compare_output_bytes,
    decode_tensor,
    encode_output_reference,
    encode_reference,
    input_specs,
    includes_host_io,
    load_reference_array,
    numeric_metrics,
    numerical_worker_bootstrap,
    process_order,
    protocol_policies,
    replay_checks,
    validate_fidelity,
    validate_host_timing,
    validate_loop_policy,
    validate_protocol,
    validate_rows,
)


def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON field: " + key)
            value[key] = item
        return value

    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify_host_timing_evidence(folder, protocol, rows, *, expected=None):
    if not includes_host_io(protocol):
        if expected is not None:
            raise ValueError("resident boundary cannot carry host timing evidence")
        return None
    path = folder / "host-timing.csv"
    if not path.is_file():
        raise ValueError("host timing evidence is missing")
    with path.open() as stream:
        timings = list(csv.DictReader(stream))
    validate_host_timing(timings, rows)
    binding = {"schema": "vlaforge.host_tensor_timing/1", "boundary": protocol["boundary"],
               "calls": len(rows), "raw_sha256": sha(path)}
    if expected is not None and binding != expected:
        raise ValueError("host timing evidence binding differs")
    return binding


def aoti_extraction_configuration(protocol, manifest):
    """Reuse the deployment loader contract and require an owned live root."""
    root = protocol.get("aoti_package_extraction_root")
    if root is None:
        return None
    from vlaforge.deployment.build import _aoti_package_extraction_configuration

    versions = (dict(manifest.backend_versions) if isinstance(manifest.backend_versions, dict)
                else {item.name: item.version for item in manifest.backend_versions})
    configuration = _aoti_package_extraction_configuration(root,
        {item.region_name: item for item in manifest.region_artifacts},
        versions, runtime_root=SOURCE)
    path = Path(root)
    info = path.lstat()
    if (path.resolve() != path or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("AOTI extraction root must be a canonical owned private directory (0700)")
    return configuration


def numerical_worker_initialization(protocol, bindings):
    """Prepare a separate checked worker initializer, never a Session setter."""
    from vlaforge.codegen.numerical import generate_libtorch_worker_initializer
    from vlaforge.deployment.numerical import PROVIDER_REQUIRED, RegionNumericalBinding

    selected = numerical_worker_bootstrap(protocol)
    bindings = tuple(bindings)
    if selected is None:
        if any(item is not None for item in bindings):
            raise ValueError("policy-bearing benchmark requires explicit numerical worker bootstrap")
        return "", "", {"mode": "none", "configured": False}
    if not bindings or any(not isinstance(item, RegionNumericalBinding) for item in bindings):
        raise ValueError("numerical worker bootstrap requires typed bindings for every Region")
    if any(item.runtime_enforcement != PROVIDER_REQUIRED for item in bindings):
        raise ValueError("numerical worker bootstrap requires actual provider enforcement")
    source = generate_libtorch_worker_initializer(
        bindings,
        acknowledge_exclusive_process=selected["acknowledge_exclusive_process"],
        acknowledge_calling_thread=selected["acknowledge_calling_thread"],
    )
    digest = bindings[0].requirement.policy.digest()
    call = f"""  {{
    const auto numerical_status = vlaforge_initialize_numerical_worker();
    if (numerical_status.code != VLAFORGE_STATUS_OK) {{
      std::fprintf(stderr, "numerical worker bootstrap failed: %u\\n", static_cast<unsigned>(numerical_status.code));
      return 21;
    }}
    std::fprintf(stderr, "NUMERICAL_WORKER_BOOTSTRAP_OK,{digest}\\n");
  }}"""
    metadata = dict(selected, configured=True, policy_sha256=digest,
                    boundary="after-owner-handshake-before-tensor-and-session",
                    binding_digests=[item.digest() for item in bindings],
                    initializer_source_sha256=hashlib.sha256(source.encode()).hexdigest())
    return source, call, metadata


def verify_numerical_worker_execution(root, folder, policy):
    metadata = read(root / "prepared.json").get("numerical_worker_initialization", {}).get(
        policy, {"mode": "none", "configured": False}
    )
    bound = root / "source/numerical-workers.json"
    if bound.exists():
        if read(bound).get(policy) != metadata:
            raise ValueError("numerical worker metadata differs from frozen source binding")
    elif metadata["configured"]:
        raise ValueError("numerical worker metadata has no frozen source binding")
    if not metadata["configured"]:
        return metadata
    observed = [line for line in (folder / "stderr.log").read_text().splitlines()
                if line.startswith("NUMERICAL_WORKER_BOOTSTRAP_OK,")]
    if observed != ["NUMERICAL_WORKER_BOOTSTRAP_OK," + metadata["policy_sha256"]]:
        raise ValueError("numerical worker initialization success marker missing or inconsistent")
    return dict(metadata, initialization_success_marker_observed=True)


def verify_allocator_evidence(folder, *, ordinal, pilot, expected=None):
    from vlaforge.validation.allocator_metrics import (
        allocator_report,
        parse_allocator_json,
        parse_allocator_snapshots,
    )

    raw = folder / "allocator-snapshots.jsonl"
    path = folder / "allocator-report.json"
    actual = {"raw_snapshots_sha256": sha(raw), "report_sha256": sha(path)}
    if expected is not None and expected != actual:
        raise ValueError("allocator evidence differs from process report binding")
    report = parse_allocator_json(path.read_text())
    recomputed = allocator_report(
        parse_allocator_snapshots(raw.read_text()), ordinal=ordinal
    )
    if (
        any(report.get(key) != value for key, value in recomputed.items())
        or report.get("raw_snapshots_sha256") != actual["raw_snapshots_sha256"]
        or report.get("execution_sha256") != sha(folder / "execution.json")
        or report.get("pilot") is not pilot
    ):
        raise ValueError("allocator report disagrees with raw observations or execution")
    return actual


def additional_output_declarations(contract):
    return ",\n".join(
        "{" + f'{item["index"]}u, VLAFORGE_DTYPE_{item["dtype"].upper()}, '
        + "{" + ",".join(map(str, item["shape"])) + "}, "
        + f'{item["size_bytes"]}u, {item["count"]}u, '
        + str(item["role"] != "exact").lower() + ", "
        + ", ".join(json.dumps(item[key]) for key in ("raw_file", "direct_file", "eager_file")) + "}"
        for item in contract.get("outputs", []) if item["role"] != "primary-action"
    )


def runtime_library_records(build, bundle, manifest):
    """Bind generated runtime DSOs as well as the bundle's delivered copies."""
    paths = {path.resolve() for path in build.rglob("libvlaforge_*.so*") if path.is_file()}
    paths.update((bundle / item.path).resolve() for item in manifest.binaries if item.role == "runtime_shared_library")
    return [{"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size} for path in sorted(paths)]


def verify_runtime_library_execution(root, folder, policy):
    """Match actual Linux mappings to the exact runtime files frozen before run."""
    inventory = root / "source/runtime-libraries.json"
    if not inventory.exists():
        return {"schema": "vlaforge.benchmark_runtime_libraries/1", "status": "not-recorded-legacy", "verified": False}
    expected = {item["path"]: item for item in read(inventory)[policy]}
    mapped = set()
    for line in (folder / "process-maps.txt").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith("libvlaforge_"):
            mapped.add(fields[5])
    if not mapped <= expected.keys():
        raise ValueError("actual runtime library was not frozen: " + str(sorted(mapped - expected.keys())))
    records = []
    for name in sorted(mapped):
        path = Path(name)
        item = expected[name]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha(path) != item["sha256"]:
            raise ValueError("actual runtime library bytes changed: " + name)
        records.append(item)
    worker = read(root / "prepared.json").get("numerical_worker_initialization", {}).get(policy, {})
    if worker.get("configured") and not any(Path(name).name == "libvlaforge_libtorch_numerical_backend.so" for name in mapped):
        raise ValueError("actual numerical runtime library mapping missing")
    return {"schema": "vlaforge.benchmark_runtime_libraries/1", "status": "verified", "verified": True,
            "inventory_sha256": sha(inventory), "maps_sha256": sha(folder / "process-maps.txt"), "actual_libraries": records}


def verify_multi_output_evidence(root, folder, protocol, contract, rows, *, expected=None, persist=False):
    """Re-read every byte, including exact integer outputs, before publication."""
    outputs, raw_hashes = [], {}
    for item in contract["outputs"]:
        path = folder / item["raw_file"]
        raw = path.read_bytes()
        if len(raw) != len(rows) * item["size_bytes"]:
            raise ValueError("multi-output raw archive is incomplete")
        raw_hashes[item["name"]] = sha(path)
        checks, references = [], {}
        for index, row in enumerate(rows):
            start = index * item["size_bytes"]
            sample = index % len(protocol["samples"])
            if sample not in references:
                reference = root / "data" / str(sample)
                references[sample] = (
                    (reference / item["direct_file"]).read_bytes(),
                    (reference / item["eager_file"]).read_bytes(),
                )
            direct, eager = references[sample]
            check = compare_output_bytes(raw[start:start + item["size_bytes"]],
                                         direct, eager, item)
            if protocol.get("eager_validation") == "bitwise" and not check["eager_bitwise_equal"]:
                raise ValueError("multi-output eager bitwise gate failed")
            if item["role"] == "primary-action" and protocol["quality_gate"] == "passed":
                metrics = check["primary_metrics"]
                if (
                    not check["eager_bitwise_equal"]
                    and (
                        metrics["mse"] > protocol["paper_gates"]["mse_max"]
                        or metrics["cosine"] is None
                        or metrics["cosine"] < protocol["paper_gates"]["cosine_min"]
                    )
                ):
                    raise ValueError("multi-output primary action fails paper numeric gate")
            checks.append(dict(check, run=index, sample=sample, revision=int(row["revision"])))
        outputs.append({"contract": item, "raw_sha256": raw_hashes[item["name"]], "calls": checks})
    value = {"schema": "vlaforge.multi_output_fidelity/1", "status": "validated_complete_outputs",
             "primary_output": contract["primary_output"], "outputs": outputs,
             "execution_sha256": sha(folder / "execution.json"),
             "samples_sha256": sha(folder / "samples.csv"),
             "protocol_sha256": sha(root / "protocol.json"),
             "tensor_contract_sha256": sha(root / "tensor-contract.json")}
    path = folder / "multi-output-fidelity.json"
    if persist:
        write(path, value)
    elif read(path) != value:
        raise ValueError("multi-output fidelity disagrees with complete raw evidence")
    actual = {"raw_sha256": raw_hashes, "fidelity_sha256": sha(path)}
    if expected is not None and expected != actual:
        raise ValueError("multi-output evidence differs from process report binding")
    return actual


def command(args, output, *, env=None):
    start = time.time_ns()
    completed = subprocess.run(
        args, capture_output=True, text=True, env=env, check=False
    )
    write(
        output,
        {
            "command": args,
            "start_ns": start,
            "elapsed_ns": time.time_ns() - start,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    )
    completed.check_returncode()
    return completed.stdout


def prepare(args):
    import numpy as np

    protocol = read(args.protocol)
    validate_protocol(protocol)
    workers = {}
    extraction = {}
    for policy in protocol_policies(protocol):
        bundle = Path(protocol["bundles"][policy]).resolve()
        manifest = load_bundle_manifest(bundle / "bundle.json")
        manifest.verify_files(bundle)
        extraction[policy] = aoti_extraction_configuration(protocol, manifest)
        existing_configuration = bundle / "metadata/build_configuration.json"
        if extraction[policy] is not None and existing_configuration.is_file():
            prior = read(existing_configuration).get("aoti_package_extraction")
            if prior is not None and prior["root"] != extraction[policy]["root"]:
                raise ValueError("benchmark extraction root conflicts with the bundle's compiled configuration")
        workers[policy] = numerical_worker_initialization(
            protocol, (item.numerical_binding for item in manifest.region_artifacts)
        )
    root = args.output
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            "benchmark output must be empty; never overwrite measured evidence"
        )
    root.mkdir(parents=True, exist_ok=True)
    write(root / "protocol.json", protocol)
    frozen = root / "source/runtime"
    frozen.mkdir(parents=True)
    shutil.copy2(SOURCE / "CMakeLists.txt", frozen / "CMakeLists.txt")
    for name in ("include", "runtime", "backends", "cmake"):
        shutil.copytree(SOURCE / name, frozen / name)
    for path in (
        Path(__file__),
        Path(__file__).with_name("session_benchmark_runner.cpp.in"),
        Path(__file__).with_name("session_allocator_observer.h"),
        SOURCE / "python/vlaforge/validation/session_benchmark.py",
        SOURCE / "python/vlaforge/validation/allocator_metrics.py",
    ):
        shutil.copy2(path, root / "source" / path.name)
    modules = []
    runtime_libraries = {}
    binding = {str(Path(path).resolve()): sha(path) for path in protocol["evidence"]}
    if any(item is not None for item in extraction.values()):
        configuration_path = root / "source/aoti-package-extraction.json"
        write(configuration_path, extraction)
        helper = SOURCE / "python/vlaforge/deployment/build.py"
        shutil.copy2(helper, root / "source/deployment_build.py")
        binding[str(helper)] = sha(helper)
    for path in (
        Path(__file__),
        Path(__file__).with_name("session_benchmark_runner.cpp.in"),
        Path(__file__).with_name("session_allocator_observer.h"),
        SOURCE / "python/vlaforge/validation/session_benchmark.py",
        SOURCE / "python/vlaforge/validation/allocator_metrics.py",
        SOURCE / "python/vlaforge/validation/deployment_metrics.py",
    ):
        binding[str(path)] = sha(path)
    if numerical_worker_bootstrap(protocol) is not None:
        for relative in (
            "codegen/numerical.py", "deployment/numerical.py",
            "deployment/libtorch_numerical.py", "numerical_context.py",
        ):
            path = SOURCE / "python/vlaforge" / relative
            destination = root / "source/python/vlaforge" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            binding[str(path)] = sha(path)
    for policy in protocol_policies(protocol):
        bundle = Path(protocol["bundles"][policy]).resolve()
        manifest = load_bundle_manifest(bundle / "bundle.json")
        manifest.verify_files(bundle)
        binding[str(bundle / "bundle.json")] = sha(bundle / "bundle.json")
        for artifact in read(bundle / "bundle.json")["region_artifacts"]:
            path = bundle / artifact["artifact_path"]
            binding[str(path)] = sha(path)
        mode = protocol.get("bundle_metadata_mode", "selection-manifest")
        ir_path = bundle / "metadata" / ("semantic_ir.json" if mode == "ordinary-source" else "input_semantic_ir.json")
        module = module_from_data(read(ir_path))
        modules.append(module)
        selected_path = bundle / "metadata/loop_execution.json"
        selected = read(selected_path) if selected_path.is_file() else None
        plan = read(bundle / "metadata/scheduled_plan.json")
        validate_loop_policy(selected, plan, policy, mode)
        source = root / "generated" / policy
        shutil.copytree(bundle / "generated", source)
        output = benchmark_output_contract(module, protocol)
        declarations, ordinal, count = input_specs(module, contract=output)
        if ordinal != protocol["gpu_ordinal"]:
            raise ValueError("protocol CUDA ordinal differs from compiled ports")
        rows = [
            {
                "task_id": task["id"],
                "policy": policy,
                "steps": len(
                    range(
                        task["attributes"]["lower"],
                        task["attributes"]["upper"],
                        task["attributes"]["step"],
                    )
                ),
            }
            for task in plan["tasks"]
            if task["opcode"] == "vla.for" and policy != "off"
        ]
        checks, failures = replay_checks(rows)
        template = (root / "source/session_benchmark_runner.cpp.in").read_text()
        runner = (
            template.replace("@INPUTS@", declarations)
            .replace("@ORDINAL@", str(ordinal))
            .replace("@SAMPLES@", str(len(protocol["samples"])))
            .replace("@OUTPUT_COUNT@", str(count))
            .replace("@OUTPUT_BYTES@", str(output["size_bytes"]))
            .replace("@OUTPUT_DTYPE@", "VLAFORGE_DTYPE_" + output["dtype"].upper())
            .replace("@OUTPUT_SHAPE@", ",".join(map(str, output["shape"])))
            .replace("@ACTIVE_INDICES@", ",".join(map(str, output["active_indices"])))
            .replace("@OUTPUT_FILE@", output["raw_file"])
            .replace("@PRIMARY_OUTPUT_INDEX@", str(output.get("index", 0)))
            .replace("@MULTI_OUTPUT_DEFINE@", "#define VLAFORGE_BENCHMARK_MULTI_OUTPUT 1" if "outputs" in output else "")
            .replace("@ADDITIONAL_OUTPUT_SPECS@", additional_output_declarations(output))
            .replace("@OWNER_HANDSHAKE@", str(protocol.get("owner_identity_mode") == "cuda-registration-handshake").lower())
            .replace("@HOST_IO@", str(includes_host_io(protocol)).lower())
            .replace("@REPLAY_AUDIT@", checks)
            .replace("@REPLAY_FAILURE@", failures)
            .replace("@SESSION_LIFECYCLE_BEGIN@", "")
            .replace("@SESSION_LIFECYCLE_END@", "")
        )
        observe_allocator = protocol.get("allocator_observation", "off") == "libtorch-native/1"
        if observe_allocator:
            if any(item["capability"]["backend"] not in ("aoti", "torchscript")
                   for item in read(bundle / "bundle.json")["region_artifacts"]):
                raise ValueError("LibTorch allocator observations require only LibTorch Regions")
            shutil.copy2(root / "source/session_allocator_observer.h", source / "session_allocator_observer.h")
            cmake = source / "CMakeLists.txt"
            cmake.write_text(cmake.read_text() + "\ntarget_link_libraries(vlaforge_generated_runner PRIVATE torch_cuda c10_cuda)\n")
        runner = runner.replace("@ALLOCATOR_INCLUDE@", '#include "session_allocator_observer.h"' if observe_allocator else "")
        for key, phase in (
            ("BEFORE_SESSION", "before_session"), ("AFTER_LOAD", "after_load"),
            ("AFTER_WARMUP", "after_warmup"), ("AFTER_MEASURED", "after_measured"),
            ("AFTER_DESTROY", "after_destroy"),
        ):
            check = f'  if (!vlaforge_benchmark::ObserveAllocator(output_root, "{phase}", kOrdinal)) return 19;'
            if key == "AFTER_WARMUP":
                check = f'    if (run == warmup && !vlaforge_benchmark::ObserveAllocator(output_root, "{phase}", kOrdinal)) return 19;'
            runner = runner.replace(f"@ALLOCATOR_{key}@", check if observe_allocator else "")
        initializer, call, worker = workers[policy]
        marker = "// @NUMERICAL_WORKER_BOOTSTRAP@"
        if worker["configured"] and runner.count(marker) != 1:
            raise ValueError("benchmark runner lacks one explicit post-handshake bootstrap location")
        runner = initializer + runner.replace(marker, call)
        (source / "runner.cpp").write_text(runner)
    if any(module != modules[0] for module in modules):
        raise ValueError("execution policies do not share the identical original IR")
    module = modules[0]
    write(root / "tensor-contract.json", output)
    for index, sample in enumerate(protocol["samples"]):
        folder = root / "data" / str(index)
        folder.mkdir(parents=True)
        if set(sample["inputs"]) != {port.name for port in module.inputs}:
            raise ValueError("benchmark input names differ from Session schema")
        for input_id, port in enumerate(module.inputs):
            path = Path(sample["inputs"][port.name]).resolve()
            decode_tensor(path.read_bytes(), port.payload.dtype, port.payload.shape)
            binding[str(path)] = sha(path)
            shutil.copy2(path, folder / f"{input_id}.bin")
        for item in output.get("outputs", [output]):
            paired = sample["outputs"][item["name"]] if "outputs" in output else sample
            for name in ("direct", "eager"):
                path = Path(paired[name]).resolve()
                binding[str(path)] = sha(path)
                value = load_reference_array(
                    path,
                    item["name"] if "outputs" in output else "output",
                    item["dtype"],
                    item["shape"],
                )
                raw = (encode_output_reference(value, item) if "outputs" in output
                       else encode_reference(value, item["dtype"], item["shape"]))
                (folder / item.get(name + "_file", f"{name}.bin")).write_bytes(raw)
    env = dict(os.environ, TORCH_CUDA_ARCH_LIST=protocol["cuda_arch"])
    for policy in protocol_policies(protocol):
        build = root / "build" / policy
        command(
            [
                "cmake",
                "-S",
                str(root / "generated" / policy),
                "-B",
                str(build),
                "-DVLAFORGE_RUNTIME_ROOT=" + str(frozen),
                "-DCMAKE_PREFIX_PATH=" + protocol["cmake_prefix_path"],
                "-DBUILD_TESTING=OFF",
                "-DCMAKE_BUILD_TYPE=Release",
                *(["-DVLAFORGE_AOTI_PACKAGE_EXTRACTION_ROOT=" + extraction[policy]["root"]]
                  if extraction[policy] is not None else []),
            ],
            root / "logs" / f"{policy}-configure.json",
            env=env,
        )
        command(
            ["cmake", "--build", str(build), "--parallel", "2"],
            root / "logs" / f"{policy}-build.json",
            env=env,
        )
        binary = build / "vlaforge_generated_runner"
        linked = command(["ldd", str(binary)], root / "logs" / f"{policy}-ldd.json")
        if any(name in linked.lower() for name in ("libpython", "libtorch_python", "not found")):
            raise ValueError("benchmark executable has Python or unresolved dependency")
        binding[str(binary)] = sha(binary)
        bundle = Path(protocol["bundles"][policy]).resolve()
        manifest = load_bundle_manifest(bundle / "bundle.json")
        runtime_libraries[policy] = runtime_library_records(build, bundle, manifest)
        for library in runtime_libraries[policy]:
            binding[library["path"]] = library["sha256"]
    library_metadata = root / "source/runtime-libraries.json"
    write(library_metadata, runtime_libraries)
    binding[str(library_metadata)] = sha(library_metadata)
    for directory in ("data", "generated", "source"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file():
                binding[str(path)] = sha(path)
    binding[str(root / "protocol.json")] = sha(root / "protocol.json")
    binding[str(root / "tensor-contract.json")] = sha(root / "tensor-contract.json")
    worker_metadata = {policy: worker[2] for policy, worker in workers.items()}
    worker_metadata_path = root / "source/numerical-workers.json"
    write(worker_metadata_path, worker_metadata)
    binding[str(worker_metadata_path)] = sha(worker_metadata_path)
    write(root / "frozen-files.json", binding)
    write(
        root / "prepared.json",
        {
            "status": "prepared_not_executed",
            "frozen_files_sha256": sha(root / "frozen-files.json"),
            "protocol_sha256": sha(root / "protocol.json"),
            "process_order": process_order(protocol["processes"], protocol_policies(protocol)),
            "numerical_worker_initialization": worker_metadata,
            **({"aoti_package_extraction": extraction} if any(item is not None for item in extraction.values()) else {}),
        },
    )
    print("PREPARED", root, flush=True)


def verify_frozen(root):
    expected = read(root / "prepared.json")["frozen_files_sha256"]
    if sha(root / "frozen-files.json") != expected:
        raise ValueError("frozen binding manifest changed")
    for path, expected in read(root / "frozen-files.json").items():
        if sha(path) != expected:
            raise ValueError("frozen benchmark file changed: " + path)


def owners(ordinal):
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(ordinal),
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    result.check_returncode()
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    return [
        {"pid": int(row[0]), "process_name": row[1], "memory_mib": row[2]}
        for row in rows
        if row and row[0].strip().isdigit()
    ]


def telemetry(ordinal):
    fields = "timestamp,index,uuid,name,driver_version,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,clocks.current.graphics,clocks.current.sm,clocks.current.memory"
    result = subprocess.run(
        ["nvidia-smi", "-i", str(ordinal), "--query-gpu=" + fields, "--format=csv"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "wall_ns": time.time_ns(),
        "command_fields": fields,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "compute_owners": owners(ordinal),
    }


def register_gpu_owner(process, folder, protocol, *, timeout=30.0):
    """Causally map one child across namespaces, never admit an arbitrary PID."""
    ordinal = protocol.get("monitor_gpu", protocol["gpu_ordinal"])
    report = {"mode": "cuda-registration-handshake", "container_pid": process.pid,
              "identity_is_nspid_verified": False, "observations": [], "status": "incomplete"}
    candidate = None
    try:
        for stage in ("registered-1", "reset", "registered-2"):
            deadline = time.monotonic() + timeout
            while True:
                if process.poll() is not None:
                    raise ValueError("owner handshake child exited before registration completed")
                current = owners(ordinal)
                marker = folder / ("owner-" + stage + ".json")
                ready = read(marker) if marker.is_file() else None
                report["observations"].append({"stage": stage, "wall_ns": time.time_ns(),
                                               "owners": current, "ready": ready})
                if len(current) > 1 or (candidate is not None and any(item["pid"] != candidate for item in current)):
                    raise ValueError("owner handshake observed foreign or changed GPU PID")
                if ready is not None:
                    if ready != {"pid": process.pid, "ordinal": protocol["gpu_ordinal"]}:
                        raise ValueError("owner handshake marker is not from the expected child/device")
                    if (stage == "reset" and not current) or (stage != "reset" and len(current) == 1):
                        if candidate is None:
                            candidate = current[0]["pid"]
                        signal = folder / ("continue-" + stage)
                        temporary = signal.with_suffix(".tmp")
                        temporary.write_text("continue\n")
                        temporary.replace(signal)
                        break
                if time.monotonic() >= deadline:
                    raise ValueError("owner handshake timed out waiting for registration/reset visibility")
                time.sleep(0.1)
        report.update(status="passed", nvml_pid=candidate,
                      identity_evidence="idle preflight plus child context register/reset/re-register on one GPU")
        return candidate
    except (OSError, ValueError) as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write(folder / "owner-handshake.json", report)


def one_process(root, protocol, repeat, policy, *, pilot):
    verify_frozen(root)
    ordinal = protocol.get("monitor_gpu", protocol["gpu_ordinal"])
    current = owners(ordinal)
    if current:
        raise RuntimeError(
            "GPU compute owner already active; no task was interrupted: " + str(current)
        )
    folder = root / ("pilot" if pilot else "runs") / f"{repeat:02d}-{policy}"
    folder.mkdir(parents=True, exist_ok=False)
    warmup, measured = (16, 32) if pilot else (protocol["warmup"], protocol["measured"])
    cmd = [
        str(root / "build" / policy / "vlaforge_generated_runner"),
        protocol["bundles"][policy],
        str(root / "data"),
        str(folder),
        str(warmup),
        str(measured),
    ]
    env = dict(os.environ, PYTHONHOME="/no/python/home", PYTHONPATH="/no/python/path")
    if "cuda_visible_devices" in protocol:
        env["CUDA_VISIBLE_DEVICES"] = protocol["cuda_visible_devices"]
    started = time.time_ns()
    collision = None
    monitor_error = None
    owner_pid = None
    write(
        folder / "execution.json",
        {
            "status": "starting",
            "command": cmd,
            "start_ns": started,
            "protocol_sha256": sha(root / "protocol.json"),
            "pilot": pilot,
        },
    )
    with (
        (folder / "samples.csv").open("w") as stdout,
        (folder / "stderr.log").open("w") as stderr,
        (folder / "telemetry.jsonl").open("w") as samples,
    ):
        process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
        try:
            owner_pid = register_gpu_owner(process, folder, protocol) if protocol.get("owner_identity_mode") == "cuda-registration-handshake" else process.pid
            while process.poll() is None:
                observation = telemetry(ordinal)
                samples.write(json.dumps(observation) + "\n")
                samples.flush()
                foreign = [
                    item
                    for item in observation["compute_owners"]
                    if item["pid"] != owner_pid
                ]
                if foreign:
                    collision = foreign
                    process.terminate()
                    break
                time.sleep(1)
            exit_code = process.wait(timeout=15)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            monitor_error = repr(error)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            exit_code = process.returncode
    elapsed = time.time_ns() - started
    execution = {
        "status": "executed"
        if exit_code == 0 and not collision and not monitor_error
        else "incomplete",
        "command": cmd,
        "pid": process.pid,
        "monitored_nvml_pid": owner_pid,
        "owner_identity_mode": protocol.get("owner_identity_mode", "process-pid"),
        "start_ns": started,
        "elapsed_ns": elapsed,
        "exit_code": exit_code,
        "foreign_compute_owners": collision,
        "pilot": pilot,
        "monitor_error": monitor_error,
        "protocol_sha256": sha(root / "protocol.json"),
    }
    write(folder / "execution.json", execution)
    if exit_code or collision or monitor_error:
        raise RuntimeError(
            "benchmark process failed or another compute owner appeared; evidence retained at "
            + str(folder)
        )
    maps = folder / "process-maps.txt"
    if not maps.is_file() or any(name in maps.read_text().lower() for name in ("libpython", "libtorch_python")):
        raise ValueError("benchmark process maps missing or contain Python libraries")
    numerical_worker = verify_numerical_worker_execution(root, folder, policy)
    runtime_library_evidence = verify_runtime_library_execution(root, folder, policy)
    if protocol.get("allocator_observation", "off") == "libtorch-native/1":
        from vlaforge.validation.allocator_metrics import (
            allocator_report,
            parse_allocator_snapshots,
        )

        path = folder / "allocator-snapshots.jsonl"
        observations = parse_allocator_snapshots(path.read_text())
        observed = allocator_report(observations, ordinal=protocol["gpu_ordinal"])
        observed["raw_snapshots_sha256"] = sha(path)
        observed["execution_sha256"] = sha(folder / "execution.json")
        observed["pilot"] = pilot
        write(folder / "allocator-report.json", observed)
    with (folder / "samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    validate_rows(
        rows, warmup=warmup, measured=measured, samples=len(protocol["samples"])
    )
    host_timing_evidence = verify_host_timing_evidence(folder, protocol, rows)
    contract = read(root / "tensor-contract.json")
    raw_path = folder / contract["raw_file"]
    raw = raw_path.read_bytes()
    chunk_bytes = contract["size_bytes"]
    if len(raw) != len(rows) * chunk_bytes:
        raise ValueError("raw output archive is incomplete")
    fidelity = []
    for index in range(len(rows)):
        actual_bytes = raw[index * chunk_bytes:(index + 1) * chunk_bytes]
        actual = decode_tensor(actual_bytes, contract["dtype"], contract["shape"]).reshape(-1)
        expected = (
            root / "data" / str(index % len(protocol["samples"])) / "direct.bin"
        ).read_bytes()
        if actual_bytes != expected:
            raise ValueError("independent raw output comparison failed")
        eager_bytes = (root / "data" / str(index % len(protocol["samples"])) / "eager.bin").read_bytes()
        eager = decode_tensor(eager_bytes, contract["dtype"], contract["shape"]).reshape(-1)
        complete = numeric_metrics(eager, actual)
        active = contract["active_indices"]
        primary = numeric_metrics(eager[active], actual[active])
        for prefix, computed in (("eager", complete), ("active_eager", primary)):
            if int(rows[index][prefix + "_cosine_defined"]) != int(computed["cosine"] is not None):
                raise ValueError("cosine definition status differs from C++ metric log")
            for key in ("mse", "max_abs", "cosine"):
                value = computed[key]
                if value is not None and not math.isclose(
                    float(rows[index][prefix + "_" + key]), value, rel_tol=1e-10, abs_tol=1e-12
                ):
                    raise ValueError("independent output metrics differ from C++ metric log")
        mse, maximum, cosine = primary["mse"], primary["max_abs"], primary["cosine"]
        fidelity.append(
            {
                "run": index,
                "sample": index % len(protocol["samples"]),
                "same_artifact_bitwise_equal": True,
                "output_dtype": contract["dtype"],
                "raw_size_bytes": chunk_bytes,
                "primary_metric_space": contract["primary_metric_space"],
                "complete_storage_metrics": complete,
                "primary_metrics": primary,
                "same_artifact_mse": 0.0,
                "same_artifact_max_abs": 0.0,
                "same_artifact_cosine": float(rows[index]["direct_cosine"])
                if rows[index]["direct_cosine_defined"] == "1"
                else None,
                "eager_mse": mse,
                "eager_bitwise_equal": actual_bytes == eager_bytes,
                "eager_max_abs": maximum,
                "eager_cosine": cosine,
                "paper_numeric_gate": (
                    actual_bytes == eager_bytes
                    or (
                        mse <= protocol["paper_gates"]["mse_max"]
                        and cosine is not None
                        and cosine >= protocol["paper_gates"]["cosine_min"]
                    )
                ),
            }
        )
    write(folder / "fidelity.json", fidelity)
    try:
        validate_fidelity(protocol, fidelity)
    except ValueError as error:
        write(folder / "report.json", {
            "status": "numerical_failed", "quality_gate": "failed", "error": str(error),
            "pilot": pilot, "policy": policy, "repeat": repeat,
        })
        raise
    steady = [
        {"index": index, "latency_ns": int(row["latency_ns"]), "repeat_id": repeat}
        for index, row in enumerate(rows[warmup:])
    ]
    wall_line = next(
        line
        for line in (folder / "stderr.log").read_text().splitlines()
        if line.startswith("WALL,")
    )
    _, all_wall, steady_wall = wall_line.split(",")
    report = {
        "status": "passed",
        "policy": policy,
        "repeat": repeat,
        "pilot": pilot,
        "warmup": warmup,
        "measured": measured,
        "validated_outputs": len(rows),
        "all_same_artifact_bitwise_equal": True,
        "all_eager_bitwise_equal": all(row["eager_bitwise_equal"] for row in fidelity),
        "tensor_contract": contract,
        "all_paper_numeric_gates_passed": all(row["paper_numeric_gate"] for row in fidelity),
        "eager_validation": protocol.get("eager_validation", "paper-gates"),
        "no_python_deployment": True,
        "process_maps_sha256": sha(maps),
        "latency": latency_report(steady),
        "quality_gate": protocol["quality_gate"],
        "eligible_for_lossless_paper_table": False,
        "boundary": protocol["boundary"],
        "all_calls_wall_ns": int(all_wall),
        "measured_calls_wall_ns": int(steady_wall),
        "measured_wall_calls_per_second": measured * 1e9 / int(steady_wall),
        "supervisor_wall_ns": elapsed,
        "supervisor_wall_calls_per_second": len(rows) * 1e9 / elapsed,
        "supervisor_wall_includes_initialization_and_up_to_one_second_exit_detection": True,
        "numerical_worker_initialization": numerical_worker,
        "runtime_library_evidence": runtime_library_evidence,
        "outputs_sha256": sha(raw_path),
        "samples_sha256": sha(folder / "samples.csv"),
    }
    if host_timing_evidence is not None:
        report["host_timing_evidence"] = host_timing_evidence
    if protocol.get("allocator_observation", "off") == "libtorch-native/1":
        report["allocator_observation"] = verify_allocator_evidence(
            folder, ordinal=protocol["gpu_ordinal"], pilot=pilot
        )
    if protocol["schema"] == MULTI_SCHEMA:
        report["multi_output_evidence"] = verify_multi_output_evidence(
            root, folder, protocol, contract, rows, persist=True
        )
        report["validated_output_tensors"] = len(rows) * len(contract["outputs"])
        report["all_eager_bitwise_equal"] = all(
            call["eager_bitwise_equal"] for item in read(folder / "multi-output-fidelity.json")["outputs"]
            for call in item["calls"]
        )
    write(folder / "report.json", report)
    print(
        "PROCESS_COMPLETE", repeat, policy, "pilot" if pilot else "formal", flush=True
    )


def _pid_alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _completed_process(folder, repeat, policy, *, pilot):
    execution_path, report_path = folder / "execution.json", folder / "report.json"
    if not execution_path.is_file() or not report_path.is_file():
        return False
    execution, report = read(execution_path), read(report_path)
    return (execution.get("status") == "executed" and execution.get("exit_code") == 0
            and execution.get("pilot") is pilot and report.get("status") == "passed"
            and report.get("pilot") is pilot and report.get("repeat") == repeat
            and report.get("policy") == policy)


def _archive_incomplete_process(root, folder, repeat, policy):
    execution_path = folder / "execution.json"
    if execution_path.is_file() and _pid_alive(read(execution_path).get("pid")):
        raise RuntimeError("cannot archive a live benchmark worker")
    archive = root / "aborted" / f"{repeat:02d}-{policy}-{time.time_ns()}"
    archive.parent.mkdir(parents=True, exist_ok=False)
    folder.rename(archive)
    return archive


def measure(args, *, pilot):
    protocol = read(args.output / "protocol.json")
    validate_protocol(protocol)
    for repeat, policy in process_order(1 if pilot else protocol["processes"], protocol_policies(protocol)):
        folder = args.output / ("pilot" if pilot else "runs") / f"{repeat:02d}-{policy}"
        if args.resume and folder.exists():
            if _completed_process(folder, repeat, policy, pilot=pilot):
                print("PROCESS_REUSED", repeat, policy, flush=True)
                continue
            print("PROCESS_ABORTED_ARCHIVED", _archive_incomplete_process(args.output, folder, repeat, policy), flush=True)
        one_process(args.output, protocol, repeat, policy, pilot=pilot)


def verify_multi_output_latency(report, rows, *, warmup, measured, repeat):
    steady = [{"index": index, "latency_ns": int(row["latency_ns"]), "repeat_id": repeat}
              for index, row in enumerate(rows[warmup:])]
    if (report.get("warmup") != warmup or report.get("measured") != measured
            or report.get("validated_outputs") != len(rows)
            or len(steady) != measured or report.get("latency") != latency_report(steady)):
        raise ValueError("multi-output process latency report does not match raw samples")


def aggregate(args):
    import statistics

    root = args.output
    verify_frozen(root)
    protocol = read(root / "protocol.json")
    validate_protocol(protocol)
    results, table = {}, []
    for policy in protocol_policies(protocol):
        processes, samples = [], []
        for repeat in range(protocol["processes"]):
            folder = root / "runs" / f"{repeat:02d}-{policy}"
            process_report = read(folder / "report.json")
            if process_report["status"] != "passed" or process_report["pilot"]:
                raise ValueError("formal process incomplete")
            if process_report.get("boundary") != protocol["boundary"]:
                raise ValueError("formal process timing boundary differs")
            if includes_host_io(protocol):
                if not isinstance(process_report.get("host_timing_evidence"), dict):
                    raise ValueError("formal process is missing host timing evidence binding")
                with (folder / "samples.csv").open() as stream:
                    host_rows = list(csv.DictReader(stream))
                validate_rows(host_rows, warmup=protocol["warmup"], measured=protocol["measured"],
                              samples=len(protocol["samples"]))
                verify_host_timing_evidence(folder, protocol, host_rows,
                                           expected=process_report["host_timing_evidence"])
                verify_multi_output_latency(process_report, host_rows, warmup=protocol["warmup"],
                                            measured=protocol["measured"], repeat=repeat)
            library_evidence = verify_runtime_library_execution(root, folder, policy)
            if library_evidence["verified"] and process_report.get("runtime_library_evidence") != library_evidence:
                raise ValueError("formal process runtime library binding differs")
            if numerical_worker_bootstrap(protocol) is not None:
                worker = verify_numerical_worker_execution(root, folder, policy)
                if not worker["configured"] or process_report.get("numerical_worker_initialization") != worker:
                    raise ValueError("formal process is missing numerical worker initialization evidence")
            if protocol["schema"] == MULTI_SCHEMA:
                binding = process_report.get("multi_output_evidence")
                if not isinstance(binding, dict) or set(binding) != {"raw_sha256", "fidelity_sha256"}:
                    raise ValueError("formal process is missing complete output evidence bindings")
                with (folder / "samples.csv").open() as stream:
                    observed_rows = list(csv.DictReader(stream))
                validate_rows(observed_rows, warmup=protocol["warmup"], measured=protocol["measured"],
                              samples=len(protocol["samples"]))
                verify_multi_output_latency(process_report, observed_rows, warmup=protocol["warmup"],
                                            measured=protocol["measured"], repeat=repeat)
                verify_multi_output_evidence(root, folder, protocol, read(root / "tensor-contract.json"),
                                             observed_rows, expected=binding)
            if protocol.get("allocator_observation", "off") == "libtorch-native/1":
                binding = process_report.get("allocator_observation")
                if type(binding) is not dict or set(binding) != {
                    "raw_snapshots_sha256", "report_sha256"
                }:
                    raise ValueError("formal process is missing allocator evidence bindings")
                verify_allocator_evidence(
                    folder, ordinal=protocol["gpu_ordinal"], pilot=False,
                    expected=process_report["allocator_observation"],
                )
            processes.append(process_report)
            offset = len(samples)
            samples.extend(
                {
                    "index": offset + index,
                    "latency_ns": row["latency_ns"],
                    "repeat_id": repeat,
                }
                for index, row in enumerate(process_report["latency"]["raw_samples"])
            )
        combined = latency_report(samples)
        results[policy] = {
            "processes": processes,
            "combined": combined,
            "quality_gate": protocol["quality_gate"],
            "eligible_for_lossless_paper_table": False,
        }
        means = [item["latency"]["summary"]["mean_ns"] for item in processes]
        summary = combined["summary"]
        row = {
            "policy": policy,
            "boundary": protocol["boundary"],
            "quality_gate": protocol["quality_gate"],
            "processes": len(processes),
            "measured_calls": len(samples),
            "mean_ms": summary["mean_ns"] / 1e6,
            "p50_ms": summary["p50_ns"] / 1e6,
            "p95_ms": summary["p95_ns"] / 1e6,
            "p99_ms": summary["p99_ns"] / 1e6,
            "std_ms": summary["std_ns"] / 1e6,
            "model_tensor_calls_per_second": summary["sequential_calls_per_second"],
            "independent_process_mean_std_ms": statistics.stdev(means) / 1e6,
            "mean_full_steady_wall_calls_per_second": statistics.fmean(
                item["measured_wall_calls_per_second"] for item in processes
            ),
            "eligible_for_lossless_paper_table": False,
        }
        table.append(row)
        with (root / f"{policy}-cdf.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(("latency_ns", "cdf"))
            for index, value in enumerate(
                sorted(item["latency_ns"] for item in samples)
            ):
                writer.writerow((value, (index + 1) / len(samples)))
    write(
        root / "report.json",
        {
            "schema": "vlaforge.session_latency_report/1",
            "status": "completed",
            "protocol_sha256": sha(root / "protocol.json"),
            "boundary": protocol["boundary"],
            "quality_gate": protocol["quality_gate"],
            "full_paper_acceptance": False,
            "process_order": process_order(protocol["processes"], protocol_policies(protocol)),
            "policies": results,
        },
    )
    with (root / "latency-table.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print("REPORT_COMPLETE", root / "report.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "pilot", "run", "report"), required=True
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse passing workers and archive incomplete workers before retry")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.stage == "prepare":
        if args.protocol is None:
            parser.error("prepare requires --protocol")
        prepare(args)
    elif args.stage == "report":
        aggregate(args)
    else:
        measure(args, pilot=args.stage == "pilot")


if __name__ == "__main__":
    main()
