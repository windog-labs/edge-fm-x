"""Observe repeated native Session lifetimes using an existing frozen benchmark."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "python"))

from vlaforge.ir.serializer import module_from_data
from vlaforge.validation.allocator_metrics import (
    PHASES,
    parse_allocator_json,
    parse_allocator_snapshots,
)
from vlaforge.validation.session_lifecycle import (
    allocator_lifecycle_report,
    lifecycle_scope,
    split_lifecycle_rows,
)


def _benchmark():
    path = Path(__file__).with_name("benchmark_session.py")
    spec = importlib.util.spec_from_file_location("vlaforge_lifecycle_benchmark_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_runner(template, *, replacements, cycles, numerical_worker=None):
    """Use explicit lifecycle hooks; allocation hooks contain observations only."""
    lifecycle_scope(cycles, 2)
    initializer, call, worker = numerical_worker or ("", "", {"configured": False})
    marker = "// @NUMERICAL_WORKER_BOOTSTRAP@"
    if worker["configured"] and template.count(marker) != 1:
        raise ValueError("lifecycle runner lacks one post-handshake bootstrap location")
    template = initializer + template.replace(marker, call)
    for name in ("SESSION_LIFECYCLE_BEGIN", "SESSION_LIFECYCLE_END"):
        if template.count(f"@{name}@") != 1:
            raise ValueError("lifecycle template hook is missing or repeated")
    begin = f"""  const std::string lifecycle_root = output_root;
  for (unsigned cycle = 0; cycle < {cycles}u; ++cycle) {{
    const std::string output_root = lifecycle_root + "/cycle-" + std::to_string(cycle);
    if (!std::filesystem::create_directory(output_root)) return 21;
    std::fprintf(stderr, "LIFECYCLE_BEGIN,%u\\n", cycle);"""
    end = """    session = nullptr;
    std::ifstream destroyed_maps("/proc/self/maps");
    std::ofstream destroyed_output(output_root + "/after-destroy-process-maps.txt");
    destroyed_output << destroyed_maps.rdbuf();
    if (!destroyed_output.good()) return 22;
    std::fprintf(stderr, "LIFECYCLE_END,%u\\n", cycle);
  }"""
    values = dict(replacements)
    values.update({
        "SESSION_LIFECYCLE_BEGIN": begin, "SESSION_LIFECYCLE_END": end,
        "ALLOCATOR_INCLUDE": '#include <filesystem>\n#include "session_allocator_observer.h"',
    })
    for name, phase in zip(
        ("BEFORE_SESSION", "AFTER_LOAD", "AFTER_WARMUP", "AFTER_MEASURED", "AFTER_DESTROY"),
        PHASES, strict=True,
    ):
        check = f'(!Cuda(cudaDeviceSynchronize()) || !vlaforge_benchmark::ObserveAllocator(output_root, "{phase}", kOrdinal))'
        values["ALLOCATOR_" + name] = (
            f"    if (run == warmup && {check}) return 19;" if name == "AFTER_WARMUP"
            else f"  if ({check}) return 19;"
        )
    for name, value in values.items():
        template = template.replace(f"@{name}@", str(value))
    if re.search(r"@[A-Z_]+@", template):
        raise ValueError("lifecycle runner has unresolved template fields")
    return template


def lifecycle_replay_counters(loops, *, cycles, samples):
    lifecycle_scope(cycles, samples)
    counters = []
    for loop in loops:
        if loop["policy"] not in ("required", "batch-only"):
            raise ValueError("unsupported lifecycle replay policy")
        fields = (1, loop["steps"], samples, 0) if loop["policy"] == "required" else (0, 0, 0, samples)
        counters.append(",".join(map(str, ("REPLAY_FINAL", loop["task_id"], *fields))))
    return counters * cycles


def graph_memory_configuration(manifest, bundle, selection):
    """Reuse the bundle builder's audited policy, without silently resetting it."""
    from vlaforge.deployment.build import _libtorch_graph_memory_configuration

    common = _benchmark()
    path = bundle / 'metadata/build_configuration.json'
    original = common.read(path).get('libtorch_graph_memory_policy') if path.is_file() else None
    if path.is_file() and original not in ('retain', 'scoped-reclaim'):
        raise ValueError('invalid source graph memory policy metadata')
    if selection == 'source' and original is None:
        raise ValueError('source graph memory policy metadata missing; choose an explicit policy')
    selected = original if selection == 'source' else selection
    versions = (dict(manifest.backend_versions) if isinstance(manifest.backend_versions, dict)
                else {item.name: item.version for item in manifest.backend_versions})
    configuration = _libtorch_graph_memory_configuration(
        selected, {item.region_name: item for item in manifest.region_artifacts}, versions
    )
    return {'schema': 'vlaforge.lifecycle_graph_memory/1', 'selection': selection,
        'source_policy': original, 'source_configuration_sha256': common.sha(path) if path.is_file() else None,
        'build_configuration': configuration}


def prepare(args):
    common = _benchmark()
    origin, root = args.prepared.resolve(), args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("lifecycle output must be empty")
    if args.cycles < 5:
        raise ValueError("native lifecycle diagnostics require at least five cycles")
    common.verify_frozen(origin)
    protocol = common.read(origin / "protocol.json")
    common.validate_protocol(protocol)
    expected_uuid = args.gpu_uuid or protocol.get("monitor_gpu", "")
    if not isinstance(expected_uuid, str) or not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", expected_uuid):
        raise ValueError("lifecycle preparation requires an explicit GPU UUID")
    if protocol["gpu_ordinal"] != 0 and "cuda_visible_devices" not in protocol:
        raise ValueError("nonzero worker ordinal requires an explicit visible-device mapping")
    scope = lifecycle_scope(args.cycles, len(protocol["samples"]))
    contract = common.read(origin / "tensor-contract.json")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("data", "source/runtime"):
        shutil.copytree(origin / name, root / name)
    shutil.copy2(origin / "source/session_allocator_observer.h", root / "source/session_allocator_observer.h")
    template_path = Path(__file__).with_name("session_benchmark_runner.cpp.in")
    template = template_path.read_text()
    for path in (
        Path(__file__), Path(__file__).with_name("benchmark_session.py"), template_path,
        SOURCE / "python/vlaforge/validation/session_lifecycle.py",
        SOURCE / "python/vlaforge/validation/session_benchmark.py",
        SOURCE / "python/vlaforge/validation/allocator_metrics.py",
    ):
        shutil.copy2(path, root / "source" / path.name)
    common.write(root / "protocol.json", protocol)
    common.write(root / "tensor-contract.json", contract)
    binding = dict(common.read(origin / "frozen-files.json"))
    modules, workers, runtime_libraries, loop_schedules, graph_memory = [], {}, {}, {}, {}
    for policy in common.protocol_policies(protocol):
        bundle = Path(protocol["bundles"][policy])
        manifest = common.load_bundle_manifest(bundle / "bundle.json")
        manifest.verify_files(bundle)
        graph_memory[policy] = graph_memory_configuration(manifest, bundle, args.graph_memory_policy)
        worker = common.numerical_worker_initialization(
            protocol, (item.numerical_binding for item in manifest.region_artifacts)
        )
        workers[policy] = worker[2]
        mode = protocol.get("bundle_metadata_mode", "selection-manifest")
        ir = "semantic_ir.json" if mode == "ordinary-source" else "input_semantic_ir.json"
        module = module_from_data(common.read(bundle / "metadata" / ir))
        modules.append(module)
        if common.benchmark_output_contract(module, protocol) != contract:
            raise ValueError("lifecycle output contract differs from source benchmark")
        declarations, ordinal, count = common.input_specs(module, contract=contract)
        if ordinal != protocol["gpu_ordinal"]:
            raise ValueError("lifecycle CUDA ordinal differs")
        plan = common.read(bundle / "metadata/scheduled_plan.json")
        selected = bundle / "metadata/loop_execution.json"
        common.validate_loop_policy(common.read(selected) if selected.is_file() else None,
                                    plan, policy, mode)
        loops = [{"task_id": task["id"], "policy": policy,
                  "steps": len(range(task["attributes"]["lower"], task["attributes"]["upper"], task["attributes"]["step"]))}
                 for task in plan["tasks"] if task["opcode"] == "vla.for" and policy != "off"]
        loop_schedules[policy] = loops
        checks, failures = common.replay_checks(loops)
        source = root / "generated" / policy
        shutil.copytree(origin / "generated" / policy, source)
        shutil.copy2(root / "source/session_allocator_observer.h", source / "session_allocator_observer.h")
        replacements = {
            "INPUTS": declarations, "ORDINAL": ordinal, "SAMPLES": len(protocol["samples"]),
            "OUTPUT_COUNT": count, "OUTPUT_BYTES": contract["size_bytes"],
            "OUTPUT_DTYPE": "VLAFORGE_DTYPE_" + contract["dtype"].upper(),
            "OUTPUT_SHAPE": ",".join(map(str, contract["shape"])),
            "ACTIVE_INDICES": ",".join(map(str, contract["active_indices"])),
            "OUTPUT_FILE": contract["raw_file"], "PRIMARY_OUTPUT_INDEX": contract.get("index", 0),
            "MULTI_OUTPUT_DEFINE": "#define VLAFORGE_BENCHMARK_MULTI_OUTPUT 1" if "outputs" in contract else "",
            "ADDITIONAL_OUTPUT_SPECS": common.additional_output_declarations(contract),
            "OWNER_HANDSHAKE": str(protocol.get("owner_identity_mode") == "cuda-registration-handshake").lower(),
            "REPLAY_AUDIT": checks, "REPLAY_FAILURE": failures,
        }
        (source / "runner.cpp").write_text(render_runner(
            template, replacements=replacements, cycles=args.cycles, numerical_worker=worker
        ))
        cmake = source / "CMakeLists.txt"
        link = "target_link_libraries(vlaforge_generated_runner PRIVATE torch_cuda c10_cuda)"
        if link not in cmake.read_text():
            cmake.write_text(cmake.read_text() + "\n" + link + "\n")
        build = root / "build" / policy
        environment = {**os.environ, "TORCH_CUDA_ARCH_LIST": protocol["cuda_arch"]}
        common.command([
            "cmake", "-S", str(source), "-B", str(build),
            "-DVLAFORGE_RUNTIME_ROOT=" + str(root / "source/runtime"),
            "-DCMAKE_PREFIX_PATH=" + protocol["cmake_prefix_path"],
            "-DBUILD_TESTING=OFF", "-DCMAKE_BUILD_TYPE=Release",
            '-D' + graph_memory[policy]['build_configuration']['cmake_definition'],
        ], root / "logs" / f"{policy}-configure.json", env=environment)
        common.command(["cmake", "--build", str(build), "--parallel", "2"],
                       root / "logs" / f"{policy}-build.json", env=environment)
        binary = build / "vlaforge_generated_runner"
        linked = common.command(["ldd", str(binary)], root / "logs" / f"{policy}-ldd.json")
        if any(name in linked.lower() for name in ("libpython", "libtorch_python", "not found")):
            raise ValueError("lifecycle runner has Python or unresolved dependencies")
        binding[str(binary)] = common.sha(binary)
        for path in build.rglob("*.so*"):
            if path.is_file():
                binding[str(path)] = common.sha(path)
        runtime_libraries[policy] = common.runtime_library_records(build, bundle, manifest)
        for library in runtime_libraries[policy]:
            binding[library["path"]] = library["sha256"]
        cache = build / 'CMakeCache.txt'
        definition = graph_memory[policy]['build_configuration']['cmake_definition']
        key, value = definition.split('=', 1)
        if key + ':BOOL=' + value not in cache.read_text().splitlines():
            raise ValueError('actual lifecycle CMake graph memory policy differs')
        binding[str(cache)] = common.sha(cache)
    if any(module != modules[0] for module in modules):
        raise ValueError("lifecycle policies do not share identical original IR")
    caller_bytes = len(protocol["samples"]) * sum(
        (root / "data/0" / f"{index}.bin").stat().st_size
        for index in range(len(modules[0].inputs))
    )
    common.verify_frozen(origin)
    common.write(root / "source/numerical-workers.json", workers)
    common.write(root / "source/runtime-libraries.json", runtime_libraries)
    common.write(root / "source/lifecycle-loops.json", loop_schedules)
    common.write(root / "source/graph-memory-policies.json", graph_memory)
    for directory in ("data", "generated", "source"):
        for path in (root / directory).rglob("*"):
            if path.is_file():
                binding[str(path)] = common.sha(path)
    for path in (root / "protocol.json", root / "tensor-contract.json", Path(__file__),
                 Path(__file__).with_name("benchmark_session.py"),
                 SOURCE / "python/vlaforge/validation/session_lifecycle.py",
                 SOURCE / "python/vlaforge/validation/session_benchmark.py",
                 SOURCE / "python/vlaforge/validation/allocator_metrics.py"):
        binding[str(path)] = common.sha(path)
    common.write(root / "frozen-files.json", binding)
    common.write(root / "prepared.json", {
        "schema": "vlaforge.session_lifecycle_prepared/1", "status": "prepared_not_executed",
        "origin_prepared": str(origin), "origin_prepared_sha256": common.sha(origin / "prepared.json"),
        "origin_frozen_files_sha256": common.sha(origin / "frozen-files.json"),
        "frozen_files_sha256": common.sha(root / "frozen-files.json"),
        "protocol_sha256": common.sha(root / "protocol.json"), **scope,
        "caller_input_cuda_malloc_bytes": caller_bytes,
        "expected_gpu_uuid": expected_uuid,
        "numerical_worker_initialization": workers,
        "loop_schedules": loop_schedules,
        "graph_memory_policies": graph_memory,
        "hardware_comparison": "same UUID is required at execution; no cross-device memory attribution",
        "runtime_source": "copied byte-for-byte from origin prepared benchmark; current main runtime not used",
    })


def _run_process(common, root, folder, protocol, policy, samples, expected_uuid):
    monitor = expected_uuid
    preflight = common.telemetry(monitor)
    common.write(folder / "preflight.json", preflight)
    if preflight["exit_code"] != 0 or preflight["compute_owners"]:
        raise ValueError("lifecycle GPU is unavailable or has another compute owner")
    hardware = list(csv.DictReader(preflight["stdout"].splitlines(), skipinitialspace=True))
    if len(hardware) != 1 or hardware[0].get("uuid", "").strip() != expected_uuid:
        raise ValueError("lifecycle preflight GPU UUID differs from prepared expectation")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(protocol.get("cuda_visible_devices", expected_uuid))}
    for name in ("PYTHONPATH", "PYTHONHOME"):
        environment[name] = "/nonexistent-lifecycle-native"
    command = [str(root / "build" / policy / "vlaforge_generated_runner"),
               protocol["bundles"][policy], str(root / "data"), str(folder), "1", str(samples - 1)]
    execution = {"status": "started", "command": command, "policy": policy,
                 "start_ns": time.time_ns(), "performance_measurement": False,
                 "owner_identity_mode": protocol.get("owner_identity_mode", "process-pid"),
                 "monitor_gpu": monitor, "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"]}
    process = None
    try:
        with (folder / "stdout.csv").open("w") as output, (folder / "stderr.log").open("w") as error:
            process = subprocess.Popen(command, stdout=output, stderr=error, env=environment)
            execution["pid"] = process.pid
            common.write(folder / "execution.json", execution)
            expected = process.pid
            if protocol.get("owner_identity_mode") == "cuda-registration-handshake":
                expected = common.register_gpu_owner(process, folder, {**protocol, "monitor_gpu": monitor})
            execution["monitored_nvml_pid"] = expected
            with (folder / "telemetry.jsonl").open("w") as telemetry:
                while process.poll() is None:
                    record = common.telemetry(monitor)
                    telemetry.write(json.dumps(record) + "\n")
                    telemetry.flush()
                    if record["exit_code"] != 0 or any(owner["pid"] != expected for owner in record["compute_owners"]):
                        raise ValueError("lifecycle monitoring failed or found a foreign GPU owner")
                    time.sleep(1)
            execution.update(exit_code=process.wait(), status="executed")
            if execution["exit_code"] != 0:
                raise ValueError("native lifecycle process failed; partial evidence retained")
    except BaseException as error:
        execution.update(status="failed", error=f"{type(error).__name__}: {error}")
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        common.write(folder / "execution.json", execution)


def verify_policy(common, root, folder, protocol, contract, cycles):
    execution = common.read(folder / "execution.json")
    if (execution.get("status") != "executed" or execution.get("exit_code") != 0
            or type(execution.get("pid")) is not int or execution["pid"] <= 0
            or type(execution.get("monitored_nvml_pid")) is not int):
        raise ValueError("lifecycle native execution is failed or incomplete")
    owner_pid = execution["monitored_nvml_pid"]
    mode = protocol.get("owner_identity_mode", "process-pid")
    if (owner_pid <= 0 or execution.get("owner_identity_mode") != mode
            or (mode == "process-pid" and owner_pid != execution["pid"])):
        raise ValueError("lifecycle owner identity differs from selected protocol")
    preflight = common.read(folder / "preflight.json")
    if preflight["exit_code"] != 0 or preflight["compute_owners"]:
        raise ValueError("lifecycle preflight has a foreign owner or failed")
    if mode == "cuda-registration-handshake":
        handshake = common.read(folder / "owner-handshake.json")
        if (handshake.get("status") != "passed" or handshake.get("nvml_pid") != owner_pid
                or handshake.get("container_pid") != execution["pid"]):
            raise ValueError("lifecycle handshake evidence is incomplete")
    telemetry = [parse_allocator_json(line) for line in (folder / "telemetry.jsonl").read_text().splitlines()]
    if not telemetry or not any(record["compute_owners"] for record in telemetry):
        raise ValueError("lifecycle has no actual sampled CUDA owner")
    if any(record["exit_code"] != 0 or any(owner["pid"] != owner_pid for owner in record["compute_owners"])
           for record in telemetry):
        raise ValueError("lifecycle sampled CUDA owner verification failed")
    if {path.name for path in folder.glob("cycle-*")} != {f"cycle-{index}" for index in range(cycles)}:
        raise ValueError("lifecycle cycle directories differ from declared count")
    rows = split_lifecycle_rows((folder / "stdout.csv").read_text(), cycles=cycles,
                                samples=len(protocol["samples"]))
    output_specs = contract.get("outputs", [{**contract, "role": "primary-action", "name": "primary",
                                             "direct_file": "direct.bin", "eager_file": "eager.bin"}])
    policy = folder.name
    numerical = {"mode": "none", "configured": False}
    if protocol.get("numerical_worker_bootstrap") is not None or (root / "source/numerical-workers.json").exists():
        numerical = common.verify_numerical_worker_execution(root, folder, policy)
    loop_path = root / "source/lifecycle-loops.json"
    replay = {"verified": False, "scope": "not-recorded-legacy"}
    if loop_path.exists():
        schedules = common.read(loop_path)
        if schedules != common.read(root / "prepared.json")["loop_schedules"]:
            raise ValueError("lifecycle loop schedule binding differs")
        observed = [line for line in (folder / "stderr.log").read_text().splitlines()
                    if line.startswith("REPLAY_FINAL,")]
        expected = lifecycle_replay_counters(schedules[policy], cycles=cycles, samples=len(protocol["samples"]))
        if observed != expected:
            raise ValueError("lifecycle replay counters do not cover every Session")
        replay = {"verified": True, "counters": observed, "schedule_sha256": common.sha(loop_path)}
    memory_path = root / 'source/graph-memory-policies.json'
    memory_policy = None
    if memory_path.exists():
        policies = common.read(memory_path)
        if policies != common.read(root / 'prepared.json')['graph_memory_policies']:
            raise ValueError('lifecycle graph memory policy binding differs')
        memory_policy = policies[policy]
    snapshots, outputs, hashes, libraries = [], [], {}, []
    for cycle, cycle_rows in enumerate(rows):
        directory = folder / f"cycle-{cycle}"
        snapshots.append(parse_allocator_snapshots((directory / "allocator-snapshots.jsonl").read_text()))
        for name in ("process-maps.txt", "after-destroy-process-maps.txt"):
            maps = (directory / name).read_text()
            if not maps or "libpython" in maps.lower() or "libtorch_python" in maps.lower():
                raise ValueError("lifecycle process maps contain Python or are empty")
        libraries.append(common.verify_runtime_library_execution(root, directory, policy))
        for spec in output_specs:
            raw = (directory / spec["raw_file"]).read_bytes()
            if len(raw) != len(cycle_rows) * spec["size_bytes"]:
                raise ValueError("lifecycle complete output archive is truncated")
            checks = []
            for sample in range(len(cycle_rows)):
                offset = sample * spec["size_bytes"]
                source = root / "data" / str(sample)
                check = common.compare_output_bytes(
                    raw[offset:offset + spec["size_bytes"]],
                    (source / spec["direct_file"]).read_bytes(),
                    (source / spec["eager_file"]).read_bytes(), spec,
                )
                if not check["eager_bitwise_equal"]:
                    raise ValueError("lifecycle output differs from complete official reference")
                checks.append({"sample": sample, **check})
            outputs.append({"cycle": cycle, "output": spec["name"], "checks": checks})
        for path in directory.iterdir():
            if path.is_file():
                hashes[str(path.relative_to(folder))] = common.sha(path)
    allocator = allocator_lifecycle_report(snapshots, expected_cycles=cycles, ordinal=protocol["gpu_ordinal"])
    result = {
        "schema": "vlaforge.session_lifecycle_report/1", "status": "passed",
        **lifecycle_scope(cycles, len(protocol["samples"])),
        "allocator": allocator, "complete_outputs": outputs, "raw_evidence_sha256": hashes,
        "same_artifact_and_official_full_output_bitwise_equal": True,
        "execution_sha256": common.sha(folder / "execution.json"),
        "telemetry_sha256": common.sha(folder / "telemetry.jsonl"),
        "preflight_sha256": common.sha(folder / "preflight.json"),
        "stdout_sha256": common.sha(folder / "stdout.csv"),
        "no_python_deployment": True, "full_paper_acceptance": False,
        "numerical_worker_initialization": numerical,
        "runtime_library_evidence_by_cycle": libraries,
        "replay_evidence": replay,
        "graph_memory_policy": memory_policy,
        "monitoring_limitation": "one-second owner samples cannot exclude a foreign process entirely between samples",
    }
    common.write(folder / "report.json", result)
    return result


def run(args):
    common = _benchmark()
    root = args.output.resolve()
    common.verify_frozen(root)
    prepared, protocol = common.read(root / "prepared.json"), common.read(root / "protocol.json")
    contract = common.read(root / "tensor-contract.json")
    reports = {}
    for policy in common.protocol_policies(protocol):
        folder = root / "runs" / policy
        folder.mkdir(parents=True, exist_ok=False)
        _run_process(common, root, folder, protocol, policy, len(protocol["samples"]), prepared["expected_gpu_uuid"])
        reports[policy] = verify_policy(common, root, folder, protocol, contract, prepared["cycles"])
        common.verify_frozen(root)
    common.write(root / "report.json", {
        "schema": "vlaforge.session_lifecycle_campaign/1", "status": "passed",
        "prepared_sha256": common.sha(root / "prepared.json"),
        "policies": {name: {"report_sha256": common.sha(root / "runs" / name / "report.json"),
                            "retained_after_destroy": report["allocator"]["retained_after_destroy"]}
                     for name, report in reports.items()},
        **lifecycle_scope(prepared["cycles"], len(protocol["samples"])),
        "process_count": len(reports), "sessions_created": prepared["cycles"] * len(reports),
        "validated_calls_per_policy": prepared["cycles"] * len(protocol["samples"]),
        "validated_calls": prepared["cycles"] * len(protocol["samples"]) * len(reports),
        "full_paper_acceptance": False,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--gpu-uuid")
    parser.add_argument('--graph-memory-policy', choices=('source', 'retain', 'scoped-reclaim'), default='source')
    args = parser.parse_args(argv)
    if args.stage == "prepare" and args.prepared is None:
        parser.error("prepare requires --prepared")
    try:
        (prepare if args.stage == "prepare" else run)(args)
    except Exception as error:
        _benchmark().write(args.output / f"{args.stage}-failure.json", {
            "status": "failed", "error": f"{type(error).__name__}: {error}",
            "performance_measurement": False,
        })
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
