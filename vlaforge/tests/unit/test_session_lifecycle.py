"""CPU contract cases only; native memory observations require the diagnostic tool."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from vlaforge.validation.allocator_metrics import COUNTS, GROUPS, PHASES
from vlaforge.validation.session_lifecycle import (
    allocator_lifecycle_report,
    lifecycle_scope,
    split_lifecycle_rows,
)


def _records(growth=0):
    result = []
    for cycle in range(5):
        rows = []
        for phase_index, phase in enumerate(PHASES):
            index = 5 * cycle + phase_index
            current = 10 + cycle * growth
            counters = {
                name: {"current": current, "peak": 1000,
                       "allocated": 1000 + 100 * index + current,
                       "freed": 1000 + 100 * index}
                for name in GROUPS
            }
            counters.update(dict.fromkeys(COUNTS, index))
            rows.append({"schema": "vlaforge.libtorch_allocator_snapshot/1",
                         "phase": phase, "torch_release": "2.10.0", "allocator": "native",
                         "ordinal": 0, "device_free_bytes": 100, "device_total_bytes": 200,
                         "initialized": True, "counters": counters})
        result.append(rows)
    return result


def _csv(cycles=5, samples=3):
    header = "run,sample,measured,revision,finite,direct_exact\n"
    return "".join(header + "".join(
        f"{run},{run},{int(run >= 1)},{run + 1},1,1\n" for run in range(samples)
    ) for _ in range(cycles))


def test_plateau_is_observed_without_claiming_reclaimed_memory_or_no_leak():
    report = allocator_lifecycle_report(_records(), expected_cycles=5, ordinal=0)
    for group in GROUPS:
        observed = report["retained_after_destroy"][group]
        assert observed["current_after_destroy"] == [10] * 5
        assert observed["unchanged_after_first_destroy"] is True
        assert observed["strictly_growing_each_cycle"] is False
    assert report["zero_allocation_claim_verified"] is False
    assert report["leak_attribution_verified"] is False
    assert report["performance_measurement"] is False
    assert report["excludes"]


def test_growth_is_preserved_separately_for_allocated_active_and_reserved():
    report = allocator_lifecycle_report(_records(2), expected_cycles=5, ordinal=0)
    for group in GROUPS:
        assert report["retained_after_destroy"][group]["current_after_destroy"] == [10, 12, 14, 16, 18]
        assert report["retained_after_destroy"][group]["strictly_growing_each_cycle"] is True
        assert report["retained_after_destroy"][group]["net_change_after_first_destroy"] == 8
    assert report["leak_attribution_verified"] is False


@pytest.mark.parametrize("mutation", ("reset", "conservation", "counts", "uninitialized", "device"))
def test_cross_cycle_faults_are_rejected_even_when_each_cycle_is_valid(mutation):
    rows = _records()
    for item in rows[1]:
        if mutation == "reset":
            item["counters"]["allocation"]["allocated"] -= 200
            item["counters"]["allocation"]["freed"] -= 200
        elif mutation == "conservation":
            item["counters"]["allocated_bytes"]["current"] += 1
        elif mutation == "counts":
            item["counters"]["num_device_alloc"] = 0
        elif mutation == "device":
            item["device_total_bytes"] += 1
    if mutation == "uninitialized":
        rows[1][0].update(initialized=False, counters=None)
    with pytest.raises(ValueError):
        allocator_lifecycle_report(rows, expected_cycles=5, ordinal=0)


def test_missing_cycles_reordered_phases_and_foreign_devices_are_rejected():
    for change in ("missing", "phase", "ordinal"):
        rows = _records()
        if change == "missing":
            rows.pop()
        elif change == "phase":
            rows[2].reverse()
        else:
            rows[2][2]["ordinal"] = 1
        with pytest.raises(ValueError):
            allocator_lifecycle_report(rows, expected_cycles=5, ordinal=0)


def test_published_snapshots_do_not_alias_the_callers_mutable_records():
    rows = _records()
    expected = deepcopy(rows)
    report = allocator_lifecycle_report(rows, expected_cycles=5, ordinal=0)
    rows[0][0]["counters"]["allocation"]["current"] = 99
    assert report["snapshots_by_cycle"] == expected


def test_all_diagnostic_rows_are_present_without_warmup_or_latency_claims():
    rows = split_lifecycle_rows(_csv(), cycles=5, samples=3)
    assert len(rows) == 5 and all(len(cycle) == 3 for cycle in rows)
    scope = lifecycle_scope(5, 16)
    assert scope["validated_calls"] == 80 and scope["template_call_partition"] == [1, 15]
    assert scope["warmup_calls"] is None and scope["measured_calls"] is None
    assert scope["formal_latency_or_cdf"] is False


@pytest.mark.parametrize("mutation", ("truncated", "repeated", "revision", "sample", "nonfinite", "header"))
def test_incomplete_or_reordered_native_rows_cannot_pass(mutation):
    text = _csv()
    if mutation == "truncated":
        text = "\n".join(text.splitlines()[:-1]) + "\n"
    elif mutation == "repeated":
        text += text.splitlines()[1] + "\n"
    elif mutation == "revision":
        text = text.replace("1,1,1,2,1,1", "1,1,1,1,1,1", 1)
    elif mutation == "sample":
        text = text.replace("1,1,1,2,1,1", "1,0,1,2,1,1", 1)
    elif mutation == "nonfinite":
        text = text.replace("1,1,1,2,1,1", "1,1,1,2,0,1", 1)
    else:
        text = text.replace("run,sample,measured,revision,finite,direct_exact", "run,run,measured,revision,finite,direct_exact", 1)
    with pytest.raises(ValueError):
        split_lifecycle_rows(text, cycles=5, samples=3)


@pytest.mark.parametrize("count", (True, 0, 1, -1, 2.0, 100001))
def test_invalid_counts_do_not_create_partial_lifecycle_evidence(count):
    with pytest.raises(ValueError):
        lifecycle_scope(count, 16)


def _tool():
    spec = importlib.util.spec_from_file_location(
        "session_lifecycle_tool", Path(__file__).resolve().parents[2] / "tools/diagnose_session_lifecycle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_lifecycle_hooks_wrap_snapshot_scope_and_leave_input_ownership_outside():
    tool = _tool()
    template = """CALLER_INPUTS
@SESSION_LIFECYCLE_BEGIN@
@ALLOCATOR_BEFORE_SESSION@
CREATE
@ALLOCATOR_AFTER_LOAD@
@ALLOCATOR_AFTER_WARMUP@
RUN
@ALLOCATOR_AFTER_MEASURED@
DESTROY
@ALLOCATOR_AFTER_DESTROY@
@SESSION_LIFECYCLE_END@
RETURN
"""
    rendered = tool.render_runner(template, replacements={}, cycles=5)
    assert rendered.index("CALLER_INPUTS") < rendered.index("for (unsigned cycle")
    assert rendered.index("before_session") < rendered.index("CREATE")
    assert rendered.index("DESTROY") < rendered.index("after_destroy") < rendered.index("LIFECYCLE_END")
    assert "cudaDeviceSynchronize()" in rendered and "emptyCache" not in rendered
    assert "@SESSION_LIFECYCLE" not in rendered
    with pytest.raises(ValueError, match="hook"):
        tool.render_runner(template.replace("@SESSION_LIFECYCLE_END@", ""), replacements={}, cycles=5)
    with pytest.raises(ValueError, match="unresolved"):
        tool.render_runner(template + "@UNKNOWN_FIELD@", replacements={}, cycles=5)


def test_current_template_optional_bootstrap_comment_remains_setter_free():
    import re

    tool = _tool()
    template = Path(tool.__file__).with_name("session_benchmark_runner.cpp.in").read_text()
    assert template.count("// @NUMERICAL_WORKER_BOOTSTRAP@") == 1
    replacements = {
        name: "" for name in re.findall(r"@([A-Z_]+)@", template)
        if name != "NUMERICAL_WORKER_BOOTSTRAP"
    }
    rendered = tool.render_runner(template, replacements=replacements, cycles=5)
    assert "@NUMERICAL_WORKER_BOOTSTRAP@" not in rendered
    assert "vlaforge_initialize_numerical_worker" not in rendered
    with pytest.raises(ValueError, match="unresolved"):
        tool.render_runner(template + "\n@NUMERICAL_WORKER_BOOTSTRAP@", replacements=replacements, cycles=5)


def test_explicit_numerical_worker_is_once_before_all_session_cycles():
    tool = _tool()
    template = "HANDSHAKE\n// @NUMERICAL_WORKER_BOOTSTRAP@\nINPUTS\n@SESSION_LIFECYCLE_BEGIN@\nSESSION\n@SESSION_LIFECYCLE_END@"
    worker = ('WORKER_DEFINITION\n', 'WORKER_CALL', {'configured': True})
    rendered = tool.render_runner(template, replacements={}, cycles=5, numerical_worker=worker)
    assert rendered.startswith('WORKER_DEFINITION')
    assert rendered.count('WORKER_CALL') == 1
    assert rendered.index('HANDSHAKE') < rendered.index('WORKER_CALL') < rendered.index('INPUTS')
    assert rendered.index('WORKER_CALL') < rendered.index('for (unsigned cycle')
    with pytest.raises(ValueError, match='bootstrap location'):
        tool.render_runner(template.replace('// @NUMERICAL_WORKER_BOOTSTRAP@', ''),
                           replacements={}, cycles=5, numerical_worker=worker)


@pytest.mark.parametrize('policy,expected', [('required', 'REPLAY_FINAL,9,1,10,16,0'),
                                           ('batch-only', 'REPLAY_FINAL,9,0,0,0,16')])
def test_lifecycle_counters_cover_each_new_session(policy, expected):
    tool = _tool()
    loops = [{'task_id': 9, 'policy': policy, 'steps': 10}]
    assert tool.lifecycle_replay_counters(loops, cycles=5, samples=16) == [expected] * 5
    assert tool.lifecycle_replay_counters([], cycles=5, samples=16) == []


def _memory_manifest(version='2.10.0+cu128'):
    from types import SimpleNamespace
    return SimpleNamespace(backend_versions={'torchscript': version}, region_artifacts=[SimpleNamespace(
        region_name='any-region', capability=SimpleNamespace(backend='torchscript', target='sm_90'))])


def test_lifecycle_preserves_source_graph_memory_policy(tmp_path):
    tool = _tool()
    metadata = tmp_path / 'metadata/build_configuration.json'
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({'libtorch_graph_memory_policy': 'scoped-reclaim'}))
    selected = tool.graph_memory_configuration(_memory_manifest(), tmp_path, 'source')
    assert selected['source_policy'] == 'scoped-reclaim'
    assert selected['build_configuration']['cmake_definition'] == 'VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM=ON'
    override = tool.graph_memory_configuration(_memory_manifest(), tmp_path, 'retain')
    assert override['source_policy'] == 'scoped-reclaim'
    assert override['build_configuration']['cmake_definition'] == 'VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM=OFF'


def test_lifecycle_memory_policy_requires_explicit_unknown_source_and_audited_sdk(tmp_path):
    tool = _tool()
    with pytest.raises(ValueError, match='source.*metadata'):
        tool.graph_memory_configuration(_memory_manifest(), tmp_path, 'source')
    selected = tool.graph_memory_configuration(_memory_manifest(), tmp_path, 'retain')
    assert selected['source_policy'] is None
    with pytest.raises(ValueError, match='2.10'):
        tool.graph_memory_configuration(_memory_manifest('2.11.0'), tmp_path, 'scoped-reclaim')


def _native_evidence_fixture(tmp_path):
    tool = _tool()
    common = tool._benchmark()
    folder = tmp_path / "runs/off"
    folder.mkdir(parents=True)
    primary = {"name": "action", "role": "primary-action", "dtype": "f32", "shape": [2],
               "size_bytes": 8, "count": 2, "active_indices": [0, 1], "raw_file": "outputs.f32",
               "direct_file": "direct.bin", "eager_file": "eager.bin"}
    auxiliary = {"name": "counter", "role": "exact", "dtype": "i64", "shape": [2],
                 "size_bytes": 16, "count": 2, "raw_file": "counter.i64",
                 "direct_file": "counter-direct.bin", "eager_file": "counter-eager.bin"}
    contract = {**primary, "outputs": [primary, auxiliary]}
    raws = {name: [] for name in ("action", "counter")}
    for sample in range(3):
        directory = tmp_path / "data" / str(sample)
        directory.mkdir(parents=True)
        for spec, value in ((primary, np.asarray([sample, 1], dtype=np.float32)),
                            (auxiliary, np.asarray([2**60 + sample, -sample], dtype=np.int64))):
            raw = value.tobytes()
            raws[spec["name"]].append(raw)
            for key in ("direct_file", "eager_file"):
                (directory / spec[key]).write_bytes(raw)
    records = _records()
    for cycle in range(5):
        directory = folder / f"cycle-{cycle}"
        directory.mkdir()
        for spec in contract["outputs"]:
            (directory / spec["raw_file"]).write_bytes(b"".join(raws[spec["name"]]))
        (directory / "allocator-snapshots.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records[cycle]))
        for name in ("process-maps.txt", "after-destroy-process-maps.txt"):
            (directory / name).write_text("synthetic CPU parser fixture, not native evidence\n")
    (folder / "stdout.csv").write_text(_csv())
    common.write(folder / "execution.json", {"status": "executed", "exit_code": 0,
                 "pid": 123, "monitored_nvml_pid": 123, "owner_identity_mode": "process-pid"})
    common.write(folder / "preflight.json", {"exit_code": 0, "compute_owners": []})
    (folder / "telemetry.jsonl").write_text(json.dumps({"exit_code": 0, "compute_owners": [{"pid": 123}]}) + "\n")
    return tool, common, folder, {"samples": [{}, {}, {}], "gpu_ordinal": 0}, contract


def test_verifier_preserves_every_primary_and_large_integer_output_byte(tmp_path):
    tool, common, folder, protocol, contract = _native_evidence_fixture(tmp_path)
    report = tool.verify_policy(common, tmp_path, folder, protocol, contract, 5)
    assert report["validated_calls"] == 15 and len(report["complete_outputs"]) == 10
    assert all(check["eager_bitwise_equal"] for output in report["complete_outputs"] for check in output["checks"])
    assert report["performance_measurement"] is False


@pytest.mark.parametrize("fault", ("aux-byte", "float-reference", "truncated", "failed-process", "missing-cycle",
                                  "foreign-owner", "unmapped-owner", "preflight-owner", "python-maps"))
def test_failed_partial_or_inexact_lifecycle_cannot_be_published(tmp_path, fault):
    tool, common, folder, protocol, contract = _native_evidence_fixture(tmp_path)
    if fault == "aux-byte":
        path = folder / "cycle-2/counter.i64"
        raw = bytearray(path.read_bytes())
        raw[0] ^= 1
        path.write_bytes(raw)
    elif fault == "float-reference":
        (tmp_path / "data/1/eager.bin").write_bytes(np.asarray([99, 1], dtype=np.float32).tobytes())
    elif fault == "truncated":
        path = folder / "cycle-3/outputs.f32"
        path.write_bytes(path.read_bytes()[:-1])
    elif fault == "failed-process":
        execution = common.read(folder / "execution.json")
        execution["exit_code"] = 12
        common.write(folder / "execution.json", execution)
    elif fault == "missing-cycle":
        (folder / "cycle-4").rename(folder / "failed-cycle-4")
    elif fault == "foreign-owner":
        (folder / "telemetry.jsonl").write_text(json.dumps({"exit_code": 0, "compute_owners": [{"pid": 999}]}) + "\n")
    elif fault == "unmapped-owner":
        (folder / "telemetry.jsonl").write_text(json.dumps({"exit_code": 0, "compute_owners": []}) + "\n")
    elif fault == "preflight-owner":
        common.write(folder / "preflight.json", {"exit_code": 0, "compute_owners": [{"pid": 999}]})
    else:
        (folder / "cycle-1/process-maps.txt").write_text("/usr/lib/libpython.so\n")
    with pytest.raises(ValueError):
        tool.verify_policy(common, tmp_path, folder, protocol, contract, 5)
    assert not (folder / "report.json").exists()
