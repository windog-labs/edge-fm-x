import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def tool(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / 'tools'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('operator_repeats_test', folder / 'run_operator_microbenchmark_repeats.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_proc_entry_does_not_grant_ownership(tool, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError('PID is outside this namespace')
    monkeypatch.setattr(tool.Path, 'read_text', missing)
    assert not tool.own_process(12345, 20000)


def test_snapshot_rejects_wrong_physical_uuid(tool, monkeypatch):
    monkeypatch.setattr(tool.subprocess, 'check_output', lambda *a, **k: '' if '--query-compute-apps=gpu_uuid,pid,process_name,used_memory' in a[0] else 'GPU-other, H20, 535, 0, 0, 1980\n')
    with pytest.raises(ValueError, match='UUID'):
        tool.gpu_snapshot('GPU-requested')


def test_repeats_use_registered_monitor_and_keep_gpu_selection(tool, monkeypatch, tmp_path):
    worker = tmp_path / 'benchmark_operator_examples.py'
    worker.write_text('')
    examples = tmp_path / 'examples.json'
    examples.write_text('{}')
    calls = []
    monkeypatch.setattr(tool, 'gpu_snapshot', lambda gpu: {'owners': [], 'device': gpu})

    def monitored(command, folder, *, gpu, environment):
        assert gpu == environment['CUDA_VISIBLE_DEVICES'] == 'GPU-selected'
        assert all(isinstance(arg, str) for arg in command)
        target = Path(command[command.index('--output') + 1])
        target.mkdir()
        folder.mkdir()
        index = len(calls)
        (folder / 'monitor.json').write_text(json.dumps({'status': 'exited', 'exitcode': 0, 'container_pid': 10+index, 'nvml_pid': 100+index}))
        (target / 'report.json').write_text(json.dumps({'status': 'measured', 'correctness_passed': True,
            'target_reference_identity': {'same': 'output'}, 'source_reference_bitwise_equal': True,
            'measurement': {'median_ms': 1.0}, 'recipe': command[command.index('--recipe') + 1]}))
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool, 'run_monitored', monitored)
    monkeypatch.setattr(sys, 'argv', ['repeats', '--tool', str(worker), '--examples', str(examples), '--node', 'node',
        '--output', str(tmp_path / 'out'), '--gpu', 'GPU-selected', '--repeats', '2', '--recipe', 'inductor-aten-preserving'])
    assert tool.main() == 0
    assert len(calls) == 5  # One qualification, then two sequential baseline/candidate pairs.
    record = json.loads((tmp_path / 'out/campaign.json').read_text())
    assert len(record['runs']) == 4
    assert record['qualification']['pid'] == 10
    assert record['schema'] == 'vlaforge.operator_repeat_campaign/2'
    assert record['selected_for_deployment'] is False


def test_failed_registration_does_not_start_another_worker(tool, monkeypatch, tmp_path):
    worker = tmp_path / 'worker.py'
    worker.write_text('')
    examples = tmp_path / 'examples.json'
    examples.write_text('{}')
    monkeypatch.setattr(tool, 'gpu_snapshot', lambda gpu: {'owners': [], 'device': gpu})
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise RuntimeError('foreign owner during registration')
    monkeypatch.setattr(tool, 'run_monitored', fail)
    monkeypatch.setattr(sys, 'argv', ['repeats', '--tool', str(worker), '--examples', str(examples), '--node', 'node',
        '--output', str(tmp_path / 'out'), '--gpu', 'GPU-selected'])
    assert tool.main() == 1
    assert len(calls) == 1
