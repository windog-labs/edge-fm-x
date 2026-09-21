import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def tool(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / 'tools'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('operator_repeat_audit_test', folder / 'audit_operator_repeats.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_or_empty_timing_samples_rejected(tool):
    for values in ([], [0], [-1], [float('nan')], [float('inf')], [True]):
        with pytest.raises(ValueError, match='invalid measured'):
            tool.timing_summary(values)


@pytest.fixture
def campaign(tool, tmp_path):
    torch = pytest.importorskip('torch')
    expected = (torch.tensor([0.0, 1.0]),)
    identity = tool.output_identity(expected)
    record = {'schema': 'vlaforge.operator_repeat_campaign/2', 'status': 'passed',
              'ownership_mode': 'register-reset-register-before-torch', 'gpu_uuid': 'GPU-test',
              'repeats': 5, 'candidate_recipe': 'inductor-aten', 'examples_sha256': 'source', 'runs': [],
              'selected_for_deployment': False, 'end_to_end_integrated': False}
    for index, (recipe, repeat) in enumerate([('inductor-aten', None)] + [(recipe, i) for i in range(5) for recipe in ('aten', 'inductor-aten')]):
        name = 'qualification' if repeat is None else f'{recipe}-{repeat:02d}'
        folder = tmp_path / name
        folder.mkdir()
        torch.save({'reference': expected, 'candidate': expected}, folder / 'outputs.pt')
        torch.save({'source_reference': expected, 'target_reference': expected}, folder / 'reference-comparison.pt')
        (folder / 'benchmark-library-source.py').write_text('# test-only library record\n')
        timing = {'raw_gpu_batch_means_ms': [1.0, 2.0], 'mean_ms': 1.5, 'median_ms': 1.5,
                  'library_source_sha256': tool.digest(folder / 'benchmark-library-source.py')}
        report = {'status': 'measured', 'recipe': recipe, 'gpu': 'test GPU', 'torch': 'test', 'cuda': 'test',
                  'compute_capability': [9, 0], 'target_numerical_context': {}, 'source_manifest_sha256': 'source',
                  'outputs_sha256': tool.digest(folder / 'outputs.pt'), 'target_reference_identity': identity,
                  'reference_comparison_sha256': tool.digest(folder / 'reference-comparison.pt'),
                  'source_reference_bitwise_equal': True, 'measurement': timing, 'artifact_sha256': 'candidate'}
        (folder / 'report.json').write_text(json.dumps(report))
        monitor = tmp_path / ('monitor-' + name)
        monitor.mkdir()
        (monitor / 'preflight-owners.json').write_text('[]')
        pid, nvml = 10 + index, 100 + index
        data = {'status': 'exited', 'exitcode': 0, 'container_pid': pid, 'nvml_pid': nvml, 'monitored_gpu': 'GPU-test',
                'observations': [{'stage': stage, 'ready': {'pid': pid, 'ordinal': 0},
                                  'owners': [] if stage == 'reset' else [{'pid': nvml}]}
                                 for stage in ('registered-1', 'reset', 'registered-2')]}
        (monitor / 'monitor.json').write_text(json.dumps(data))
        entry = {'recipe': recipe, 'repeat': repeat, 'command': ['python', 'worker.py', '--output', str(folder)],
                 'pid': pid, 'nvml_pid': nvml, 'exit_code': 0, 'preflight': {'owners': [], 'device': 'GPU-test, test GPU'},
                 'start_ns': index * 2 + 1, 'end_ns': index * 2 + 2,
                 'report_sha256': tool.digest(folder / 'report.json'), 'monitor_sha256': tool.digest(monitor / 'monitor.json'),
                 'measurement': timing}
        if repeat is None:
            record['qualification'] = entry
        else:
            record['runs'].append(entry)
    record['summary'] = {recipe: {'process_medians_ms': [1.5] * 5, 'mean_process_median_ms': 1.5}
                         for recipe in ('aten', 'inductor-aten')}
    (tmp_path / 'campaign.json').write_text(json.dumps(record))
    return tmp_path, record


def test_complete_archived_outputs_and_pair_count(tool, campaign):
    root, _ = campaign
    result = tool.audit(root)
    assert result['repeated_processes'] == 10
    assert result['whole_tensor_outputs_exact'] is True


def test_standalone_cli_finds_repo_package_without_pythonpath(tool, campaign):
    root, _ = campaign
    output = root / 'audit.json'
    result = subprocess.run([sys.executable, tool.__file__, '--campaign', str(root), '--output', str(output)],
                            env={**os.environ, 'PYTHONPATH': '', 'CUDA_VISIBLE_DEVICES': ''},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())['status'] == 'passed'


def test_changed_tensor_rejected_even_if_report_hashes_are_updated(tool, campaign):
    import torch
    root, data = campaign
    folder = root / 'aten-00'
    torch.save({'reference': (torch.tensor([0.0, 1.0]),), 'candidate': (torch.tensor([-0.0, 1.0]),)}, folder / 'outputs.pt')
    report = json.loads((folder / 'report.json').read_text())
    report['outputs_sha256'] = tool.digest(folder / 'outputs.pt')
    (folder / 'report.json').write_text(json.dumps(report))
    data['runs'][0]['report_sha256'] = tool.digest(folder / 'report.json')
    (root / 'campaign.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='complete output tensor'):
        tool.audit(root)


def test_foreign_owner_cannot_be_hidden_by_pass_status(tool, campaign):
    root, data = campaign
    path = root / 'monitor-aten-00/monitor.json'
    monitor = json.loads(path.read_text())
    monitor['observations'][0]['owners'].append({'pid': 123456789})
    path.write_text(json.dumps(monitor))
    data['runs'][0]['monitor_sha256'] = tool.digest(path)
    (root / 'campaign.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='foreign GPU'):
        tool.audit(root)


def test_concurrent_workers_are_not_independent_latency_pairs(tool, campaign):
    root, data = campaign
    data['runs'][1]['start_ns'] = data['runs'][0]['start_ns']
    (root / 'campaign.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='overlap'):
        tool.audit(root)


def test_legacy_pid_guessing_is_not_upgraded_to_registered_evidence(tool, campaign):
    root, data = campaign
    data['schema'] = 'vlaforge.operator_repeat_campaign/1'
    (root / 'campaign.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='v2 required'):
        tool.audit(root)
