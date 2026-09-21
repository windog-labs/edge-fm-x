import importlib.util
import json
from pathlib import Path
import statistics

import pytest


@pytest.fixture
def tool(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / 'tools'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('session_campaign_summary_test', folder / 'summarize_session_campaigns.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('entries', [[], [{'id': 'same'}, {'id': 'same'}]])
def test_empty_or_duplicate_campaigns_cannot_publish(tool, tmp_path, entries):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema': 'vlaforge.session_campaign_index/1', 'campaigns': entries}))
    with pytest.raises(ValueError, match='empty or duplicate'):
        tool.main(['--manifest', str(manifest), '--output', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists()


def test_existing_output_not_overwritten(tool, tmp_path):
    with pytest.raises(ValueError, match='output exists'):
        tool.main(['--manifest', str(tmp_path / 'not-needed.json'), '--output', str(tmp_path)])


def test_index_change_during_collection_cannot_publish(tool, tmp_path, monkeypatch):
    manifest = tmp_path / 'index.json'
    manifest.write_text(json.dumps({'schema': 'vlaforge.session_campaign_index/1', 'campaigns': [{'id': 'one'}]}))

    def changed(entry):
        manifest.write_text('{}')
        return {'inputs_sha256': {}}

    monkeypatch.setattr(tool, 'summarize_campaign', changed)
    with pytest.raises(ValueError, match='index changed'):
        tool.main(['--manifest', str(manifest), '--output', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists()


def test_formal_audit_digest_required(tool, tmp_path):
    path = tmp_path / 'audit.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='audit digest'):
        tool.summarize_campaign({'prepared': str(tmp_path), 'formal_audit': {'path': str(path), 'sha256': '0' * 64}})


def test_optional_process_dispersion_is_recomputed_not_discarded(tool):
    means = [1., 2., 3., 4., 5.]
    summary = {'mean_ns': 3., 'count': 5}
    tool.verify_audited_summary(summary, summary, means)
    tool.verify_audited_summary(summary, dict(summary, process_mean_std_ns=statistics.stdev(means)), means)
    with pytest.raises(ValueError, match='audit timing'):
        tool.verify_audited_summary(summary, dict(summary, process_mean_std_ns=0.), means)
    with pytest.raises(ValueError, match='audit timing'):
        tool.verify_audited_summary(summary, dict(summary, unknown_statistic=0.), means)


@pytest.mark.parametrize('status,complete', [('failed', True), ('passed', False)])
def test_failed_and_pilot_audits_are_not_formal_evidence(tool, tmp_path, status, complete):
    path = tmp_path / 'audit.json'
    path.write_text(json.dumps({'status': status, 'formal_experiment_complete': complete}))
    with pytest.raises(ValueError, match='not complete'):
        tool.summarize_campaign({'prepared': str(tmp_path), 'formal_audit': {'path': str(path), 'sha256': tool.sha(path)}})
