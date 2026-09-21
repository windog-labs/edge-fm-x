"""Explicit host timing boundaries and raw-evidence rejection tests."""

import copy
import csv

import pytest
from vlaforge.validation import session_benchmark as bench
from test_session_benchmark import benchmark_tool, protocol


def test_host_boundary_is_explicit_and_legacy_default_retained():
    original = protocol()
    bench.validate_protocol(original)
    assert bench.includes_host_io(original) is False
    host = copy.deepcopy(original)
    host['boundary'] = bench.HOST_BOUNDARY
    bench.validate_protocol(host)
    assert bench.includes_host_io(host) is True
    host['boundary'] = 'sensor-to-action'
    with pytest.raises(ValueError, match='boundary'):
        bench.validate_protocol(host)


def test_host_timing_matches_elapsed_and_conserves_segments():
    data = [{'run': '0', 'h2d_ns': '10', 'bind_ns': '20', 'model_ns': '100', 'd2h_ns': '30', 'host_call_ns': '160'}]
    rows = [{'run': '0', 'latency_ns': '160'}]
    bench.validate_host_timing(data, rows)
    with pytest.raises(ValueError, match='missing'):
        bench.validate_host_timing([], rows)
    data[0]['d2h_ns'] = '0'
    with pytest.raises(ValueError, match='segments'):
        bench.validate_host_timing(data, rows)
    data[0]['d2h_ns'] = '30'
    rows[0]['latency_ns'] = '100'
    with pytest.raises(ValueError, match='boundary'):
        bench.validate_host_timing(data, rows)


@pytest.mark.parametrize('field,value', [('run', '1'), ('h2d_ns', '-1'), ('model_ns', '0'),
    ('d2h_ns', 'nan'), ('bind_ns', 2.5), ('host_call_ns', True)])
def test_invalid_call_or_segment_is_rejected(field, value):
    data = [{'run': '0', 'h2d_ns': '10', 'bind_ns': '20', 'model_ns': '100', 'd2h_ns': '30', 'host_call_ns': '160'}]
    data[0][field] = value
    with pytest.raises(ValueError):
        bench.validate_host_timing(data, [{'run': '0', 'latency_ns': '160'}])


def test_raw_timing_binding_detects_changed_segments_even_when_total_is_preserved(tmp_path):
    api = benchmark_tool()
    value = dict(protocol(), boundary=bench.HOST_BOUNDARY)
    rows = [{'run': '0', 'latency_ns': '160'}]
    with pytest.raises(ValueError, match='missing'):
        api.verify_host_timing_evidence(tmp_path, value, rows)
    path = tmp_path / 'host-timing.csv'
    columns = ['run', 'h2d_ns', 'bind_ns', 'model_ns', 'd2h_ns', 'host_call_ns']
    with path.open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerow([0, 10, 20, 100, 30, 160])
    binding = api.verify_host_timing_evidence(tmp_path, value, rows)
    assert api.verify_host_timing_evidence(tmp_path, value, rows, expected=binding) == binding
    path.write_text(path.read_text().replace('10,20,100,30', '11,19,100,30'))
    with pytest.raises(ValueError, match='binding differs'):
        api.verify_host_timing_evidence(tmp_path, value, rows, expected=binding)
    with pytest.raises(ValueError, match='resident boundary'):
        api.verify_host_timing_evidence(tmp_path, protocol(), rows, expected=binding)
