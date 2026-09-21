"""Recheck formal Session archives and summarize them without merging platforms."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import statistics

from benchmark_session import read, sha, verify_host_timing_evidence, verify_numerical_worker_execution
from vlaforge.validation.deployment_metrics import latency_report
from vlaforge.validation.session_benchmark import compare_output_bytes, includes_host_io, validate_protocol, validate_rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_audited_summary(summary, recorded, process_means):
    expected = dict(summary)
    if 'process_mean_std_ns' in recorded:
        expected['process_mean_std_ns'] = statistics.stdev(process_means)
    require(expected == recorded, 'independent audit timing differs')


def summarize_campaign(entry):
    """Recheck raw storage and timing; remote model payloads remain separately audited."""
    prepared = Path(entry['prepared']).resolve()
    audit_path = Path(entry['formal_audit']['path']).resolve()
    require(sha(audit_path) == entry['formal_audit']['sha256'], 'formal audit digest differs')
    audit = read(audit_path)
    require(audit['status'] == 'passed' and audit['formal_experiment_complete'] is True,
            'formal audit is not complete')
    protocol = read(prepared / 'protocol.json')
    validate_protocol(protocol)
    require(protocol['schema'] == 'vlaforge.session_latency_protocol/2', 'complete-output protocol required')
    require(protocol['eager_validation'] == 'bitwise', 'this summary requires exact-output campaigns')
    require(protocol['processes'] >= 5 and protocol['measured'] >= 1000, 'formal repeat/sample threshold not met')
    require(audit['protocol_sha256'] == sha(prepared / 'protocol.json'), 'protocol audit binding differs')
    require(audit['frozen_files_sha256'] == sha(prepared / 'frozen-files.json'), 'frozen manifest differs')
    frozen = read(prepared / 'frozen-files.json')
    contracts = [Path(name) for name in frozen if name.endswith('/tensor-contract.json')]
    require(len(contracts) == 1, 'ambiguous frozen tensor contract')
    original_prepared = contracts[0].parent
    require(sha(prepared / 'tensor-contract.json') == frozen[str(contracts[0])], 'tensor contract changed')
    contract = read(prepared / 'tensor-contract.json')
    specs = contract['outputs']
    primary = next(item for item in specs if item['role'] == 'primary-action')
    audited = {(worker['repeat'], worker['policy']): worker for worker in audit['processes']}
    expected = {(repeat, policy) for repeat in range(protocol['processes']) for policy in protocol['policies']}
    require(set(audited) == expected and len(audited) == len(audit['processes']), 'formal worker coverage differs')
    inputs = {str(audit_path): sha(audit_path)}

    def bind(path, expected_digest=None):
        digest = sha(path)
        require(expected_digest is None or digest == expected_digest, 'evidence digest differs: ' + str(path))
        inputs[str(path)] = digest

    for name in ('protocol.json', 'frozen-files.json', 'tensor-contract.json', 'prepared.json', 'report.json'):
        bind(prepared / name)
    aggregate = read(prepared / 'report.json')
    require(aggregate['status'] == 'completed' and aggregate['protocol_sha256'] == audit['protocol_sha256'],
            'aggregate is incomplete or bound to another protocol')
    require(aggregate['boundary'] == protocol['boundary'], 'aggregate timing boundary differs')
    refs, metrics = {}, {}
    for index in range(len(protocol['samples'])):
        for spec in specs:
            values = []
            for kind in ('direct', 'eager'):
                path = prepared / 'data' / str(index) / spec[kind + '_file']
                original_path = original_prepared / path.relative_to(prepared)
                require(str(original_path) in frozen, 'reference lacks frozen binding')
                bind(path, frozen[str(original_path)])
                values.append(path.read_bytes())
            require(len(values[0]) == spec['size_bytes'] and values[0] == values[1], 'reference storage differs')
            refs[index, spec['name']] = values[0]
            metrics[index, spec['name']] = compare_output_bytes(values[0], values[0], values[1], spec)
    hardware, pids, results = None, set(), []
    for policy in protocol['policies']:
        samples, peak_memory, complete_tensors, worker_markers, replay = [], 0, 0, [], []
        process_means = []
        for repeat in range(protocol['processes']):
            folder = prepared / 'runs' / f'{repeat:02d}-{policy}'
            record = audited[repeat, policy]
            bind(folder / 'report.json', record['report_sha256'])
            report = read(folder / 'report.json')
            execution = read(folder / 'execution.json')
            require(report['status'] == 'passed' and execution['exit_code'] == 0, 'formal worker failed')
            require(report['pilot'] is False and execution['pilot'] is False, 'pilot cannot enter formal summary')
            require(report['repeat'] == repeat and report['policy'] == policy, 'worker identity differs')
            require(execution['pid'] == record['pid'] and execution['pid'] not in pids, 'independent worker PID differs')
            pids.add(execution['pid'])
            require(execution['protocol_sha256'] == audit['protocol_sha256'], 'worker protocol differs')
            require(execution['foreign_compute_owners'] is None and execution['monitor_error'] is None,
                    'worker ownership or monitoring failed')
            bind(folder / 'execution.json', record.get('execution_sha256'))
            bind(folder / 'samples.csv', report['samples_sha256'])
            bind(folder / 'process-maps.txt', report['process_maps_sha256'])
            maps = (folder / 'process-maps.txt').read_text().lower()
            require('libpython' not in maps and 'libtorch_python' not in maps, 'Python mapped in deployment worker')
            bind(folder / 'stderr.log')
            observed_replay = [line for line in (folder / 'stderr.log').read_text().splitlines()
                               if line.startswith('REPLAY_FINAL,')]
            require(observed_replay == record['replay_final'], 'audited replay counters differ')
            replay.append(observed_replay)
            worker_markers.append(verify_numerical_worker_execution(prepared, folder, policy)['configured'])
            with (folder / 'samples.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            validate_rows(rows, warmup=protocol['warmup'], measured=protocol['measured'], samples=len(protocol['samples']))
            require(report['boundary'] == protocol['boundary'], 'worker timing boundary differs')
            if includes_host_io(protocol):
                binding = report.get('host_timing_evidence')
                require(isinstance(binding, dict), 'host timing evidence binding missing')
                verify_host_timing_evidence(folder, protocol, rows, expected=binding)
                bind(folder / 'host-timing.csv', binding['raw_sha256'])
            require(record['calls'] == len(rows), 'audited complete call count differs')
            evidence = report['multi_output_evidence']
            bind(folder / 'multi-output-fidelity.json', evidence['fidelity_sha256'])
            for spec in specs:
                path = folder / spec['raw_file']
                bind(path, evidence['raw_sha256'][spec['name']])
                raw = path.read_bytes()
                size = spec['size_bytes']
                require(len(raw) == len(rows) * size, 'raw output missing or truncated')
                for index, row in enumerate(rows):
                    require(raw[index * size:(index + 1) * size] == refs[int(row['sample']), spec['name']],
                            'complete output differs from both references')
                complete_tensors += len(rows)
            steady = [{'index': index, 'latency_ns': int(row['latency_ns']), 'repeat_id': repeat}
                      for index, row in enumerate(rows[protocol['warmup']:])]
            require(latency_report(steady) == report['latency'], 'per-worker timing differs from raw samples')
            process_means.append(report['latency']['summary']['mean_ns'])
            offset = len(samples)
            samples.extend(dict(row, index=offset + index) for index, row in enumerate(steady))
            path = folder / 'telemetry.jsonl'
            bind(path, record.get('telemetry_sha256'))
            telemetry = path.read_text().splitlines()
            require(bool(telemetry), 'missing GPU telemetry')
            for line in telemetry:
                observed = json.loads(line)
                require(observed['exit_code'] == 0, 'GPU telemetry query failed')
                require({owner['pid'] for owner in observed['compute_owners']} <= {execution['monitored_nvml_pid']},
                        'foreign GPU owner observed')
                devices = list(csv.DictReader(io.StringIO(observed['stdout']), skipinitialspace=True))
                require(len(devices) == 1 and devices[0]['uuid'] == protocol['monitor_gpu'], 'GPU identity differs')
                device = devices[0]
                identity = {key: device[key] for key in ('name', 'uuid', 'driver_version')}
                require(hardware is None or hardware == identity, 'campaign mixes hardware identities')
                hardware = identity
                peak_memory = max(peak_memory, int(device['memory.used [MiB]'].split()[0]))
        combined = latency_report(samples)
        require(combined == aggregate['policies'][policy]['combined'], 'aggregate timing differs from raw samples')
        verify_audited_summary(combined['summary'], audit['summaries'][policy], process_means)
        path = prepared / (policy + '-cdf.csv')
        bind(path)
        with path.open() as stream:
            cdf = list(csv.DictReader(stream))
        require([int(row['latency_ns']) for row in cdf] == sorted(row['latency_ns'] for row in samples),
                'CDF differs from raw latency samples')
        require([float(row['cdf']) for row in cdf] == [(i + 1) / len(samples) for i in range(len(samples))], 'CDF ranks differ')
        action = [metrics[index, primary['name']]['complete_storage_metrics'] for index in range(len(protocol['samples']))]
        cosines = [item['cosine'] for item in action if item['cosine'] is not None]
        summary = combined['summary']
        results.append({'campaign': entry['id'], 'model': protocol['model'], 'hardware': hardware['name'],
            'gpu_uuid': hardware['uuid'], 'driver': hardware['driver_version'], 'policy': policy,
            'independent_processes': protocol['processes'], 'measured_calls': len(samples),
            'complete_output_tensors': complete_tensors, 'primary_dtype': primary['dtype'],
            'primary_shape': json.dumps(primary['shape']), 'primary_cosine_min': min(cosines) if cosines else None,
            'primary_mse_max': max(item['mse'] for item in action),
            'primary_max_abs': max(item['max_abs'] for item in action), 'complete_outputs_byte_exact': True,
            **{field + '_ms': summary[field + '_ns'] / 1e6 for field in ('mean', 'p50', 'p95', 'p99', 'max', 'std')},
            'fresh_chunks_per_second': summary['sequential_calls_per_second'],
            'peak_sampled_device_used_mib': peak_memory, 'numerical_worker_markers_observed': all(worker_markers),
            'boundary': protocol['boundary'], 'reference_limitations': json.dumps(protocol.get('reference_limitations', [])),
            'formal_audit_sha256': sha(audit_path), 'board_verified': False, 'full_paper_acceptance': False})
    return {'id': entry['id'], 'rows': results, 'inputs_sha256': inputs,
        'reference_limitations': protocol.get('reference_limitations', []),
        'scope': 'local full raw output and timing recheck; remote model payload and numerical providers use separate audits'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'output exists')
    manifest_digest = sha(args.manifest)
    manifest = read(args.manifest)
    require(manifest['schema'] == 'vlaforge.session_campaign_index/1', 'unsupported index schema')
    entries = manifest['campaigns']
    require(bool(entries) and len({item['id'] for item in entries}) == len(entries), 'empty or duplicate campaigns')
    campaigns = [summarize_campaign(entry) for entry in entries]
    for campaign in campaigns:
        for name, digest in campaign['inputs_sha256'].items():
            require(sha(Path(name)) == digest, 'input changed during summary')
    require(sha(args.manifest) == manifest_digest, 'index changed during summary')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [row for campaign in campaigns for row in campaign['rows']]
    with (args.output / 'cuda-session-table.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'schema': 'vlaforge.session_campaign_summary/1', 'status': 'rechecked-complete-formal-archives',
        'manifest_sha256': manifest_digest, 'script_sha256': sha(Path(__file__)), 'campaigns': campaigns,
        'outliers_removed': False, 'cross_campaign_speedup_computed': False,
        'memory_scope': 'sampled whole-worker NVML device used memory, not allocator peak',
        'not_a_complete_paper_main_table': True, 'board_verified': False, 'full_paper_acceptance': False,
        'remaining_main_table_metadata': ['checkpoint and complete model scale', 'locked model precision and scheduler profile',
            'complete-input boundary measurements', 'official and vendor baseline coverage', 'board measurements']}
    with (args.output / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': report['status'], 'campaigns': len(campaigns), 'rows': len(rows)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
