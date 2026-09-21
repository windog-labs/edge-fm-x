"""Recheck full tensors, device ownership and raw repeated operator timings."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

from benchmark_operator_examples import digest, output_identity


def require(value, message):
    if not value:
        raise ValueError(message)


def tensor_identity(path, key):
    import torch
    from torch.utils._pytree import tree_flatten

    payload = torch.load(path, map_location='cpu', weights_only=True)
    tensors, _ = tree_flatten(payload[key])
    require(bool(tensors) and all(torch.is_tensor(x) and bool(torch.isfinite(x).all()) for x in tensors),
            'invalid or nonfinite tensor output')
    return output_identity(payload[key])


def timing_summary(samples):
    require(bool(samples) and all(isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x) and x > 0 for x in samples), 'invalid measured timing samples')
    return {'median_ms': statistics.median(samples), 'mean_ms': statistics.mean(samples)}


def audit(root, *, verify_sources=False):
    bindings = {}

    def bind(path, expected=None):
        observed = digest(path)
        require(expected is None or observed == expected, 'file digest differs: ' + str(path))
        bindings[str(path)] = observed
        return observed

    def read(path, expected=None):
        bind(path, expected)
        def unique(pairs):
            result = {}
            for key, value in pairs:
                require(key not in result, 'duplicate JSON field')
                result[key] = value
            return result
        return json.loads(path.read_text(), object_pairs_hook=unique)

    campaign = read(root / 'campaign.json')
    require(campaign['schema'] == 'vlaforge.operator_repeat_campaign/2', 'registered campaign v2 required')
    require(campaign['status'] == 'passed', 'campaign did not pass')
    require(campaign['selected_for_deployment'] is False and campaign['end_to_end_integrated'] is False,
            'operator measurement cannot assert model integration')
    require(campaign['ownership_mode'] == 'register-reset-register-before-torch', 'ownership handshake missing')
    require(campaign['repeats'] >= 5, 'five independent pairs required')
    require(len(campaign['runs']) == 2 * campaign['repeats'], 'repeat coverage differs')
    if verify_sources:
        for filename, expected in campaign['source_files'].items():
            bind(Path(filename), expected)
    entries = [campaign['qualification'], *campaign['runs']]
    target_identity, context, pids, artifact = None, None, set(), None
    observed_hardware, previous_end = None, 0
    for entry in entries:
        require(previous_end <= entry['start_ns'] < entry['end_ns'], 'operator worker executions overlap')
        previous_end = entry['end_ns']
        command = entry['command']
        folder = root / Path(command[command.index('--output') + 1]).name
        require(folder.parent == root, 'output path escaped campaign')
        report = read(folder / 'report.json', entry['report_sha256'])
        if verify_sources:
            bind(Path(command[1]), campaign['tool_sha256'])
            bind(Path(command[command.index('--examples') + 1]), campaign['examples_sha256'])
        require(entry['exit_code'] == 0 and report['status'] == 'measured', 'worker failed')
        require(report['recipe'] == entry['recipe'], 'recipe changed')
        require(report['source_manifest_sha256'] == campaign['examples_sha256'], 'different input source')
        hardware = (report['gpu'], report['torch'], report['cuda'], report['compute_capability'])
        require(observed_hardware is None or hardware == observed_hardware, 'mixed hardware or runtime')
        observed_hardware = hardware
        require(context is None or context == report['target_numerical_context'], 'numerical policy changed')
        context = report['target_numerical_context']
        require(entry['preflight']['owners'] == [], 'nonempty preflight owners')
        require(entry['preflight']['device'].split(',')[0].strip() == campaign['gpu_uuid'], 'preflight UUID differs')
        monitor_folder = root / ('monitor-' + folder.name)
        monitor = read(monitor_folder / 'monitor.json', entry['monitor_sha256'])
        require(read(monitor_folder / 'preflight-owners.json') == [], 'monitor did not start idle')
        require(monitor['status'] == 'exited' and monitor['exitcode'] == 0, 'monitor failed')
        require(monitor['monitored_gpu'] == campaign['gpu_uuid'], 'monitor UUID differs')
        require(monitor['container_pid'] == entry['pid'] and monitor['nvml_pid'] == entry['nvml_pid'], 'PID binding differs')
        require(entry['pid'] not in pids, 'duplicate worker PID')
        pids.add(entry['pid'])
        stages = set()
        for event in monitor['observations']:
            require({o['pid'] for o in event['owners']} <= {monitor['nvml_pid']}, 'foreign GPU owner observed')
            if event.get('ready'):
                require(event['ready'] == {'pid': entry['pid'], 'ordinal': 0}, 'owner registration marker differs')
                if event['stage'] != 'reset' or event['owners'] == []:
                    stages.add(event['stage'])
        require({'registered-1', 'reset', 'registered-2'} <= stages, 'incomplete owner handshake')
        bind(folder / 'outputs.pt', report['outputs_sha256'])
        expected = tensor_identity(folder / 'outputs.pt', 'reference')
        actual = tensor_identity(folder / 'outputs.pt', 'candidate')
        require(expected == actual == report['target_reference_identity'], 'complete output tensor differs')
        require(target_identity is None or expected == target_identity, 'baseline and candidate references differ')
        target_identity = expected
        bind(folder / 'reference-comparison.pt', report['reference_comparison_sha256'])
        source = tensor_identity(folder / 'reference-comparison.pt', 'source_reference')
        local = tensor_identity(folder / 'reference-comparison.pt', 'target_reference')
        require(local == expected and (source == local) == report['source_reference_bitwise_equal'], 'source-reference comparison differs')
        measurement = report['measurement']
        recomputed = timing_summary(measurement['raw_gpu_batch_means_ms'])
        for key, value in recomputed.items():
            require(math.isclose(measurement[key], value, rel_tol=1e-12), 'timing summary differs')
        require(entry['measurement'] == measurement, 'parent and worker timings differ')
        if entry['recipe'] != 'aten':
            require(artifact is None or artifact == report['artifact_sha256'], 'candidate package changed between workers')
            artifact = report['artifact_sha256']
        bind(folder / 'benchmark-library-source.py', measurement['library_source_sha256'])
    recipes = ('aten', campaign['candidate_recipe'])
    summary = {}
    for recipe in recipes:
        entries = [e for e in campaign['runs'] if e['recipe'] == recipe]
        require(sorted(e['repeat'] for e in entries) == list(range(campaign['repeats'])), 'missing repetition')
        values = [e['measurement']['median_ms'] for e in entries]
        summary[recipe] = {'process_medians_ms': values, 'mean_process_median_ms': statistics.mean(values)}
    require(summary == campaign['summary'], 'aggregate differs from measured worker medians')
    for filename, expected in bindings.items():
        require(digest(Path(filename)) == expected, 'evidence changed during audit')
    return {'schema': 'vlaforge.operator_repeat_audit/2', 'status': 'passed', 'gpu_uuid': campaign['gpu_uuid'],
        'hardware': observed_hardware[0], 'summary': summary, 'repeated_processes': len(campaign['runs']),
        'whole_tensor_outputs_exact': True, 'source_files_verified_live': verify_sources,
        'candidate_artifact_sha256': artifact, 'inputs_sha256': bindings,
        'latency_reduction_percent': 100 * (1 - summary[recipes[1]]['mean_process_median_ms'] / summary['aten']['mean_process_median_ms']),
        'selected_for_deployment': False, 'end_to_end_integrated': False,
        'scope': 'graph-batch operator device time; qualification excluded; no integrated-model or per-call CDF claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-sources', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists(), 'audit output already exists')
    result = audit(args.campaign.resolve(), verify_sources=args.verify_sources)
    result['audit_source_sha256'] = digest(Path(__file__))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'inputs_sha256'}))
