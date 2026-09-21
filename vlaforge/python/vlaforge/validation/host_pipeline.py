"""Complete-input host timing and full-output storage, independent of model family.

A pipeline implements prepare(sample) and infer(prepared). infer must return
every declared output as a contiguous CPU Torch tensor. The caller owns loading,
device/process registration, numerical policy and pipeline lifetime. Validation,
raw output logging and diagnostics occur after the end timestamp.
"""

from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import time

from vlaforge.validation.deployment_metrics import latency_report
from vlaforge.validation.session_benchmark import compare_output_bytes, tensor_bytes


SEGMENTS = ('preprocess_h2d_ns', 'inference_postprocess_d2h_ns')
TORCH_DTYPES = {'bool': 'bool', 'i32': 'int32', 'i64': 'int64', 'u8': 'uint8', 'u64': 'uint64',
                'f16': 'float16', 'bf16': 'bfloat16', 'f32': 'float32', 'f64': 'float64'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_timing(rows, *, warmup, measured, sample_count):
    if not sample_count or len(rows) != warmup + measured:
        raise ValueError('incomplete pipeline timing or sample coverage')
    expected = {'run', 'sample', 'measured', 'latency_ns', *SEGMENTS}
    for index, row in enumerate(rows):
        if (set(row) != expected or any(type(value) not in (int, str) or not str(value).isdecimal()
                                       for value in row.values())):
            raise ValueError('pipeline timing requires exact integer columns')
        values = {key: int(value) for key, value in row.items()}
        if (values['run'] != index or values['sample'] != index % sample_count
                or values['measured'] != int(index >= warmup) or values['latency_ns'] <= 0
                or any(values[key] <= 0 for key in SEGMENTS)
                or sum(values[key] for key in SEGMENTS) != values['latency_ns']):
            raise ValueError('pipeline timing order or full-call accounting differs')


def complete_cpu_bytes(outputs, specs):
    import torch

    if set(outputs) != {spec['name'] for spec in specs}:
        raise ValueError('pipeline must return every declared output')
    result = {}
    for spec in specs:
        value = outputs[spec['name']]
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or tuple(value.shape) != tuple(spec['shape'])
                or value.dtype != getattr(torch, TORCH_DTYPES[spec['dtype']]) or not value.is_contiguous()):
            raise ValueError('pipeline output must already be a complete contiguous CPU tensor: ' + spec['name'])
        result[spec['name']] = value.detach().reshape(-1).view(torch.uint8).numpy().tobytes()
    return result


def measure_pipeline(pipeline, samples, specs, references, output, *, warmup, measured,
                     synchronize, checkpoint=lambda: None, clock=time.perf_counter_ns):
    """Measure explicit preparation and full inference/output completion segments.

Samples and reference bytes are loaded before entry. Initialization and input
file/video decoding are excluded. These two segments do not claim separate CPU
preprocessing/H2D or inference/postprocessing/D2H attribution. The reference map
has the same eager/direct complete typed-output contract as the Session runner.
"""
    if (type(warmup) is not int or warmup < 0 or type(measured) is not int or measured < 1
            or not samples or len(references) != len(samples) or not specs
            or len({spec['name'] for spec in specs}) != len(specs)):
        raise ValueError('invalid pipeline sample, repeat or output contract')
    names = [spec['raw_file'] for spec in specs]
    if len(set(names)) != len(names) or any(Path(name).name != name or name in ('', '.', '..', 'samples.csv') for name in names):
        raise ValueError('output storage names must be unique flat filenames')
    for reference in references:
        if set(reference) != {spec['name'] for spec in specs}:
            raise ValueError('reference output set differs')
        for spec in specs:
            value = reference[spec['name']]
            if set(value) != {'eager', 'direct'} or any(
                not isinstance(raw, bytes) or len(raw) != tensor_bytes(spec['dtype'], spec['shape']) for raw in value.values()
            ):
                raise ValueError('complete reference storage differs')
            if not compare_output_bytes(value['direct'], value['direct'], value['eager'], spec)['eager_bitwise_equal']:
                raise ValueError('this lossless pipeline benchmark requires byte-exact references')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    timings, fidelity = [], []
    with ExitStack() as stack:
        raw = {spec['name']: stack.enter_context((output / spec['raw_file']).open('xb')) for spec in specs}
        stream = stack.enter_context((output / 'samples.csv').open('x'))
        writer = csv.DictWriter(stream, fieldnames=['run', 'sample', 'measured', 'latency_ns', *SEGMENTS])
        writer.writeheader()
        for index in range(warmup + measured):
            sample_id = index % len(samples)
            checkpoint()
            synchronize()
            start = clock()
            prepared = pipeline.prepare(samples[sample_id])
            synchronize()
            split = clock()
            outputs = pipeline.infer(prepared)
            synchronize()
            end = clock()
            checkpoint()
            row = {'run': index, 'sample': sample_id, 'measured': int(index >= warmup),
                'latency_ns': end - start, SEGMENTS[0]: split - start, SEGMENTS[1]: end - split}
            actual = complete_cpu_bytes(outputs, specs)
            metrics = {}
            for spec in specs:
                reference = references[sample_id][spec['name']]
                value = actual[spec['name']]
                raw[spec['name']].write(value)
                metric = compare_output_bytes(value, reference['direct'], reference['eager'], spec)
                if not metric['eager_bitwise_equal']:
                    raise ValueError('complete pipeline output differs from eager reference')
                metrics[spec['name']] = metric
            writer.writerow(row)
            stream.flush()
            timings.append(row)
            fidelity.append({'run': index, 'sample': sample_id, 'outputs': metrics})
    validate_timing(timings, warmup=warmup, measured=measured, sample_count=len(samples))
    latency = latency_report([{'index': index, 'latency_ns': row['latency_ns'], 'repeat_id': 0}
                              for index, row in enumerate(timings[warmup:])])
    with (output / 'output-fidelity.json').open('x') as stream:
        json.dump(fidelity, stream, indent=2, allow_nan=False)
    return {'status': 'measured_and_checked', 'warmup': warmup, 'measured': measured, 'latency': latency,
            'complete_output_tensors': (warmup + measured) * len(specs), 'all_byte_exact': True,
            'files_sha256': {path.name: digest(path) for path in output.iterdir()},
            'independent_audit_complete': False}
