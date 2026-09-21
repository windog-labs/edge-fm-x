"""Real generated CPU shared library calls; no fake Session/CUDA implementation."""

import ctypes as C
from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from vlaforge.codegen import CppRegionDefinition, CppValidatorDefinition, generate_cpp_session
from vlaforge.compiler import compile_module
from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType
from vlaforge.validation import native_session as native


@pytest.fixture(scope='module')
def compiled(tmp_path_factory):
    root = tmp_path_factory.mktemp('native-host-cpu')
    first, second, output_type = TensorType((2, 3), 'f32'), TensorType((1,), 'i64'), TensorType((2, 3), 'f64')
    builder = InvocationBuilder('host_fixture', inputs=(
        InputPort('image', first), InputPort('counter', second),
        InputPort('accepted', TensorType((1,), 'bool'))), outputs=(
        OutputPort('actions', output_type, group='output'),
        OutputPort('receipt', second, group='output')))

    @tensor_region('calculate', inputs=(Value('x', first), Value('count', second)), outputs=(output_type, second))
    def calculate(*_):
        raise AssertionError('declaration-only fixture')

    actions, receipt = builder.call(calculate, builder.input('image'), builder.input('counter'))
    module = builder.finish({'actions': actions, 'receipt': receipt}, accepted=builder.input('accepted')).module
    compilation = compile_module(module)
    module = compilation.module
    validators = {'vlaforge_predicate_true': CppValidatorDefinition('vlaforge_predicate_true',
        'return data != nullptr && size_bytes == 1u && *static_cast<const bool*>(data);')}
    sources = generate_cpp_session(compilation.plan, module, validators=validators, regions={
        'calculate': CppRegionDefinition('calculate', '''
const auto* input = Input<float>(executable, 0u);
const auto* counter = Input<std::int64_t>(executable, 1u);
auto* actions = Output<double>(executable, 0u);
for (std::size_t index = 0; index < 6u; ++index) actions[index] = static_cast<double>(input[index]) * 2.0;
Output<std::int64_t>(executable, 1u)[0] = counter[0] + 1;
return vlaforge_status_ok();
''')})
    sources.write(root / 'generated')
    path = root / 'generated/session_generated.cpp'
    path.write_text(path.read_text() + '''
#include <cstddef>
#include "vlaforge/runtime/bounded_replay.h"
extern "C" void fixture_abi(std::size_t* out) {
  out[0] = sizeof(VLAForgeStatus); out[1] = sizeof(VLAForgeTensorView);
  out[2] = sizeof(VLAForgeBoundTensor); out[3] = sizeof(VLAForgeInputStamp);
  out[4] = sizeof(VLAForgeSessionApi);
  out[5] = offsetof(VLAForgeSessionApi, run);
  out[6] = offsetof(VLAForgeBoundTensor, tensor);
  out[7] = offsetof(VLAForgeTensorView, device);
  out[8] = offsetof(VLAForgeInputStamp, revision);
  out[9] = sizeof(VLAForgeBoundedReplayInfo);
  out[10] = offsetof(VLAForgeBoundedReplayInfo, replay_count);
  out[11] = offsetof(VLAForgeBoundedReplayInfo, reason);
}
''')
    runtime = Path(__file__).resolve().parents[2]
    subprocess.run(['cmake', '-S', str(root / 'generated'), '-B', str(root / 'build'),
        '-DVLAFORGE_RUNTIME_ROOT=' + str(runtime), '-DVLAFORGE_GENERATED_SHARED=ON',
        '-DBUILD_TESTING=OFF', '-DCMAKE_BUILD_TYPE=Release'], check=True, capture_output=True)
    subprocess.run(['cmake', '--build', str(root / 'build'), '--parallel', '2'], check=True, capture_output=True)
    library = root / 'build/libvlaforge_generated_session.so'
    assert 'libpython' not in subprocess.check_output(['ldd', str(library)], text=True).lower()
    return library, module, hashlib.sha256(library.read_bytes()).hexdigest()


def inputs():
    return {'image': torch.arange(6, dtype=torch.float32).reshape(2, 3),
            'counter': torch.tensor([2**60 + 1], dtype=torch.int64),
            'accepted': torch.tensor([True])}


def test_ffi_layout_matches_actual_cpp_headers(compiled):
    library, _, _ = compiled
    probe = C.CDLL(str(library)).fixture_abi
    probe.argtypes, probe.restype = [C.POINTER(C.c_size_t)], None
    values = (C.c_size_t * 12)()
    probe(values)
    assert list(values) == [C.sizeof(native._Status), C.sizeof(native._Tensor), C.sizeof(native._BoundTensor),
        C.sizeof(native._Stamp), C.sizeof(native._Api), native._Api.run.offset,
        native._BoundTensor.tensor.offset, native._Tensor.device.offset, native._Stamp.revision.offset,
        C.sizeof(native._ReplayInfo), native._ReplayInfo.replay_count.offset, native._ReplayInfo.reason.offset]


def test_complete_mixed_outputs_and_repeated_calls_keep_owned_storage(compiled):
    library, module, digest = compiled
    with native.NativeTensorSession(library, module, library_sha256=digest) as session:
        source = inputs()
        first = session.run(source)
        assert set(first) == {'actions', 'receipt'}
        assert first['actions'].dtype == torch.float64 and first['receipt'].dtype == torch.int64
        assert torch.equal(first['actions'], source['image'].double() * 2)
        assert first['receipt'].item() == 2**60 + 2
        source['image'].add_(10)
        second = session.run(source)
        assert torch.equal(second['actions'], source['image'].double() * 2)
        assert torch.equal(first['actions'], torch.arange(6).reshape(2, 3).double() * 2)
    session.close()
    with pytest.raises(RuntimeError, match='closed'):
        session.run(inputs())


@pytest.mark.parametrize('bad', ['shape', 'dtype', 'noncontiguous', 'missing', 'extra'])
def test_bad_input_is_rejected_before_native_run(compiled, bad):
    library, module, digest = compiled
    with native.NativeTensorSession(library, module, library_sha256=digest) as session:
        source = inputs()
        if bad == 'shape': source['image'] = torch.zeros(6)
        if bad == 'dtype': source['image'] = source['image'].double()
        if bad == 'noncontiguous': source['image'] = torch.zeros(3, 2).T
        if bad == 'missing': source.pop('accepted')
        if bad == 'extra': source['unexpected'] = torch.ones(1)
        with pytest.raises(ValueError, match='input'):
            session.run(source)
        assert session.run(inputs())['receipt'].item() == 2**60 + 2


def test_library_and_schema_identity_cannot_be_substituted(compiled):
    library, module, digest = compiled
    with pytest.raises(ValueError, match='digest'):
        native.NativeTensorSession(library, module, library_sha256='0' * 64)
    changed = replace(module, outputs=(replace(module.outputs[0], name='changed'), *module.outputs[1:]))
    with pytest.raises(ValueError, match='schema'):
        native.NativeTensorSession(library, changed, library_sha256=digest)


def test_owner_thread_and_native_rejection_are_enforced(compiled):
    library, module, digest = compiled
    with native.NativeTensorSession(library, module, library_sha256=digest) as session:
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(RuntimeError, match='process/thread'):
                pool.submit(session.run, inputs()).result()
        rejected = inputs()
        rejected['accepted'].zero_()
        with pytest.raises(RuntimeError, match='Session'):
            session.run(rejected)


def test_replay_queries_reject_invalid_ids_and_unsupported_cpu_session(compiled):
    library, module, digest = compiled
    with native.NativeTensorSession(library, module, library_sha256=digest) as session:
        for value in (-1, 2**32, True, '1'):
            with pytest.raises(ValueError, match='uint32'):
                session.replay_info(value)
        with pytest.raises(RuntimeError, match='replay'):
            session.replay_info(0)
    with pytest.raises(RuntimeError, match='closed'):
        session.replay_info(0)


def test_host_pipeline_keeps_full_cpp_outputs_and_cycles_raw_samples(compiled, tmp_path):
    import csv
    from vlaforge.validation.host_pipeline import measure_pipeline, validate_timing

    library, module, digest = compiled
    specs = [{'name': port.name, 'shape': list(port.payload.shape), 'dtype': port.payload.dtype,
              'role': 'exact' if port.payload.dtype == 'i64' else 'float',
              'raw_file': port.name + '.bin'} for port in module.outputs]
    samples = [inputs(), inputs()]
    samples[1]['image'].add_(7)
    references = []
    for sample in samples:
        values = {'actions': sample['image'].double() * 2, 'receipt': sample['counter'] + 1}
        references.append({name: dict(eager=value.view(torch.uint8).numpy().tobytes(),
                                      direct=value.view(torch.uint8).numpy().tobytes()) for name, value in values.items()})
    with native.NativeTensorSession(library, module, library_sha256=digest) as session:
        class Pipeline:
            def prepare(self, sample):
                return {name: value.clone() for name, value in sample.items()}

            def infer(self, prepared):
                return session.run(prepared)

        report = measure_pipeline(Pipeline(), samples, specs, references, tmp_path / 'measurement',
                                  warmup=3, measured=5, synchronize=lambda: None)
    assert report['complete_output_tensors'] == 16 and report['all_byte_exact']
    assert report['latency']['summary']['count'] == 5
    with (tmp_path / 'measurement/samples.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    validate_timing(rows, warmup=3, measured=5, sample_count=2)
    for spec in specs:
        expected = b''.join(references[index % 2][spec['name']]['eager'] for index in range(8))
        assert (tmp_path / 'measurement' / spec['raw_file']).read_bytes() == expected
    rows[-1]['preprocess_h2d_ns'] = str(int(rows[-1]['preprocess_h2d_ns']) + 1)
    with pytest.raises(ValueError, match='accounting'):
        validate_timing(rows, warmup=3, measured=5, sample_count=2)


@pytest.mark.parametrize('bad', ['missing', 'dtype', 'shape', 'noncontiguous'])
def test_pipeline_cannot_silently_convert_incomplete_output(bad):
    from vlaforge.validation.host_pipeline import complete_cpu_bytes

    specs = [{'name': 'actions', 'dtype': 'f64', 'shape': [2, 3]}]
    values = {'actions': torch.ones((2, 3), dtype=torch.float64)}
    if bad == 'missing': values = {}
    if bad == 'dtype': values['actions'] = values['actions'].float()
    if bad == 'shape': values['actions'] = values['actions'].reshape(6)
    if bad == 'noncontiguous': values['actions'] = torch.ones((3, 2), dtype=torch.float64).T
    with pytest.raises(ValueError, match='output'):
        complete_cpu_bytes(values, specs)
