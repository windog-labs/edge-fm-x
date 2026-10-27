from dataclasses import replace
import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

from vlaforge.adapters import build_driving_trajectory_fixture
from vlaforge.codegen import CppArtifactRegionDefinition, CppValidatorDefinition, generate_compiled_cpp_session
from vlaforge.compiler import compile_module


def compilation_and_definitions():
    compilation = compile_module(build_driving_trajectory_fixture().module, profile='off', default_device='cuda:0')
    definitions = {region.name: CppArtifactRegionDefinition(region_name=region.name, backend='tensorrt',
        artifact_path='artifacts/%s.engine' % region.name, artifact_sha256='1' * 64, artifact_size_bytes=123,
        io_schema_digest=compilation.plan.io_schema_digest, target='sm_87', device='cuda:0',
        backend_variant='tensorrt-8.5.2.2', residency='session') for region in compilation.module.regions}
    return compilation, definitions


def generate(compilation, definitions, **kwargs):
    return generate_compiled_cpp_session(compilation, artifact_regions=definitions,
        validators={'finite_trajectory': CppValidatorDefinition('finite_trajectory', 'return data != nullptr && size_bytes > 0u;')},
        **kwargs).as_dict()


def test_region_completion_is_opt_in_and_receives_borrowed_outputs_after_sync():
    compilation, definitions = compilation_and_definitions()
    original = generate(compilation, definitions)
    explicit_off = generate(compilation, definitions, observe_region_completion=False)
    assert original == explicit_off
    assert 'SetRegionCompletionSink' not in original['session_generated.h']
    observed = generate(compilation, definitions, observe_region_completion=True)
    assert 'SetRegionCompletionSink' in observed['session_generated.h']
    source = observed['session_generated.cpp']
    assert 'RegionCompletionSink' in observed['session_generated.h']
    assert 'completion_outputs' in source
    assert '&region_' in source
    position = source.index('region_completion_.emit(')
    assert source.rfind('api->synchronize(executable)', 0, position) >= 0
    assert source.rfind('if (!status.ok())', 0, position) > source.rfind('api->synchronize(executable)', 0, position)
    assert 'if (region_completion_.emit != nullptr)' in source


def test_region_completion_refuses_unsynchronized_execution_context_claims():
    compilation, definitions = compilation_and_definitions()
    key = next(iter(definitions))
    definitions[key] = replace(definitions[key], supports_execution_context=True)
    with pytest.raises(ValueError, match='synchronous artifact'):
        generate(compilation, definitions, observe_region_completion=True)


@pytest.mark.parametrize('value', [None, 1, 'on'])
def test_region_completion_requires_explicit_boolean(value):
    compilation, definitions = compilation_and_definitions()
    with pytest.raises(ValueError, match='boolean'):
        generate(compilation, definitions, observe_region_completion=value)


def test_actual_completion_views_are_read_after_sync_and_suppressed_on_sync_failure(tmp_path):
    spec = importlib.util.spec_from_file_location('workspace_completion_fixture',
        Path(__file__).with_name('test_artifact_workspace_codegen.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    runtime = Path(__file__).resolve().parents[2]
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    plugin = bundle / 'nonmodel-workspace.so'
    source = tmp_path / 'provider.cpp'
    source.write_text(fixture._PLUGIN.replace('VLAForgeStatus Sync(VLAForgeRegionExecutable*) { return Ok(); }',
        'VLAForgeStatus Sync(VLAForgeRegionExecutable* e) { return Mode("sync-fail") && e->id == 1 ? Error() : Ok(); }'))
    fixture._run(['c++', '-std=c++17', '-O2', '-shared', '-fPIC', '-I' + str(runtime / 'include'),
        str(source), '-o', str(plugin)], tmp_path, 'provider-build')
    program = fixture._program()
    compilation = fixture._compile(program)
    definitions = {region.name: CppArtifactRegionDefinition(region_name=region.name, backend='shared_plugin',
        artifact_path='nonmodel-workspace.so', artifact_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
        artifact_size_bytes=plugin.stat().st_size, io_schema_digest=compilation.plan.io_schema_digest,
        target='cpu', device='cpu', backend_variant='shared-plugin/1') for region in compilation.module.regions}
    callback = r'''
struct CompletionProbe { unsigned calls = 0; bool valid = true; };
void Completed(void* context, const vlaforge::runtime::TraceEvent* event,
    const VLAForgeValueView* const* outputs, std::size_t count) noexcept {
  auto& p = *static_cast<CompletionProbe*>(context);
  const float expected[] = {2, 4, 5, 9, 11, 12, 16, 18, 19};
  if (p.calls >= 9 || event == nullptr || event->kind != vlaforge::runtime::TraceKind::kRegion ||
      event->subject_id != (p.calls % 3 == 1 ? 1u : 0u) || count != 1 || outputs == nullptr ||
      outputs[0] == nullptr || outputs[0]->kind != VLAFORGE_VALUE_TENSOR) { p.valid = false; return; }
  const auto& t = outputs[0]->value.tensor.tensor;
  if (t.dtype != VLAFORGE_DTYPE_F32 || t.device.kind != VLAFORGE_DEVICE_CPU || t.size_bytes != 8 ||
      t.data == nullptr || static_cast<const float*>(t.data)[0] != expected[p.calls]) p.valid = false;
  ++p.calls;
}
'''
    runner = fixture._RUNNER.replace('int main(', callback + '\nint main(')
    runner = runner.replace('vlaforge_generated::ModelSession session(argv[1]);',
        'CompletionProbe probe;\n  vlaforge_generated::ModelSession session(argv[1]);\n'
        '  session.SetRegionCompletionSink({&probe, &Completed});')
    runner = runner.replace('std::printf("NONMODEL_RUN_REJECTION',
        'if (mode != nullptr && std::strcmp(mode, "sync-fail") == 0 && (probe.calls != 1 || !probe.valid)) return 8;\n'
        '      std::printf("NONMODEL_RUN_REJECTION')
    runner = runner.replace('std::puts("NONMODEL_WORKSPACE_CPU_PASS',
        'if (probe.calls != 9 || !probe.valid) return 9;\n  std::puts("NONMODEL_WORKSPACE_CPU_PASS')
    generated = tmp_path / 'generated'
    generate_compiled_cpp_session(compilation, artifact_regions=definitions, validators=program.cpp_validators(),
        runner_source=runner, observe_region_completion=True).write(generated)
    fixture._run(['cmake', '-S', str(generated), '-B', str(tmp_path / 'build'),
        '-DVLAFORGE_RUNTIME_ROOT=' + str(runtime), '-DBUILD_TESTING=OFF', '-DCMAKE_BUILD_TYPE=Release'], tmp_path, 'configure')
    fixture._run(['cmake', '--build', str(tmp_path / 'build'), '--parallel', '2'], tmp_path, 'build')
    for mode in ('normal', 'sync-fail'):
        result = fixture._run([str(tmp_path / 'build/vlaforge_generated_runner'), str(bundle)], tmp_path, mode,
            environment={**os.environ, 'NONMODEL_WORKSPACE_MODE': mode, 'CUDA_VISIBLE_DEVICES': ''})
        assert ('NONMODEL_WORKSPACE_CPU_PASS' if mode == 'normal' else 'NONMODEL_RUN_REJECTION') in result.stdout
