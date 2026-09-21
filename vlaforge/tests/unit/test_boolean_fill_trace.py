import pytest
import os
from pathlib import Path
import subprocess


@pytest.mark.parametrize('fill', [False, True])
@pytest.mark.parametrize('dtype', [None, 'bool', 'int64', 'float32'])
def test_boolean_fill_trace_preserves_inferred_and_explicit_dtype(tmp_path, fill, dtype):
    torch = pytest.importorskip('torch')
    from vlaforge.deployment.torchscript_export import trace_boolean_fill_compatible

    class Module(torch.nn.Module):
        def forward(self, value):
            return torch.full((value.shape[0],), fill, dtype=getattr(torch, dtype) if dtype else None, device=value.device)

    example = (torch.ones(3),)
    traced, ledger = trace_boolean_fill_compatible(Module(), example, strict=True, check_trace=True)
    path = tmp_path / 'candidate.pt'
    traced.save(str(path))
    actual = torch.jit.load(str(path))(*example)
    expected = Module()(*example)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert ledger['normalized_boolean_fills'] > 0
    assert ledger['numerical_certificate'] is False


def test_non_boolean_factory_calls_are_unchanged():
    torch = pytest.importorskip('torch')
    from vlaforge.deployment.torchscript_export import trace_boolean_fill_compatible

    class Module(torch.nn.Module):
        def forward(self, value):
            return torch.full((value.shape[0],), 1.25, device=value.device)

    traced, ledger = trace_boolean_fill_compatible(Module(), (torch.ones(3),))
    assert ledger['normalized_boolean_fills'] == 0
    assert torch.equal(traced(torch.ones(3)), torch.full((3,), 1.25))


@pytest.mark.skipif(os.getenv('VLAFORGE_RUN_CPP_BOOL_TRACE') != '1', reason='opt-in real LibTorch C++ artifact consumer')
def test_cpp_consumer_executes_boolean_factory_trace(tmp_path):
    torch = pytest.importorskip('torch')
    from vlaforge.deployment.torchscript_export import trace_boolean_fill_compatible

    class Module(torch.nn.Module):
        def forward(self, value):
            return (torch.full((value.shape[0],), False, dtype=torch.bool),
                    torch.full((value.shape[0],), True, dtype=torch.bool))

    traced, _ = trace_boolean_fill_compatible(Module(), (torch.ones(3),), strict=True)
    artifact = tmp_path / 'boolean.pt'
    traced.save(str(artifact))
    root = Path(torch.__file__).parent
    binary = tmp_path / 'consumer'
    source = Path(__file__).parents[1] / 'cpp/boolean_fill_trace_smoke.cpp'
    command = ['c++', '-std=c++17', '-O1', str(source), '-o', str(binary),
        '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        '-I' + str(root / 'include'), '-I' + str(root / 'include/torch/csrc/api/include'),
        '-L' + str(root / 'lib'), '-Wl,-rpath,' + str(root / 'lib'), '-ltorch', '-ltorch_cpu', '-lc10']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    linked = subprocess.check_output(['ldd', str(binary)], text=True)
    assert 'libpython' not in linked and 'libtorch_python' not in linked and 'not found' not in linked
    subprocess.run([str(binary), str(artifact)], check=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONHOME': '/invalid', 'PYTHONPATH': '/invalid'})
