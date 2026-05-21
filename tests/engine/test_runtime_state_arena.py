import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.edge_fm_build_paths import prepend_built_python_paths

prepend_built_python_paths(project_root)

import edge_fm


def test_runtime_state_arena_reuses_named_buffer_with_stable_address():
    arena = edge_fm.RuntimeStateArena()

    state = arena.get_or_create(
        "layer.0.conv_state",
        [2, 3, 4],
        edge_fm.DType.Float32,
        edge_fm.Device.CPU,
    )
    first_ptr = state.data_ptr()

    same_state = arena.get_or_create(
        "layer.0.conv_state",
        [2, 3, 4],
        edge_fm.DType.Float32,
        edge_fm.Device.CPU,
    )

    assert same_state.data_ptr() == first_ptr
    assert list(same_state.shape()) == [2, 3, 4]
    assert same_state.dtype() == edge_fm.DType.Float32
    assert same_state.device()[0] == edge_fm.Device.CPU
    assert arena.contains("layer.0.conv_state")


def test_runtime_state_arena_creates_shape_view_without_reallocating():
    arena = edge_fm.RuntimeStateArena()
    state = arena.get_or_create(
        "layer.4.recurrent_state",
        [2, 4],
        edge_fm.DType.Float16,
        edge_fm.Device.CPU,
    )

    flat_view = arena.view("layer.4.recurrent_state", [8])

    assert flat_view.data_ptr() == state.data_ptr()
    assert list(flat_view.shape()) == [8]
    assert flat_view.dtype() == edge_fm.DType.Float16


def test_runtime_state_arena_rejects_incompatible_reuse():
    arena = edge_fm.RuntimeStateArena()
    arena.get_or_create("state", [4], edge_fm.DType.Int32, edge_fm.Device.CPU)

    with pytest.raises(edge_fm.InvalidRequestError):
        arena.get_or_create("state", [5], edge_fm.DType.Int32, edge_fm.Device.CPU)

    with pytest.raises(edge_fm.InvalidRequestError):
        arena.get_or_create("state", [4], edge_fm.DType.Float32, edge_fm.Device.CPU)


def test_runtime_state_arena_isolated_per_request():
    request_a = edge_fm.RuntimeStateArena()
    request_b = edge_fm.RuntimeStateArena()

    a_state = request_a.get_or_create("shared.name", [16], edge_fm.DType.Int32, edge_fm.Device.CPU)
    b_state = request_b.get_or_create("shared.name", [16], edge_fm.DType.Int32, edge_fm.Device.CPU)

    assert a_state.data_ptr() != b_state.data_ptr()
    assert request_a.names() == ["shared.name"]
    assert request_b.names() == ["shared.name"]
