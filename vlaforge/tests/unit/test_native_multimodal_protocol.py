"""CPU-only contract tests for native multimodal benchmark protocols."""

import importlib.util
from pathlib import Path

import numpy as np


_PATH = Path(__file__).resolve().parents[2] / "tools/build_native_multimodal_protocol.py"
_SPEC = importlib.util.spec_from_file_location("native_multimodal_protocol", _PATH)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def test_bf16_decode_preserves_bit_pattern(tmp_path):
    path = tmp_path / "values.bin"
    np.array([0x3F80, 0xBF80, 0x0000], dtype="<u2").tofile(path)
    values = tool.decode(path, "bf16", [3])
    assert values.dtype == np.float32
    assert values.tolist() == [1.0, -1.0, 0.0]
    assert values.view(np.uint32).tolist() == [0x3F800000, 0xBF800000, 0]


def test_protocol_supported_dtypes_are_bounded():
    assert set(tool.DTYPES) == {
        "f32", "f64", "f16", "bf16", "i64", "i32", "u64", "u8", "bool"
    }
