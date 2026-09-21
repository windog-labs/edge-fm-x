"""CPU-only contract checks for the generic native rejection probe."""

import importlib.util
from pathlib import Path


_PATH = Path(__file__).resolve().parents[2] / "tools/probe_native_session_contract.py"
_SPEC = importlib.util.spec_from_file_location("native_contract_probe", _PATH)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def test_dtype_code_contract_is_explicit():
    assert tool.DTYPE_CODES == {
        "bool": 1,
        "i32": 2,
        "i64": 3,
        "f16": 4,
        "bf16": 5,
        "f32": 6,
        "f64": 7,
        "u64": 8,
        "u8": 9,
    }


def test_tensor_byte_count_uses_schema_shape_and_dtype():
    assert tool.tensor_bytes(
        {"payload": {"dtype": "i64", "shape": [1, 327]}}
    ) == 1 * 327 * 8
    assert tool.tensor_bytes(
        {"payload": {"dtype": "f32", "shape": [1200, 1536]}}
    ) == 1200 * 1536 * 4


def test_sample_input_hashes_bind_content_and_input_ids(tmp_path):
    (tmp_path / "0.bin").write_bytes(b"primary")
    schema = {
        "inputs": [
            {
                "input_id": 0,
                "name": "input_ids",
                "payload": {"kind": "tensor"},
            }
        ]
    }
    first = tool.sample_input_hashes(tmp_path, schema)

    (tmp_path / "0.bin").write_bytes(b"alternate")
    second = tool.sample_input_hashes(tmp_path, schema)

    assert set(first) == {"input_ids"}
    assert first != second
