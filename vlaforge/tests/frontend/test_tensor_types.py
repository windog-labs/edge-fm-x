"""Explicit dtype aliases do not silently change historical IR or tensor bytes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from vlaforge.frontend.tensor_types import (
    canonical_tensor_dtype,
    tensor_type_from_torch,
)
from vlaforge.ir.types import TensorType


@pytest.mark.parametrize("long,short", [
    ("bool", "bool"), ("int8", "i8"), ("uint8", "u8"), ("int16", "i16"), ("uint16", "u16"),
    ("int32", "i32"), ("uint32", "u32"), ("int64", "i64"), ("uint64", "u64"),
    ("float16", "f16"), ("bfloat16", "bf16"), ("float32", "f32"), ("float64", "f64"),
])
def test_supported_torch_and_storage_names(long, short):
    torch = pytest.importorskip("torch")
    assert canonical_tensor_dtype(long) == canonical_tensor_dtype("torch." + long) == short
    assert canonical_tensor_dtype(short) == short
    assert canonical_tensor_dtype(getattr(torch, long)) == short


@pytest.mark.parametrize("name", ["complex64", "complex128", "qint8", "quint8", "float8_e4m3fn", "float128", "", "numpy.float32"])
def test_unsupported_dtype_never_falls_back(name):
    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        canonical_tensor_dtype(name)


def test_import_and_string_conversion_do_not_import_torch():
    completed = subprocess.run([sys.executable, "-c", "import sys; from vlaforge.frontend.tensor_types import canonical_tensor_dtype; assert canonical_tensor_dtype('float32') == 'f32'; assert 'torch' not in sys.modules"],
        capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "python")})
    assert completed.returncode == 0, completed.stderr


def test_no_implicit_historical_type_migration():
    assert TensorType((1,), "float32").dtype == "float32"


def test_type_extraction_does_not_cast_copy_or_restride():
    torch = pytest.importorskip("torch")
    value = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    pointer, before = value.data_ptr(), value.clone()
    assert tensor_type_from_torch(value) == TensorType((3, 4), "f64")
    assert value.data_ptr() == pointer and torch.equal(value, before)
    with pytest.raises(ValueError, match="contiguous"):
        tensor_type_from_torch(value.t())


def test_arbitrary_object_is_not_stringified():
    with pytest.raises(TypeError, match="explicit dtype"):
        canonical_tensor_dtype(object())
