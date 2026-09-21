"""Explicit framework dtype conversion without changing historical IR parsing."""

from __future__ import annotations

from vlaforge.ir.types import TensorType

_ALIASES = {
    "bool": "bool", "i8": "i8", "int8": "i8", "char": "i8",
    "u8": "u8", "uint8": "u8", "byte": "u8",
    "i16": "i16", "int16": "i16", "short": "i16", "u16": "u16", "uint16": "u16",
    "i32": "i32", "int32": "i32", "int": "i32", "u32": "u32", "uint32": "u32",
    "i64": "i64", "int64": "i64", "long": "i64", "u64": "u64", "uint64": "u64",
    "f16": "f16", "float16": "f16", "half": "f16", "bf16": "bf16", "bfloat16": "bf16",
    "f32": "f32", "float32": "f32", "float": "f32", "f64": "f64", "float64": "f64", "double": "f64",
}


def canonical_tensor_dtype(dtype: object) -> str:
    """Map supported ordinary numeric aliases to IR names; never cast a tensor.

    This mapping describes IR storage types, not a backend capability claim.
    Quantized, complex, float8, opaque and unknown dtypes must use an explicit
    extension instead of silently falling back to a different representation.
    """
    if not isinstance(dtype, str):
        import torch

        if not isinstance(dtype, torch.dtype):
            raise TypeError("expected an explicit dtype string or torch.dtype")
        dtype = str(dtype)
    name = dtype.removeprefix("torch.")
    try:
        return _ALIASES[name]
    except KeyError as error:
        raise ValueError(f"unsupported tensor dtype: {dtype}") from error


def tensor_type_from_torch(tensor: object) -> TensorType:
    """Describe one static contiguous Torch tensor without moving or copying it."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise ValueError("static contiguous strided tensor required")
    if any(type(dimension) is not int for dimension in tensor.shape):
        raise ValueError("symbolic shape requires an explicit shape profile")
    return TensorType(tuple(tensor.shape), canonical_tensor_dtype(tensor.dtype))
