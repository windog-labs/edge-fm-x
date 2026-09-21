"""Unit coverage for generic SmolVLA HMCT recipe overrides."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[2] / "tools/quantize_smolvla_step_hmct.py"
_SPEC = importlib.util.spec_from_file_location("quantize_smolvla_step_hmct", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def test_parse_operator_qtype_strips_qualifier() -> None:
    assert tool.parse_op_qtype(" MatMul = int16 ") == ("MatMul", "int16")


@pytest.mark.parametrize("value", ["MatMul", "=int16", "MatMul="])
def test_parse_operator_qtype_rejects_malformed_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="OP=TYPE"):
        tool.parse_op_qtype(value)


def test_apply_operator_qtypes_preserves_recipe_config() -> None:
    config = {
        "model_config": {
            "all_node_type": "float16",
            "model_output_type": "float32",
        }
    }

    applied = tool.apply_op_qtypes(
        config, [("Conv", "int16"), ("MatMul", "int16")]
    )

    assert config == {
        "model_config": {
            "all_node_type": "float16",
            "model_output_type": "float32",
        },
        "op_config": {
            "Conv": {"qtype": "int16"},
            "MatMul": {"qtype": "int16"},
        },
    }
    assert applied == {
        "Conv": {"qtype": "int16"},
        "MatMul": {"qtype": "int16"},
    }
