"""Regression coverage for the exact Qwen3.5 TorchScript bundle ABI."""

import importlib.util
from pathlib import Path


_PATH = Path(__file__).resolve().parents[2] / "tools/build_qwen35_exact_bundle.py"
_SPEC = importlib.util.spec_from_file_location("qwen35_bundle_tool", _PATH)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def test_saved_artifact_and_region_contract_use_same_output_order():
    module, region = tool.build_module(
        prompt_length=327,
        new_tokens=16,
        pixel_shape=(1200, 1536),
        vocab_size=248320,
        device="cuda:0",
    )
    contracts = tool.build_region_output_contracts(
        new_tokens=16,
        vocab_size=248320,
        device="cuda:0",
    )

    assert [output.name for output in module.outputs] == ["tokens", "logits"]
    assert [(value.dtype, value.shape) for value in region.outputs] == [
        ("i64", (1, 16)),
        ("bf16", (1, 1, 248320)),
    ]
    assert [(value.type.dtype, value.type.shape) for value in contracts] == [
        ("i64", (1, 16)),
        ("bf16", (1, 1, 248320)),
    ]
