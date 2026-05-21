import json
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.edge_fm_build_paths import prepend_built_python_paths

prepend_built_python_paths(project_root)

import edge_fm


def _write_engine_config(model_config: dict, model_name: str = "Qwen3.5") -> str:
    model_dir = Path(tempfile.mkdtemp())
    (model_dir / "config.json").write_text(json.dumps(model_config), encoding="utf-8")

    cfg_dir = Path(tempfile.mkdtemp())
    cfg_path = cfg_dir / "engine_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "runtime": {"device": "cuda", "device_id": 0, "use_cuda_graph": True},
                "prefill_model_path": str(model_dir),
                "kvcache": {
                    "attention_type": "gqa",
                    "dtype": "fp16",
                    "requests": [{"request_id": 0, "max_tokens": 32}],
                },
            }
        ),
        encoding="utf-8",
    )
    return str(cfg_path)


def _qwen3_5_text_config() -> dict:
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 24,
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248320,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        },
    }


def test_qwen3_5_runtime_spec_uses_explicit_head_dim_and_layer_types():
    cfg_path = _write_engine_config(_qwen3_5_text_config())

    spec = edge_fm.resolve_model_runtime_spec(cfg_path)

    assert spec["model_name"] == "qwen3_5"
    assert spec["num_layers"] == 24
    assert spec["head_dim"] == 256
    assert spec["num_attention_heads"] == 8
    assert spec["num_key_value_heads"] == 2
    assert spec["kv_cache_layers"] == [3, 7, 11, 15, 19, 23]
    assert spec["supports_decode_cuda_graph"] is False


def test_qwen2_5_runtime_spec_keeps_all_layers_and_decode_graph():
    cfg_path = _write_engine_config(
        {
            "model_type": "qwen2",
            "num_hidden_layers": 3,
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
        },
        model_name="Qwen2.5",
    )

    spec = edge_fm.resolve_model_runtime_spec(cfg_path)

    assert spec["model_name"] == "qwen2_5"
    assert spec["head_dim"] == 128
    assert spec["kv_cache_layers"] == [0, 1, 2]
    assert spec["supports_decode_cuda_graph"] is True
