import json
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent.parent
for _p in [project_root / "build" / "python", project_root / "build" / "install" / "python"]:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

import edge_fm


def _find_hf_model_path() -> str | None:
    candidates = [
        os.environ.get("EDGE_FM_TEST_MODEL_PATH"),
        os.environ.get("EDGE_FM_QWEN_MODEL_PATH"),
        str(project_root / "examples" / "qwen2.5-1.5b-instruct" / "qwen2.5-1.5b-instruct"),
        str(project_root / "examples" / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct"),
        str(project_root / "examples" / "qwen2.5-1.5b-instruct"),
        str(project_root / "examples" / "qwen2.5-0.5b-instruct"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists() and (path / "config.json").exists():
            if (path / "model.safetensors").exists() or list(path.glob("model-*.safetensors")):
                return str(path.resolve())
    return None


def _load_model_config(model_path: str) -> dict:
    with open(Path(model_path) / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    if "text_config" in config and isinstance(config["text_config"], dict):
        return config["text_config"]
    return config


def _create_engine_config(model_path: str, max_tokens: int = 32, runtime_device: str = "cuda") -> str:
    config = _load_model_config(model_path)
    num_heads = config.get("num_attention_heads", 8)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    attention_type = "gqa" if num_kv_heads < num_heads else "mha"
    engine_config_dir = tempfile.mkdtemp()
    engine_config_path = Path(engine_config_dir) / "engine_config.json"
    with open(engine_config_path, "w", encoding="utf-8") as f:
        json.dump({
            "runtime": {"device": runtime_device, "device_id": int(os.environ.get("EDGE_FM_DEVICE_ID", "0"))},
            "prefill_model_path": str(Path(model_path).resolve()),
            "kvcache": {
                "dtype": "fp16",
                "attention_type": attention_type,
                "requests": [{"request_id": 0, "prefix_token_ids": [], "max_tokens": max_tokens}],
            },
            "sampling": {"temperature": 0.0, "seed": 42},
        }, f, indent=2)
    return str(engine_config_path)


@pytest.fixture(scope="module")
def hf_model_path() -> str:
    model_path = _find_hf_model_path()
    if model_path is None:
        pytest.skip("Model path not found for from_model API tests")
    return model_path


def test_from_model_rejects_layer_mismatch(hf_model_path: str):
    engine_json = _create_engine_config(hf_model_path)
    model_config = _load_model_config(hf_model_path)
    num_layers = model_config["num_hidden_layers"]
    model = edge_fm.DecoderOnlyModel.hf(num_layers + 1)
    with pytest.raises(Exception):
        edge_fm.EdgeFM.from_model(model, engine_json)


def test_from_model_and_tune_api(hf_model_path: str):
    engine_json = _create_engine_config(hf_model_path)
    model_config = _load_model_config(hf_model_path)
    num_layers = model_config["num_hidden_layers"]

    model = edge_fm.DecoderOnlyModel.hf(num_layers)
    engine = edge_fm.EdgeFM.from_model(model, engine_json)
    engine.tune()

    request = edge_fm.Request(request_id=0, token_ids=[0, 1, 2, 3])
    response = engine.generate(request)
    assert isinstance(list(response.token_ids()), list)


def test_from_model_horizon_tune_emits_compile_spec(hf_model_path: str):
    engine_json = _create_engine_config(hf_model_path, runtime_device="horizon")
    model_config = _load_model_config(hf_model_path)
    num_layers = model_config["num_hidden_layers"]

    model = edge_fm.DecoderOnlyModel.hf(num_layers)
    engine = edge_fm.EdgeFM.from_model(model, engine_json)
    engine.tune()

    with pytest.raises(Exception) as exc_info:
        engine.generate(edge_fm.Request(request_id=0, token_ids=[0, 1]))

    message = str(exc_info.value)
    match = re.search(r"generated spec: (.+)$", message)
    assert match is not None, message
    compile_spec_path = Path(match.group(1).strip())
    assert compile_spec_path.exists()

    compile_spec = json.loads(compile_spec_path.read_text(encoding="utf-8"))
    module_path = Path(compile_spec["generated_module"]["module_path"])
    assert module_path.exists()
    assert compile_spec["generated_module"]["factory_function"] == "build_model"
    assert compile_spec["helper_script"] == "scripts/compile_horizon_from_spec.py"
