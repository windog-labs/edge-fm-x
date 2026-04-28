import json
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.edge_fm_build_paths import prepend_built_python_paths
from scripts.operator_table.utils import resolve_operator_table_path

prepend_built_python_paths(project_root)

import edge_fm


def test_edgefm_tensor_stage_api_surface():
    assert hasattr(edge_fm.EdgeFM, "prefill")
    assert hasattr(edge_fm.EdgeFM, "decode")
    assert not hasattr(edge_fm.EdgeFM, "smolvla_prefill")
    assert not hasattr(edge_fm.EdgeFM, "smolvla_expert_denoise")
    assert not hasattr(edge_fm, "SmolVLAPrefillResult")


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


def _find_small_hf_model_path() -> str | None:
    candidates = [
        str(project_root / "examples" / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct"),
        str(project_root / "examples" / "qwen2.5-0.5b-instruct"),
        _find_hf_model_path(),
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


def _create_engine_config(
    model_path: str,
    max_tokens: int = 32,
    runtime_device: str = "cuda",
    model_name: str = "Qwen2.5",
    include_model_name: bool = True,
) -> str:
    config = _load_model_config(model_path)
    torch_dtype = str(config.get("torch_dtype", "float16")).lower()
    kvcache_dtype = "bf16" if ("bfloat" in torch_dtype or "bf16" in torch_dtype) else "fp16"
    num_heads = config.get("num_attention_heads", 8)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    attention_type = "gqa" if num_kv_heads < num_heads else "mha"
    engine_config_dir = tempfile.mkdtemp()
    engine_config_path = Path(engine_config_dir) / "engine_config.json"

    engine_config = {
        "runtime": {
            "device": runtime_device,
            "device_id": int(os.environ.get("EDGE_FM_DEVICE_ID", "0")),
            "hw_profile": "cuda_sm80" if runtime_device == "cuda" else runtime_device,
        },
        "operator_impl_table_path": str(
            resolve_operator_table_path(
                model_path=Path(model_path).resolve(),
                model_name=model_name,
            )
        ),
        "prefill_model_path": str(Path(model_path).resolve()),
        "kvcache": {
            "dtype": kvcache_dtype,
            "attention_type": attention_type,
            "requests": [{"request_id": 0, "prefix_token_ids": [], "max_tokens": max_tokens}],
        },
        "sampling": {"temperature": 0.0, "seed": 42},
    }
    if include_model_name:
        engine_config["model_name"] = model_name

    with open(engine_config_path, "w", encoding="utf-8") as f:
        json.dump(engine_config, f, indent=2)
    return str(engine_config_path)


@pytest.fixture(scope="module")
def hf_model_path() -> str:
    model_path = _find_hf_model_path()
    if model_path is None:
        pytest.skip("Model path not found for engine config API tests")
    return model_path


@pytest.fixture(scope="module")
def small_hf_model_path() -> str:
    model_path = _find_small_hf_model_path()
    if model_path is None:
        pytest.skip("Small model path not found for tuning smoke tests")
    return model_path


def test_from_model_is_deprecated(hf_model_path: str):
    engine_json = _create_engine_config(hf_model_path)
    with pytest.raises(Exception, match="deprecated"):
        edge_fm.EdgeFM.from_model(object(), engine_json)


def test_engine_requires_explicit_model_name(hf_model_path: str):
    engine_json = _create_engine_config(hf_model_path, include_model_name=False)
    with pytest.raises(Exception, match="model_name"):
        edge_fm.EdgeFM(engine_json)


def test_engine_from_config_and_tune_api(hf_model_path: str):
    engine_json = _create_engine_config(hf_model_path, model_name="Qwen2.5")
    engine = edge_fm.EdgeFM(engine_json)
    engine.tune()

    request = edge_fm.Request(request_id=0, token_ids=[0, 1, 2, 3])
    response = engine.generate(request)
    assert isinstance(list(response.token_ids()), list)


def test_horizon_tune_emits_compile_spec_v2(hf_model_path: str):
    engine_json = _create_engine_config(hf_model_path, runtime_device="horizon", model_name="Qwen2.5")
    engine = edge_fm.EdgeFM(engine_json)
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
    assert compile_spec["schema"] == "edgefm_horizon_compile_spec_v2"
    assert compile_spec["model_name"] == "qwen2_5"
    assert "graph_tuning" in compile_spec
    assert "model_description" not in compile_spec
    assert "linear_operator_table" in compile_spec["graph_tuning"]
    assert "linear_impl_overrides" not in compile_spec["graph_tuning"]
    assert compile_spec["generated_module"]["factory_function"] == "build_model"
    assert compile_spec["helper_script"] == "scripts/horizon/compile_horizon_from_spec.py"


def test_config_driven_cuda_tuning_smoke(
    small_hf_model_path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EDGE_FM_TUNING_REDUCED_CANDIDATES", "1")
    monkeypatch.setenv("EDGE_FM_TUNING_PREFILL_LIST", "128")
    monkeypatch.setenv("EDGE_FM_TUNING_KV_LENS", "128")
    monkeypatch.setenv("EDGE_FM_TUNING_ATTENTION_WARMUP", "2")
    monkeypatch.setenv("EDGE_FM_TUNING_ATTENTION_ITERS", "5")
    monkeypatch.setenv("EDGE_FM_TUNING_LINEAR_WARMUP", "1")
    monkeypatch.setenv("EDGE_FM_TUNING_LINEAR_ITERS", "3")

    engine_json = Path(_create_engine_config(small_hf_model_path, model_name="Qwen2.5"))
    config = json.loads(engine_json.read_text(encoding="utf-8"))
    config["runtime"]["hw_profile"] = "cuda_sm86"
    config["tuning"] = {"enabled": True}
    engine_json.write_text(json.dumps(config, indent=2), encoding="utf-8")

    capfd.readouterr()

    engine = edge_fm.EdgeFM(str(engine_json))
    response = engine.generate(edge_fm.Request(request_id=0, token_ids=[151644, 198, 151645]))
    assert isinstance(list(response.token_ids()), list)

    first_logs = "".join(capfd.readouterr())
    assert "[tuning] start" in first_logs
    assert "[tuning] completed" in first_logs

    cache_root = Path(os.environ["HOME"]) / ".cache" / "edge-fm" / "backend_artifacts"
    tuned_tables = list(cache_root.glob("*/cuda_operator_tuning/operator_impl_table.json"))
    tuning_reports = list(cache_root.glob("*/cuda_operator_tuning/tuning_report.json"))
    assert tuned_tables
    assert tuning_reports

    engine = edge_fm.EdgeFM(str(engine_json))
    response = engine.generate(edge_fm.Request(request_id=0, token_ids=[151644, 198, 151645]))
    assert isinstance(list(response.token_ids()), list)

    second_logs = "".join(capfd.readouterr())
    assert "[tuning] cache hit" in second_logs or "[tuning] reusing active tuned operator table" in second_logs


def test_config_driven_tuning_rejects_horizon(hf_model_path: str):
    engine_json = Path(_create_engine_config(hf_model_path, runtime_device="horizon", model_name="Qwen2.5"))
    config = json.loads(engine_json.read_text(encoding="utf-8"))
    config["tuning"] = {"enabled": True}
    engine_json.write_text(json.dumps(config, indent=2), encoding="utf-8")

    with pytest.raises(Exception, match="supports CUDA only|Horizon continues to use explicit engine.tune"):
        edge_fm.EdgeFM(str(engine_json))
