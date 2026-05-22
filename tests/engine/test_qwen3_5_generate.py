import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.edge_fm_build_paths import prepend_built_python_paths

prepend_built_python_paths(project_root)

import edge_fm
from scripts.operator_table.utils import resolve_operator_table_path, resolve_target_hw_profile
from tests._support.temp_paths import make_temp_dir


DEVICE_ID = int(os.environ.get("EDGE_FM_DEVICE_ID", "0"))
CUDA_HW_PROFILE = resolve_target_hw_profile()
REGENERATE_DUMP_ENV = "EDGE_FM_QWEN3_5_REGENERATE_DUMP"
_REGENERATED_DUMP_DIRS: set[Path] = set()
MODEL_CASES = [
    pytest.param(
        "0.8b",
        (project_root / "examples" / "qwen3.5-0.8b" / "qwen3.5-0.8b").resolve(),
        "EDGE_FM_QWEN3_5_0P8B_DUMP_DIR",
        "tests/data/decode_dump_qwen3_5_0p8b",
        id="0.8b",
    ),
    pytest.param(
        "2b",
        (project_root / "examples" / "qwen3.5-2b" / "qwen3.5-2b").resolve(),
        "EDGE_FM_QWEN3_5_2B_DUMP_DIR",
        "tests/data/decode_dump_qwen3_5_2b",
        id="2b",
    ),
]


def _has_qwen3_5_weights(model_path: Path) -> bool:
    return (
        model_path.exists()
        and (model_path / "config.json").exists()
        and any(model_path.glob("model*.safetensors"))
    )


def _resolve_dump_dir(env_key: str, default_relative_path: str) -> Path:
    override = os.environ.get(env_key)
    if override:
        return Path(override).expanduser().resolve()
    return (project_root / default_relative_path).resolve()


def _has_reference_dump(dump_dir: Path) -> bool:
    return (
        dump_dir.exists()
        and (dump_dir / "input_ids.npy").exists()
        and (dump_dir / "decode_tokens.npy").exists()
        and (dump_dir / "metadata.json").exists()
    )


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _ensure_reference_dump(label: str, model_path: Path, dump_dir: Path) -> None:
    force_regenerate = _env_flag_enabled(REGENERATE_DUMP_ENV)
    if _has_reference_dump(dump_dir) and (not force_regenerate or dump_dir in _REGENERATED_DUMP_DIRS):
        return
    if not _has_qwen3_5_weights(model_path):
        pytest.skip(f"Qwen3.5-{label} checkpoint is not available")
    dump_script = project_root / "tests" / "scripts" / "dump_qwen3_5_decode.py"
    subprocess.run(
        [
            sys.executable,
            str(dump_script),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(dump_dir),
            "--num-steps",
            "20",
        ],
        cwd=str(project_root),
        check=True,
    )
    _REGENERATED_DUMP_DIRS.add(dump_dir)


def _create_engine_config(
    model_path: Path,
    seq_len: int,
    num_steps: int,
    *,
    use_cuda_graph: bool = False,
    lm_head_top1: bool | None = None,
) -> str:
    cfg_dir = Path(make_temp_dir("efm_qwen3_5_generate_cfg_"))
    cfg_path = cfg_dir / "engine_config.json"
    operator_table_path = resolve_operator_table_path(
        model_path=model_path,
        model_name="qwen3_5",
        config=json.loads((model_path / "config.json").read_text(encoding="utf-8"))["text_config"],
    )
    max_tokens = seq_len + num_steps
    runtime = {
        "device": "cuda",
        "device_id": DEVICE_ID,
        "hw_profile": CUDA_HW_PROFILE,
    }
    if use_cuda_graph:
        runtime["use_cuda_graph"] = True
    if lm_head_top1 is not None:
        runtime["lm_head_top1"] = {"enabled": lm_head_top1}
    engine_config = {
        "model_name": "Qwen3.5",
        "runtime": runtime,
        "operator_impl_table_path": str(operator_table_path),
        "prefill_model_path": str(model_path),
        "kvcache": {
            "dtype": "bf16",
            "attention_type": "gqa",
            "requests": [{"request_id": 0, "prefix_token_ids": [], "max_tokens": max_tokens}],
        },
        "sampling": {"temperature": 0.0, "seed": 42, "max_new_tokens": num_steps + 1},
    }
    cfg_path.write_text(json.dumps(engine_config, indent=2), encoding="utf-8")
    return str(cfg_path)


def _create_exact_decode_engine_config(
    model_path: Path,
    seq_len: int,
    decode_len: int,
    *,
    use_cuda_graph: bool = False,
    lm_head_top1: bool | None = None,
) -> str:
    cfg_dir = Path(make_temp_dir("efm_qwen3_5_exact_decode_cfg_"))
    cfg_path = cfg_dir / "engine_config.json"
    text_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))["text_config"]
    operator_table_path = resolve_operator_table_path(
        model_path=model_path,
        model_name="qwen3_5",
        config=text_config,
    )
    runtime = {
        "device": "cuda",
        "device_id": DEVICE_ID,
        "hw_profile": CUDA_HW_PROFILE,
        "use_cuda_graph": use_cuda_graph,
    }
    if lm_head_top1 is not None:
        runtime["lm_head_top1"] = {"enabled": lm_head_top1}
    engine_config = {
        "model_name": "Qwen3.5",
        "runtime": runtime,
        "operator_impl_table_path": str(operator_table_path),
        "prefill_model_path": str(model_path),
        "kvcache": {
            "dtype": "bf16",
            "attention_type": "gqa",
            "requests": [{"request_id": 0, "prefix_token_ids": [], "max_tokens": seq_len + decode_len - 1}],
        },
        "sampling": {"temperature": 0.0, "seed": 42, "max_new_tokens": decode_len},
    }
    cfg_path.write_text(json.dumps(engine_config, indent=2), encoding="utf-8")
    return str(cfg_path)


def _repeat_tokens_to_length(token_ids: list[int], target_len: int) -> list[int]:
    assert token_ids
    repeat = (target_len + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeat)[:target_len]


@pytest.mark.parametrize(("label", "model_path", "dump_env_key", "default_dump_dir"), MODEL_CASES)
def test_qwen3_5_engine_can_construct_text_model(
    label: str,
    model_path: Path,
    dump_env_key: str,
    default_dump_dir: str,
):
    if not _has_qwen3_5_weights(model_path):
        pytest.skip(f"Qwen3.5-{label} checkpoint is not available")
    cfg_path = _create_engine_config(model_path, seq_len=8, num_steps=1)
    engine = edge_fm.EdgeFM(cfg_path)
    assert engine is not None


@pytest.mark.parametrize(("label", "model_path", "dump_env_key", "default_dump_dir"), MODEL_CASES)
def test_qwen3_5_generate_token_alignment(
    label: str,
    model_path: Path,
    dump_env_key: str,
    default_dump_dir: str,
):
    if not _has_qwen3_5_weights(model_path):
        pytest.skip(f"Qwen3.5-{label} checkpoint is not available")
    dump_dir = _resolve_dump_dir(dump_env_key, default_dump_dir)
    _ensure_reference_dump(label, model_path, dump_dir)

    token_ids = np.load(dump_dir / "input_ids.npy").astype(np.int32).flatten().tolist()
    decode_tokens = np.load(dump_dir / "decode_tokens.npy").astype(np.int32).flatten().tolist()
    num_steps = len(decode_tokens)
    cfg_path = _create_engine_config(model_path, seq_len=len(token_ids), num_steps=num_steps)

    engine = edge_fm.EdgeFM(cfg_path)
    request = edge_fm.Request(0, token_ids)
    request.set_ignore_stop_tokens(True)
    response = engine.generate(request)
    got_tokens = response.token_ids()[:num_steps]

    assert got_tokens == decode_tokens[:num_steps]
    metrics = engine.last_generate_metrics()
    assert float(metrics["lm_head_top1_enabled"]) == 1.0
    assert float(metrics["lm_head_top1_decode_steps"]) >= float(num_steps)


@pytest.mark.parametrize(("label", "model_path", "dump_env_key", "default_dump_dir"), MODEL_CASES)
def test_qwen3_5_cuda_graph_config_enables_decode_graph(
    label: str,
    model_path: Path,
    dump_env_key: str,
    default_dump_dir: str,
):
    if not _has_qwen3_5_weights(model_path):
        pytest.skip(f"Qwen3.5-{label} checkpoint is not available")
    dump_dir = _resolve_dump_dir(dump_env_key, default_dump_dir)
    _ensure_reference_dump(label, model_path, dump_dir)

    token_ids = np.load(dump_dir / "input_ids.npy").astype(np.int32).flatten().tolist()
    decode_tokens = np.load(dump_dir / "decode_tokens.npy").astype(np.int32).flatten().tolist()
    num_steps = len(decode_tokens)
    cfg_path = _create_engine_config(
        model_path,
        seq_len=len(token_ids),
        num_steps=num_steps,
        use_cuda_graph=True,
    )

    engine = edge_fm.EdgeFM(cfg_path)
    request = edge_fm.Request(0, token_ids)
    request.set_ignore_stop_tokens(True)
    response = engine.generate(request)
    got_tokens = response.token_ids()[:num_steps]

    assert got_tokens == decode_tokens[:num_steps]
    metrics = engine.last_generate_metrics()
    assert float(metrics["cuda_graph_enabled"]) == 1.0
    assert float(metrics["lm_head_top1_enabled"]) == 1.0
    assert float(metrics["lm_head_top1_decode_steps"]) >= float(num_steps)


@pytest.mark.parametrize(("label", "model_path", "dump_env_key", "default_dump_dir"), MODEL_CASES)
def test_qwen3_5_cuda_graph_tokens_match_regular_decode(
    label: str,
    model_path: Path,
    dump_env_key: str,
    default_dump_dir: str,
):
    if not _has_qwen3_5_weights(model_path):
        pytest.skip(f"Qwen3.5-{label} checkpoint is not available")
    dump_dir = _resolve_dump_dir(dump_env_key, default_dump_dir)
    _ensure_reference_dump(label, model_path, dump_dir)

    token_ids = np.load(dump_dir / "input_ids.npy").astype(np.int32).flatten().tolist()
    decode_tokens = np.load(dump_dir / "decode_tokens.npy").astype(np.int32).flatten().tolist()
    num_steps = len(decode_tokens)

    regular_engine = edge_fm.EdgeFM(_create_engine_config(model_path, seq_len=len(token_ids), num_steps=num_steps))
    regular_request = edge_fm.Request(0, token_ids)
    regular_request.set_ignore_stop_tokens(True)
    regular_tokens = regular_engine.generate(regular_request).token_ids()[:num_steps]

    graph_engine = edge_fm.EdgeFM(_create_engine_config(
        model_path,
        seq_len=len(token_ids),
        num_steps=num_steps,
        use_cuda_graph=True,
    ))
    graph_request = edge_fm.Request(0, token_ids)
    graph_request.set_ignore_stop_tokens(True)
    graph_response = graph_engine.generate(graph_request)
    graph_tokens = graph_response.token_ids()[:num_steps]

    assert regular_tokens == decode_tokens[:num_steps]
    assert graph_tokens == regular_tokens
    assert float(graph_engine.last_generate_metrics()["cuda_graph_enabled"]) == 1.0


@pytest.mark.parametrize(("label", "model_path", "dump_env_key", "default_dump_dir"), MODEL_CASES)
def test_qwen3_5_lm_head_top1_token_alignment(
    label: str,
    model_path: Path,
    dump_env_key: str,
    default_dump_dir: str,
):
    if not _has_qwen3_5_weights(model_path):
        pytest.skip(f"Qwen3.5-{label} checkpoint is not available")
    dump_dir = _resolve_dump_dir(dump_env_key, default_dump_dir)
    _ensure_reference_dump(label, model_path, dump_dir)

    token_ids = np.load(dump_dir / "input_ids.npy").astype(np.int32).flatten().tolist()
    decode_tokens = np.load(dump_dir / "decode_tokens.npy").astype(np.int32).flatten().tolist()
    num_steps = len(decode_tokens)
    cfg_path = _create_engine_config(
        model_path,
        seq_len=len(token_ids),
        num_steps=num_steps,
        lm_head_top1=True,
    )

    engine = edge_fm.EdgeFM(cfg_path)
    request = edge_fm.Request(0, token_ids)
    request.set_ignore_stop_tokens(True)
    response = engine.generate(request)
    got_tokens = response.token_ids()[:num_steps]

    assert got_tokens == decode_tokens[:num_steps]
    metrics = engine.last_generate_metrics()
    assert float(metrics["lm_head_top1_enabled"]) == 1.0
    assert float(metrics["lm_head_top1_decode_steps"]) >= float(num_steps)


@pytest.mark.parametrize(("label", "model_path", "dump_env_key", "default_dump_dir"), MODEL_CASES)
def test_qwen3_5_cuda_graph_lm_head_top1_token_alignment(
    label: str,
    model_path: Path,
    dump_env_key: str,
    default_dump_dir: str,
):
    if not _has_qwen3_5_weights(model_path):
        pytest.skip(f"Qwen3.5-{label} checkpoint is not available")
    dump_dir = _resolve_dump_dir(dump_env_key, default_dump_dir)
    _ensure_reference_dump(label, model_path, dump_dir)

    token_ids = np.load(dump_dir / "input_ids.npy").astype(np.int32).flatten().tolist()
    decode_tokens = np.load(dump_dir / "decode_tokens.npy").astype(np.int32).flatten().tolist()
    num_steps = len(decode_tokens)
    cfg_path = _create_engine_config(
        model_path,
        seq_len=len(token_ids),
        num_steps=num_steps,
        use_cuda_graph=True,
        lm_head_top1=True,
    )

    engine = edge_fm.EdgeFM(cfg_path)
    request = edge_fm.Request(0, token_ids)
    request.set_ignore_stop_tokens(True)
    response = engine.generate(request)
    got_tokens = response.token_ids()[:num_steps]

    assert got_tokens == decode_tokens[:num_steps]
    metrics = engine.last_generate_metrics()
    assert float(metrics["cuda_graph_enabled"]) == 1.0
    assert float(metrics["lm_head_top1_enabled"]) == 1.0
    assert float(metrics["lm_head_top1_decode_steps"]) >= float(num_steps)


def test_qwen3_5_cuda_graph_repeated_generate_with_long_prefill_matches_regular_decode():
    label = "0.8b"
    model_path = (project_root / "examples" / "qwen3.5-0.8b" / "qwen3.5-0.8b").resolve()
    if not _has_qwen3_5_weights(model_path):
        pytest.skip("Qwen3.5-0.8b checkpoint is not available")
    dump_dir = _resolve_dump_dir("EDGE_FM_QWEN3_5_0P8B_DUMP_DIR", "tests/data/decode_dump_qwen3_5_0p8b")
    _ensure_reference_dump(label, model_path, dump_dir)

    base_token_ids = np.load(dump_dir / "input_ids.npy").astype(np.int32).flatten().tolist()
    token_ids = _repeat_tokens_to_length(base_token_ids, 512)
    decode_len = 32

    graph_engine = edge_fm.EdgeFM(_create_exact_decode_engine_config(
        model_path,
        seq_len=len(token_ids),
        decode_len=decode_len,
        use_cuda_graph=True,
    ))
    graph_runs = []
    for _ in range(2):
        graph_request = edge_fm.Request(0, token_ids)
        graph_request.set_ignore_stop_tokens(True)
        graph_response = graph_engine.generate(graph_request)
        graph_runs.append(graph_response.token_ids()[:decode_len])
        metrics = graph_engine.last_generate_metrics()
        assert float(metrics["cuda_graph_enabled"]) == 1.0
        expected_captures = 1.0 if len(graph_runs) == 1 else 0.0
        assert float(metrics["decode_graph_captures"]) == expected_captures

    regular_engine = edge_fm.EdgeFM(_create_exact_decode_engine_config(
        model_path,
        seq_len=len(token_ids),
        decode_len=decode_len,
    ))
    regular_request = edge_fm.Request(0, token_ids)
    regular_request.set_ignore_stop_tokens(True)
    regular_tokens = regular_engine.generate(regular_request).token_ids()[:decode_len]
    assert graph_runs == [regular_tokens, regular_tokens]
