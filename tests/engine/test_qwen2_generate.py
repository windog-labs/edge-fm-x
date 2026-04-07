"""
Qwen2.5 generate 对齐测试（pytest）

验证 edge_fm.EdgeFM.generate() 的 greedy 解码输出与 Transformers 参考 dump 一致。
dump 数据位于 tests/data/decode_dump/，首次运行时自动通过 Transformers 生成。

默认使用 GPU device 1（可通过环境变量 EDGE_FM_DEVICE_ID 覆盖，如 EDGE_FM_DEVICE_ID=0）。

运行（建议在项目根目录 /xs-train-nas/zzm/repos/edge-fm 下）:
  pytest -s tests/engine/test_qwen2_generate.py
  pytest -s tests/engine/test_qwen2_generate.py -k test_generate_token_alignment
  pytest -s tests/engine/test_qwen2_generate.py -k benchmark  # 性能基准
"""

import json
import os
import statistics as stats
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
for _p in [project_root / "build" / "python", project_root / "build" / "install" / "python"]:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

import edge_fm

# Optional: TRT-Edge-LLM in-process runtime (built with BUILD_TRT_EDGELLM_PYBIND=ON)
try:
    import edge_fm_trt
except ImportError:
    edge_fm_trt = None

DUMP_DIR = project_root / "tests" / "data" / "decode_dump"
DUMP_DIR_VL = project_root / "tests" / "data" / "decode_dump_vl"

DEFAULT_PROMPT = "Hello, how are you today?"
DEFAULT_VLM_PROMPT = "What animal is on the candy?"
DEFAULT_VLM_IMAGE_PATH = project_root / "tests" / "data" / "candy.JPG"
DEFAULT_SEQ_LEN = 6
DEFAULT_NUM_STEPS = 20
DEFAULT_SEED = 42
DEFAULT_BENCH_PREFILL_LENGTHS = [512, 1024, 2048]
DEFAULT_BENCH_DECODE_LENGTHS = [32, 64]
DEFAULT_BENCH_LLM_MODEL_SIZES = ["0.5b", "1.5b", "3b"]
DEFAULT_BENCH_VLM_MODEL_SIZES = ["3b", "7b"]

LLM_MODEL_SPECS = {
    "0.5b": {
        "label": "Qwen2.5-0.5B-Instruct",
        "dir_name": "qwen2.5-0.5b-instruct",
        "trt_workspace_name": "qwen2.5-0.5b",
        "env_keys": ["EDGE_FM_QWEN_0_5B_MODEL_PATH"],
        "trt_engine_env_keys": ["TRT_EDGELLM_ENGINE_DIR_0_5B"],
    },
    "1.5b": {
        "label": "Qwen2.5-1.5B-Instruct",
        "dir_name": "qwen2.5-1.5b-instruct",
        "trt_workspace_name": "qwen2.5-1.5b",
        "env_keys": ["EDGE_FM_QWEN_1_5B_MODEL_PATH", "EDGE_FM_QWEN_MODEL_PATH"],
        "trt_engine_env_keys": ["TRT_EDGELLM_ENGINE_DIR_1_5B", "TRT_EDGELLM_ENGINE_DIR"],
    },
    "3b": {
        "label": "Qwen2.5-3B-Instruct",
        "dir_name": "qwen2.5-3b-instruct",
        "trt_workspace_name": "qwen2.5-3b",
        "env_keys": ["EDGE_FM_QWEN_3B_MODEL_PATH"],
        "trt_engine_env_keys": ["TRT_EDGELLM_ENGINE_DIR_3B", "TRT_EDGELLM_ENGINE_DIR"],
    },
}

VLM_MODEL_SPECS = {
    "3b": {
        "label": "Qwen2.5-VL-3B-Instruct",
        "dir_name": "qwen2.5-vl-3b-instruct",
        "env_keys": ["EDGE_FM_QWEN_VL_3B_MODEL_PATH", "EDGE_FM_QWEN_VL_MODEL_PATH"],
        "trt_workspace_name": "qwen2.5-vl-3b",
        "trt_engine_env_keys": ["TRT_EDGELLM_VLM_ENGINE_DIR_3B", "TRT_EDGELLM_VLM_ENGINE_DIR"],
        "trt_multimodal_engine_env_keys": [
            "TRT_EDGELLM_VLM_MULTIMODAL_ENGINE_DIR_3B",
            "TRT_EDGELLM_VLM_MULTIMODAL_ENGINE_DIR",
        ],
    },
    "7b": {
        "label": "Qwen2.5-VL-7B-Instruct",
        "dir_name": "qwen2.5-vl-7b-instruct",
        "env_keys": ["EDGE_FM_QWEN_VL_7B_MODEL_PATH", "EDGE_FM_QWEN_VL_MODEL_PATH"],
        "trt_workspace_name": "qwen2.5-vl-7b",
        "trt_engine_env_keys": ["TRT_EDGELLM_VLM_ENGINE_DIR_7B", "TRT_EDGELLM_VLM_ENGINE_DIR"],
        "trt_multimodal_engine_env_keys": [
            "TRT_EDGELLM_VLM_MULTIMODAL_ENGINE_DIR_7B",
            "TRT_EDGELLM_VLM_MULTIMODAL_ENGINE_DIR",
        ],
    },
}

BENCH_MODEL_SPECS = {
    "llm": LLM_MODEL_SPECS,
    "vlm": VLM_MODEL_SPECS,
}

# GPU device：默认 1，避免占用 device 0；可通过环境变量 EDGE_FM_DEVICE_ID 覆盖
DEVICE_ID = int(os.environ.get("EDGE_FM_DEVICE_ID", "1"))
CUDA_DEVICE = f"cuda:{DEVICE_ID}"


# ---------------------------------------------------------------------------
# Dump generation (runs Transformers, only when dump is missing)
# ---------------------------------------------------------------------------

def _model_path_has_weights(path: Path) -> bool:
    if not path.exists() or not (path / "config.json").exists():
        return False
    if (path / "model.safetensors").exists():
        return True
    for f in path.glob("model-*.safetensors"):
        if f.exists():
            return True
    return False


def _candidate_model_paths(dir_name: str) -> list[Path]:
    return [
        (project_root / "examples" / dir_name / dir_name).resolve(),
        (project_root / "examples" / dir_name).resolve(),
    ]


def _find_bench_model_path(kind: str, model_size: str) -> str | None:
    spec = BENCH_MODEL_SPECS[kind][model_size]
    candidates = []
    for env_key in spec.get("env_keys", []):
        value = os.environ.get(env_key)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(_candidate_model_paths(spec["dir_name"]))
    for path in candidates:
        if _model_path_has_weights(path):
            return str(path.resolve())
    return None


def _find_qwen_model_path(model_size: str | None = None) -> str | None:
    if model_size is not None:
        return _find_bench_model_path("llm", model_size)
    for size in ["1.5b", "0.5b", "3b"]:
        path = _find_bench_model_path("llm", size)
        if path is not None:
            return path
    return None


def _find_qwen_vl_model_path(model_size: str | None = None) -> str | None:
    """查找 Qwen2.5-VL 模型路径（默认先尝试 3B，再尝试 7B）。"""
    if model_size is not None:
        return _find_bench_model_path("vlm", model_size)
    for size in ["3b", "7b"]:
        path = _find_bench_model_path("vlm", size)
        if path is not None:
            return path
    return None


def _generate_dump(
    model_path: str,
    output_dir: Path,
    prompt: str = DEFAULT_PROMPT,
    seq_len: int = DEFAULT_SEQ_LEN,
    num_steps: int = DEFAULT_NUM_STEPS,
    seed: int = DEFAULT_SEED,
) -> None:
    """Run Transformers prefill + greedy decode, save reference dump."""
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = AutoConfig.from_pretrained(model_path)
    torch_dtype_str = str(getattr(config, "torch_dtype", "float16")).lower()
    model_dtype = torch.bfloat16 if "bfloat" in torch_dtype_str or "bf16" in torch_dtype_str else torch.float16

    device = CUDA_DEVICE if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=False,
    )
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    encoded = tokenizer.encode(prompt, add_special_tokens=True)
    token_ids = encoded[:seq_len]
    pad_id = getattr(config, "pad_token_id", None) or getattr(config, "eos_token_id", 0)
    if len(token_ids) < seq_len:
        token_ids = token_ids + [pad_id] * (seq_len - len(token_ids))

    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    next_token = logits[0, -1].argmax(dim=-1).item()

    np.save(str(output_dir / "token_ids.npy"), np.array(token_ids, dtype=np.int32))
    np.save(str(output_dir / "prefill_logits.npy"), logits.detach().float().cpu().numpy())

    decode_tokens = [next_token]
    current_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)

    for step in range(num_steps):
        with torch.no_grad():
            outputs = model(current_ids)
        logits_step = outputs.logits
        last_logits = logits_step[0, -1].float().cpu().numpy()
        next_tok = int(logits_step[0, -1].argmax(dim=-1).item())

        input_tok = int(current_ids[0, -1].item())
        np.savez(
            str(output_dir / f"step_{step}.npz"),
            input_token_id=np.int32(input_tok),
            logits=last_logits.astype(np.float32),
            next_token_id=np.int32(next_tok),
        )
        decode_tokens.append(next_tok)
        current_ids = torch.cat([current_ids, torch.tensor([[next_tok]], dtype=torch.long, device=device)], dim=1)

    manifest = {
        "model_path": str(Path(model_path).resolve()),
        "seed": seed,
        "prompt": prompt,
        "seq_len": seq_len,
        "num_decode_steps": num_steps,
        "vocab_size": config.vocab_size,
        "token_ids_shape": list(np.array(token_ids).shape),
        "decode_tokens": decode_tokens,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    np.save(str(output_dir / "decode_tokens.npy"), np.array(decode_tokens, dtype=np.int32))
    print(f"[dump] Saved decode dump to {output_dir}")
    print(f"  prompt: {prompt!r}, token_ids: {token_ids}")
    print(f"  {num_steps} steps, decode_tokens={decode_tokens}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dump_data() -> dict:
    """Load (or generate) the Transformers reference dump."""
    manifest_path = DUMP_DIR / "manifest.json"
    if not manifest_path.exists():
        model_path = _find_qwen_model_path()
        if model_path is None:
            pytest.skip("Qwen2.5 model not found; set EDGE_FM_QWEN_MODEL_PATH or place model under examples/")
        print(f"\n[fixture] Generating reference dump with Transformers → {DUMP_DIR}")
        _generate_dump(model_path, DUMP_DIR)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    model_path = manifest["model_path"]
    if not Path(model_path).exists():
        fallback_model_path = _find_qwen_model_path()
        if fallback_model_path is None:
            pytest.skip(f"Model path in dump manifest not found: {model_path}")
        print(
            f"[fixture] Dump manifest model path missing: {model_path}\n"
            f"          Falling back to local model path: {fallback_model_path}"
        )
        model_path = fallback_model_path
        manifest["model_path"] = model_path

    token_ids = np.load(DUMP_DIR / "token_ids.npy")
    decode_tokens = np.load(DUMP_DIR / "decode_tokens.npy")

    return {
        "manifest": manifest,
        "model_path": model_path,
        "token_ids": token_ids,
        "decode_tokens": decode_tokens,
        "dump_dir": DUMP_DIR,
    }


def _load_model_config_for_engine(model_path: str) -> dict:
    """加载模型 config，VLM 时使用 text_config."""
    config_path = Path(model_path) / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if "text_config" in config and isinstance(config["text_config"], dict):
        return config["text_config"]
    return config


def _create_engine_config(
    model_path: str,
    seq_len: int,
    num_steps: int,
    use_cuda_graph: bool = False,
    prefix_token_ids: list | None = None,
    generated_tokens_total: int | None = None,
    model_name: str = "Qwen2.5",
) -> str:
    config = _load_model_config_for_engine(model_path)
    torch_dtype = str(config.get("torch_dtype", "float16")).lower()
    kvcache_dtype = "bf16" if ("bfloat" in torch_dtype or "bf16" in torch_dtype) else "fp16"
    num_heads = config.get("num_attention_heads", 8)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    attention_type = "gqa" if num_kv_heads < num_heads else "mha"
    if generated_tokens_total is None:
        # Alignment fixtures interpret num_steps as decode steps after the
        # prefill sample, so the engine must allow one extra generated token.
        generated_tokens_total = num_steps + 1
    max_tokens = seq_len + generated_tokens_total - 1
    engine_config_dir = tempfile.mkdtemp()
    engine_config_path = Path(engine_config_dir) / "engine_config.json"
    runtime = {"device": "cuda", "device_id": DEVICE_ID, "hw_profile": "cuda_sm80"}
    if use_cuda_graph:
        runtime["use_cuda_graph"] = True
    prefix = prefix_token_ids if prefix_token_ids is not None else []
    with open(engine_config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "runtime": runtime,
            "operator_impl_table_path": str((project_root / "examples" / "config" / "operator_impl_table.json").resolve()),
            "prefill_model_path": str(Path(model_path).resolve()),
            "kvcache": {
                "dtype": kvcache_dtype,
                "attention_type": attention_type,
                "requests": [{"request_id": 0, "prefix_token_ids": prefix, "max_tokens": max_tokens}],
            },
            "sampling": {
                "temperature": 0.0,
                "seed": 42,
            },
        }, f, indent=2)
    return str(engine_config_path)


@pytest.fixture(scope="module")
def engine_and_dump(dump_data):
    """Create the EdgeFM engine and provide dump data together."""
    manifest = dump_data["manifest"]
    model_path = dump_data["model_path"]
    num_steps = manifest["num_decode_steps"]
    seq_len = int(dump_data["token_ids"].size)
    engine_config_path = _create_engine_config(model_path, seq_len, num_steps)
    engine = edge_fm.EdgeFM(engine_config_path)
    return {**dump_data, "engine": engine, "num_steps": num_steps, "seq_len": seq_len}


@pytest.fixture(scope="module")
def engine_and_dump_cuda_graph(dump_data):
    """Create the EdgeFM engine with use_cuda_graph=True for decode graph verification.
    Uses prefix so engine warmup can eagerly capture the decode graph before request-time decode.
    """
    manifest = dump_data["manifest"]
    model_path = dump_data["model_path"]
    num_steps = manifest["num_decode_steps"]
    seq_len = int(dump_data["token_ids"].size)
    token_ids_flat = dump_data["token_ids"].flatten()
    prefix = token_ids_flat[: min(4, len(token_ids_flat))].tolist()
    engine_config_path = _create_engine_config(
        model_path, seq_len, num_steps, use_cuda_graph=True, prefix_token_ids=prefix
    )
    engine = edge_fm.EdgeFM(engine_config_path)
    return {**dump_data, "engine": engine, "num_steps": num_steps, "seq_len": seq_len}


# ---------------------------------------------------------------------------
# VL（含图）dump 与 fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dump_data_vl() -> dict:
    """加载 VL 参考 dump（需先运行 tests/scripts/dump_qwen2_5_vl_decode.py 生成）。"""
    manifest_path = DUMP_DIR_VL / "manifest.json"
    if not manifest_path.exists():
        model_path = _find_qwen_vl_model_path()
        if model_path is None:
            pytest.skip(
                "VL dump 不存在且未找到 Qwen2.5-VL 模型；"
                "请先运行: python tests/scripts/dump_qwen2_5_vl_decode.py"
            )
        pytest.skip(
            f"VL dump 不存在，请先运行: python tests/scripts/dump_qwen2_5_vl_decode.py "
            f"(模型路径: {model_path})"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    model_path = manifest["model_path"]
    if not Path(model_path).exists():
        fallback_model_path = _find_qwen_vl_model_path()
        if fallback_model_path is None:
            pytest.skip(f"VL dump 中的模型路径不存在: {model_path}")
        print(
            f"[fixture] VL dump manifest model path missing: {model_path}\n"
            f"          Falling back to local model path: {fallback_model_path}"
        )
        model_path = fallback_model_path
        manifest["model_path"] = model_path
    token_ids = np.load(DUMP_DIR_VL / "token_ids.npy")
    decode_tokens = np.load(DUMP_DIR_VL / "decode_tokens.npy")
    image_embeddings = np.load(DUMP_DIR_VL / "image_embeddings.npy")
    embed_token_id = int(manifest["embed_token_id"])

    position_ids_path = DUMP_DIR_VL / "position_ids.npy"
    position_ids = np.load(str(position_ids_path)) if position_ids_path.exists() else None

    return {
        "manifest": manifest,
        "model_path": model_path,
        "token_ids": token_ids,
        "decode_tokens": decode_tokens,
        "dump_dir": DUMP_DIR_VL,
        "image_embeddings": image_embeddings,
        "embed_token_id": embed_token_id,
        "position_ids": position_ids,
    }


@pytest.fixture(scope="module")
def engine_and_dump_vl(dump_data_vl):
    """Create the EdgeFM engine 与 VL dump 数据（含图 embedding）。"""
    manifest = dump_data_vl["manifest"]
    model_path = dump_data_vl["model_path"]
    num_steps = manifest["num_decode_steps"]
    seq_len = int(dump_data_vl["token_ids"].size)
    engine_config_path = _create_engine_config(model_path, seq_len, num_steps, model_name="Qwen2.5-VL")
    engine = edge_fm.EdgeFM(engine_config_path)
    return {
        **dump_data_vl,
        "engine": engine,
        "num_steps": num_steps,
        "seq_len": seq_len,
    }


@pytest.fixture(scope="module")
def engine_and_dump_vl_cuda_graph(dump_data_vl):
    """Create the EdgeFM engine with use_cuda_graph=True for VLM decode alignment.

    Intentionally avoids prefix_token_ids so graph capture happens on the full
    multimodal request path instead of a text-only warmup prefix.
    """
    manifest = dump_data_vl["manifest"]
    model_path = dump_data_vl["model_path"]
    num_steps = manifest["num_decode_steps"]
    seq_len = int(dump_data_vl["token_ids"].size)
    engine_config_path = _create_engine_config(
        model_path,
        seq_len,
        num_steps,
        use_cuda_graph=True,
        model_name="Qwen2.5-VL",
    )
    engine = edge_fm.EdgeFM(engine_config_path)
    return {
        **dump_data_vl,
        "engine": engine,
        "num_steps": num_steps,
        "seq_len": seq_len,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size != b.size:
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 and nb < 1e-12:
        return 1.0
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _make_vl_request_factory(
    token_ids_list: list[int],
    image_embeddings: np.ndarray,
    embed_token_id: int,
    position_ids: np.ndarray | None,
    ignore_stop_tokens: bool = True,
):
    """Build a request factory for VLM runs while keeping backing tensors alive."""
    import torch

    emb_tensor = torch.from_numpy(image_embeddings).to(dtype=torch.bfloat16).to(CUDA_DEVICE).contiguous()
    embedding_tensor = edge_fm.Tensor.from_dlpack(emb_tensor.__dlpack__())

    pos_tensor = None
    position_ids_tensor = None
    if position_ids is not None:
        pos_tensor = torch.from_numpy(position_ids.astype(np.int32)).to(CUDA_DEVICE).contiguous()
        position_ids_tensor = edge_fm.Tensor.from_dlpack(pos_tensor.__dlpack__())

    keepalive = (emb_tensor, pos_tensor)

    def make_request():
        _ = keepalive
        if position_ids_tensor is not None:
            request = edge_fm.Request(
                0, token_ids_list, embedding_tensor, embed_token_id, position_ids_tensor
            )
        else:
            request = edge_fm.Request(0, token_ids_list, embedding_tensor, embed_token_id)
        request.set_ignore_stop_tokens(ignore_stop_tokens)
        return request

    return make_request


def _assert_vl_alignment(engine_bundle: dict, label: str):
    """Assert token alignment for a VLM request on the provided engine bundle."""
    import torch

    engine = engine_bundle["engine"]
    token_ids = engine_bundle["token_ids"]
    decode_tokens = engine_bundle["decode_tokens"]
    num_steps = engine_bundle["num_steps"]
    image_embeddings = engine_bundle["image_embeddings"]
    embed_token_id = engine_bundle["embed_token_id"]
    position_ids = engine_bundle["position_ids"]

    token_ids_list = token_ids.flatten().tolist()
    ref_tokens = decode_tokens[1: num_steps + 1].tolist()
    make_request = _make_vl_request_factory(
        token_ids_list,
        image_embeddings,
        embed_token_id,
        position_ids,
        ignore_stop_tokens=True,
    )

    response = engine.generate(make_request())
    all_tokens = response.token_ids()
    got_tokens = all_tokens[1: 1 + num_steps]

    torch.cuda.synchronize()

    assert len(got_tokens) >= num_steps, (
        f"EdgeFM only generated {len(got_tokens)} tokens, expected {num_steps}"
    )

    mismatches = []
    for step in range(num_steps):
        efm_tok = got_tokens[step] if step < len(got_tokens) else -1
        ref_tok = int(ref_tokens[step]) if step < len(ref_tokens) else -1
        if efm_tok != ref_tok:
            mismatches.append((step, efm_tok, ref_tok))

    aligned = num_steps - len(mismatches)
    alignment_ratio = aligned / num_steps

    detail_lines = []
    for step in range(num_steps):
        efm_tok = got_tokens[step] if step < len(got_tokens) else -1
        ref_tok = int(ref_tokens[step]) if step < len(ref_tokens) else -1
        marker = "✓" if efm_tok == ref_tok else "✗"
        detail_lines.append(f"  step {step}: edge_fm={efm_tok}, ref={ref_tok} {marker}")
    detail = "\n".join(detail_lines)

    print(f"\n[{label}] {aligned}/{num_steps} steps aligned ({alignment_ratio:.0%})")
    print(detail)

    if mismatches:
        mismatch_detail = "\n".join(
            f"  step {s}: edge_fm={e}, ref={r}" for s, e, r in mismatches
        )
        pytest.fail(
            f"{label} token mismatch: {len(mismatches)}/{num_steps} steps mismatch\n{mismatch_detail}"
        )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_token_alignment(engine_and_dump):
    """Edge-FM generate() greedy 输出 token 应与 Transformers dump 完全一致。"""
    import torch

    engine = engine_and_dump["engine"]
    token_ids = engine_and_dump["token_ids"]
    decode_tokens = engine_and_dump["decode_tokens"]
    num_steps = engine_and_dump["num_steps"]

    token_ids_list = token_ids.flatten().tolist()
    ref_tokens = decode_tokens[1: num_steps + 1].tolist()

    request = edge_fm.Request(0, token_ids_list)
    response = engine.generate(request)
    all_tokens = response.token_ids()
    got_tokens = all_tokens[1: 1 + num_steps]

    torch.cuda.synchronize()

    mismatches = []
    for step in range(num_steps):
        efm_tok = got_tokens[step] if step < len(got_tokens) else -1
        ref_tok = int(ref_tokens[step]) if step < len(ref_tokens) else -1
        if efm_tok != ref_tok:
            mismatches.append((step, efm_tok, ref_tok))

    if mismatches:
        detail = "\n".join(f"  step {s}: edge_fm={e}, ref={r}" for s, e, r in mismatches)
        pytest.fail(
            f"Token 不对齐：{len(mismatches)}/{num_steps} 步不一致\n{detail}"
        )


def test_generate_logits_cosine_similarity(engine_and_dump):
    """Edge-FM generate() 各步 logits 的 argmax 应与 Transformers dump 的 argmax 一致。

    由于 generate() 不直接导出 logits，这里通过 token 级别的 argmax 一致性间接验证。
    """
    import torch

    engine = engine_and_dump["engine"]
    token_ids = engine_and_dump["token_ids"]
    decode_tokens = engine_and_dump["decode_tokens"]
    dump_dir = engine_and_dump["dump_dir"]
    num_steps = engine_and_dump["num_steps"]

    token_ids_list = token_ids.flatten().tolist()
    ref_tokens = decode_tokens[1: num_steps + 1].tolist()

    request = edge_fm.Request(0, token_ids_list)
    response = engine.generate(request)
    all_tokens = response.token_ids()
    got_tokens = all_tokens[1: 1 + num_steps]

    torch.cuda.synchronize()

    for step in range(num_steps):
        step_npz = dump_dir / f"step_{step}.npz"
        if not step_npz.exists():
            continue
        ref = np.load(step_npz)
        ref_logits = ref["logits"].astype(np.float32)
        ref_last = ref_logits[-1] if ref_logits.ndim > 1 else ref_logits.ravel()
        ref_argmax = int(np.argmax(ref_last))

        efm_tok = got_tokens[step] if step < len(got_tokens) else -1
        assert efm_tok == ref_argmax, (
            f"step {step}: edge_fm token={efm_tok}, ref argmax={ref_argmax}"
        )


@pytest.mark.parametrize("checkpoint", [5, 10, 15, 20])
def test_generate_checkpoint_alignment(engine_and_dump, checkpoint):
    """在不同步数截断点上验证 token 完全对齐。"""
    import torch

    engine = engine_and_dump["engine"]
    token_ids = engine_and_dump["token_ids"]
    decode_tokens = engine_and_dump["decode_tokens"]
    num_steps = engine_and_dump["num_steps"]

    if checkpoint > num_steps:
        pytest.skip(f"dump 仅有 {num_steps} 步，跳过 checkpoint={checkpoint}")

    token_ids_list = token_ids.flatten().tolist()
    ref_tokens = decode_tokens[1: checkpoint + 1].tolist()

    request = edge_fm.Request(0, token_ids_list)
    response = engine.generate(request)
    all_tokens = response.token_ids()
    got_tokens = all_tokens[1: 1 + checkpoint]

    torch.cuda.synchronize()

    aligned = sum(1 for i in range(checkpoint) if i < len(got_tokens) and got_tokens[i] == int(ref_tokens[i]))
    assert aligned == checkpoint, (
        f"checkpoint={checkpoint}: 仅 {aligned}/{checkpoint} 步对齐"
    )


def test_generate_token_alignment_cuda_graph(engine_and_dump_cuda_graph):
    """启用 CUDA graph decode 时，token 输出应与 Transformers dump 完全一致。"""
    import torch

    engine = engine_and_dump_cuda_graph["engine"]
    token_ids = engine_and_dump_cuda_graph["token_ids"]
    decode_tokens = engine_and_dump_cuda_graph["decode_tokens"]
    num_steps = engine_and_dump_cuda_graph["num_steps"]

    token_ids_list = token_ids.flatten().tolist()
    ref_tokens = decode_tokens[1: num_steps + 1].tolist()

    request = edge_fm.Request(0, token_ids_list)
    response = engine.generate(request)
    all_tokens = response.token_ids()
    got_tokens = all_tokens[1: 1 + num_steps]

    torch.cuda.synchronize()

    mismatches = []
    for step in range(num_steps):
        efm_tok = got_tokens[step] if step < len(got_tokens) else -1
        ref_tok = int(ref_tokens[step]) if step < len(ref_tokens) else -1
        if efm_tok != ref_tok:
            mismatches.append((step, efm_tok, ref_tok))

    if mismatches:
        detail = "\n".join(f"  step {s}: edge_fm={e}, ref={r}" for s, e, r in mismatches)
        pytest.fail(
            f"CUDA graph token 不对齐：{len(mismatches)}/{num_steps} 步不一致\n{detail}"
        )


# ---------------------------------------------------------------------------
# VL（含图）测试
# ---------------------------------------------------------------------------

def test_generate_vl_token_alignment(engine_and_dump_vl):
    """含图请求：Edge-FM generate() 在 VLM 上应与 Transformers dump 完全对齐。"""
    _assert_vl_alignment(engine_and_dump_vl, "VL alignment")



def test_generate_vl_token_alignment_cuda_graph(engine_and_dump_vl_cuda_graph):
    """含图请求：启用 CUDA graph decode 时，VLM token 输出也应与 dump 完全对齐。"""
    _assert_vl_alignment(engine_and_dump_vl_cuda_graph, "VL alignment cuda graph")


# ---------------------------------------------------------------------------
# 性能基准测试
# ---------------------------------------------------------------------------

BENCH_NUM_STEPS = DEFAULT_BENCH_DECODE_LENGTHS[0]
BENCH_WARMUP_RUNS = 3
BENCH_TIMED_RUNS = 5


def _summarize_times_ms(times_ms: list[float]) -> dict:
    xs = list(times_ms)
    trimmed = sorted(xs)[:-1] if len(xs) > 1 else xs
    stdev = stats.stdev(xs) if len(xs) > 1 else 0.0
    mean = stats.mean(xs)
    return {
        "mean_ms": mean,
        "median_ms": stats.median(xs),
        "trimmed_mean_drop_max_ms": stats.mean(trimmed),
        "min_ms": min(xs),
        "max_ms": max(xs),
        "stdev_ms": stdev,
        "cv_pct": (stdev / mean * 100.0) if mean else 0.0,
    }


def _with_latency_summary(result: dict) -> dict:
    out = dict(result)
    out["latency_summary"] = _summarize_times_ms(result["times_ms"])
    return out


def _parse_bench_model_sizes(env_key: str, default_sizes: list[str], valid_sizes: list[str]) -> list[str]:
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return list(default_sizes)
    sizes = []
    for part in raw.split(","):
        size = part.strip().lower()
        if not size:
            continue
        if size not in valid_sizes:
            raise ValueError(f"{env_key} contains unsupported model size: {size} (valid: {valid_sizes})")
        if size not in sizes:
            sizes.append(size)
    return sizes or list(default_sizes)


def _resolve_bench_model_specs(kind: str) -> tuple[list[dict], list[dict]]:
    if kind == "llm":
        env_key = "EDGE_FM_BENCH_LLM_MODELS"
        default_sizes = DEFAULT_BENCH_LLM_MODEL_SIZES
    elif kind == "vlm":
        env_key = "EDGE_FM_BENCH_VLM_MODELS"
        default_sizes = DEFAULT_BENCH_VLM_MODEL_SIZES
    else:
        raise ValueError(f"Unsupported benchmark kind: {kind}")

    valid_sizes = list(BENCH_MODEL_SPECS[kind].keys())
    requested_sizes = _parse_bench_model_sizes(env_key, default_sizes, valid_sizes)

    resolved = []
    missing = []
    for size in requested_sizes:
        spec = dict(BENCH_MODEL_SPECS[kind][size])
        path = _find_bench_model_path(kind, size)
        entry = {**spec, "kind": kind, "model_size": size, "model_path": path}
        if path is None:
            missing.append(entry)
        else:
            resolved.append(entry)
    return resolved, missing


def _print_missing_bench_models(kind: str, missing: list[dict]) -> None:
    if not missing:
        return
    print(f"\n[benchmark] skipping missing {kind} models:")
    for item in missing:
        print(f"  - {item['label']} ({item['model_size']}): model weights not found")


def _resolve_trt_requested_engine_dir(model_size: str) -> Path | None:
    spec = LLM_MODEL_SPECS[model_size]
    for env_key in spec.get("trt_engine_env_keys", []):
        value = os.environ.get(env_key, "").strip()
        if value:
            return Path(value).resolve()
    return None


def _resolve_trt_requested_vlm_engine_dir(model_size: str) -> Path | None:
    spec = VLM_MODEL_SPECS[model_size]
    for env_key in spec.get("trt_engine_env_keys", []):
        value = os.environ.get(env_key, "").strip()
        if value:
            return Path(value).resolve()
    return None


def _resolve_trt_requested_vlm_multimodal_engine_dir(model_size: str) -> Path | None:
    spec = VLM_MODEL_SPECS[model_size]
    for env_key in spec.get("trt_multimodal_engine_env_keys", []):
        value = os.environ.get(env_key, "").strip()
        if value:
            return Path(value).resolve()
    return None


def _parse_bench_lengths(env_key: str) -> list[int] | None:
    """Parse comma-separated positive integer list from env, e.g. '256,512,1024'."""
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return None
    vals = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        n = int(s)
        if n <= 0:
            raise ValueError(f"{env_key} must contain positive integers, got {n}")
        vals.append(n)
    return vals or None


def _build_prefill_token_ids(token_ids_list: list[int], prefill_len: int) -> list[int]:
    """Resize token_ids to target prefill length by repeating the base sequence."""
    if prefill_len <= 0:
        raise ValueError(f"prefill_len must be > 0, got {prefill_len}")
    if not token_ids_list:
        raise ValueError("token_ids_list is empty")
    if len(token_ids_list) >= prefill_len:
        return token_ids_list[:prefill_len]
    mul = (prefill_len + len(token_ids_list) - 1) // len(token_ids_list)
    expanded = (token_ids_list * mul)[:prefill_len]
    return expanded


def _build_llm_bench_token_ids(tokenizer, prefill_len: int, prompt: str | None = None) -> list[int]:
    bench_prompt = prompt or os.environ.get("EDGE_FM_BENCH_PROMPT", DEFAULT_PROMPT)
    base_token_ids = tokenizer.encode(bench_prompt, add_special_tokens=True)
    return _build_prefill_token_ids(base_token_ids, prefill_len)


def _resolve_bench_cases(default_prefill: int, default_decode: int) -> list[tuple[int, int]]:
    """Resolve benchmark cases from env.

    - EDGE_FM_BENCH_PREFILL_LIST: comma-separated prefill lengths
    - EDGE_FM_BENCH_DECODE_LIST: comma-separated decode lengths
    By default run the long-prefill matrix used for optimization:
    prefill in {512, 1024, 2048}, decode in {32, 64}.
    """
    p_list = _parse_bench_lengths("EDGE_FM_BENCH_PREFILL_LIST")
    d_list = _parse_bench_lengths("EDGE_FM_BENCH_DECODE_LIST")
    if p_list is None:
        p_list = list(DEFAULT_BENCH_PREFILL_LENGTHS)
    if d_list is None:
        d_list = list(DEFAULT_BENCH_DECODE_LENGTHS)
    return [(p, d) for p in p_list for d in d_list]


def _default_trt_workspace_dir(model_size: str = "1.5b") -> Path:
    workspace_name = LLM_MODEL_SPECS[model_size]["trt_workspace_name"]
    return (project_root / "tests" / "data" / "trt_edgellm_workspace" / workspace_name).resolve()


def _default_trt_vlm_workspace_dir(model_size: str = "3b") -> Path:
    workspace_name = VLM_MODEL_SPECS[model_size]["trt_workspace_name"]
    return (project_root / "tests" / "data" / "trt_edgellm_workspace" / workspace_name).resolve()


def _load_trt_engine_config(engine_dir: Path) -> dict:
    config_path = engine_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"TRT-Edge-LLM config not found at {config_path}")
    return json.loads(config_path.read_text())


def _trt_engine_max_input_len(engine_dir: Path) -> int:
    builder_config = _load_trt_engine_config(engine_dir).get("builder_config", {})
    return int(builder_config.get("max_input_len", 0) or 0)


def _resolve_trt_engine_dir(
    required_prefill_len: int,
    requested_engine_dir: Path | None = None,
    model_size: str = "1.5b",
) -> Path:
    if required_prefill_len <= 0:
        raise ValueError(f"required_prefill_len must be > 0, got {required_prefill_len}")

    if requested_engine_dir is not None:
        engine_dir = requested_engine_dir.resolve()
        if not engine_dir.exists() or not (engine_dir / "llm.engine").exists():
            raise FileNotFoundError(f"TRT-Edge-LLM engine not found at {engine_dir}")
        max_input_len = _trt_engine_max_input_len(engine_dir)
        if max_input_len and required_prefill_len > max_input_len:
            raise RuntimeError(
                f"TRT-Edge-LLM engine at {engine_dir} only supports max_input_len={max_input_len}, "
                f"but required prefill_len={required_prefill_len}"
            )
        return engine_dir

    workspace_dir = _default_trt_workspace_dir(model_size)
    candidate_dirs = []
    for name in ["engines_mxil2048", "engines"]:
        engine_dir = workspace_dir / name
        if engine_dir.exists() and (engine_dir / "llm.engine").exists():
            candidate_dirs.append(engine_dir.resolve())

    if not candidate_dirs:
        raise FileNotFoundError(f"No TRT-Edge-LLM engine directory found under {workspace_dir}")

    supported_dirs = []
    for engine_dir in candidate_dirs:
        max_input_len = _trt_engine_max_input_len(engine_dir)
        if max_input_len >= required_prefill_len:
            supported_dirs.append((max_input_len, engine_dir))

    if not supported_dirs:
        supported = ", ".join(
            f"{engine_dir.name}(max_input_len={_trt_engine_max_input_len(engine_dir)})"
            for engine_dir in candidate_dirs
        )
        raise RuntimeError(
            f"No TRT-Edge-LLM engine supports prefill_len={required_prefill_len}. "
            f"Available engines: {supported}"
        )

    supported_dirs.sort(key=lambda item: item[0])
    return supported_dirs[0][1]


def _resolve_trt_vlm_multimodal_engine_dir(
    workspace_dir: Path,
    requested_multimodal_engine_dir: Path | None = None,
) -> Path:
    if requested_multimodal_engine_dir is not None:
        engine_dir = requested_multimodal_engine_dir.resolve()
        if not engine_dir.exists() or not (engine_dir / "visual.engine").exists():
            raise FileNotFoundError(f"TRT-Edge-LLM multimodal engine not found at {engine_dir}")
        return engine_dir

    candidate_dirs = []
    for name in [
        "visual_engines_mxil2048",
        "visual_engines",
        "multimodal_engines_mxil2048",
        "multimodal_engines",
    ]:
        engine_dir = workspace_dir / name
        if engine_dir.exists() and (engine_dir / "visual.engine").exists():
            candidate_dirs.append(engine_dir.resolve())

    if not candidate_dirs:
        raise FileNotFoundError(f"No TRT-Edge-LLM multimodal engine directory found under {workspace_dir}")

    return candidate_dirs[0]


def _resolve_trt_vlm_engine_dirs(
    required_prefill_len: int,
    requested_engine_dir: Path | None = None,
    requested_multimodal_engine_dir: Path | None = None,
    model_size: str = "3b",
) -> tuple[Path, Path]:
    workspace_dir = _default_trt_vlm_workspace_dir(model_size)
    if requested_engine_dir is not None:
        engine_dir = requested_engine_dir.resolve()
        if not engine_dir.exists() or not (engine_dir / "llm.engine").exists():
            raise FileNotFoundError(f"TRT-Edge-LLM engine not found at {engine_dir}")
        max_input_len = _trt_engine_max_input_len(engine_dir)
        if max_input_len and required_prefill_len > max_input_len:
            raise RuntimeError(
                f"TRT-Edge-LLM engine at {engine_dir} only supports max_input_len={max_input_len}, "
                f"but required prefill_len={required_prefill_len}"
            )
    else:
        candidate_dirs = []
        for name in ["engines_mxil2048", "engines"]:
            cur = workspace_dir / name
            if cur.exists() and (cur / "llm.engine").exists():
                candidate_dirs.append(cur.resolve())
        if not candidate_dirs:
            raise FileNotFoundError(f"No TRT-Edge-LLM engine directory found under {workspace_dir}")
        supported_dirs = []
        for cur in candidate_dirs:
            max_input_len = _trt_engine_max_input_len(cur)
            if max_input_len >= required_prefill_len:
                supported_dirs.append((max_input_len, cur))
        if not supported_dirs:
            supported = ", ".join(
                f"{cur.name}(max_input_len={_trt_engine_max_input_len(cur)})"
                for cur in candidate_dirs
            )
            raise RuntimeError(
                f"No TRT-Edge-LLM VLM engine supports prefill_len={required_prefill_len}. "
                f"Available engines: {supported}"
            )
        supported_dirs.sort(key=lambda item: item[0])
        engine_dir = supported_dirs[0][1]

    multimodal_engine_dir = _resolve_trt_vlm_multimodal_engine_dir(
        workspace_dir, requested_multimodal_engine_dir=requested_multimodal_engine_dir
    )
    return engine_dir, multimodal_engine_dir


def _load_transformers_llm_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(model_path)
    torch_dtype_str = str(getattr(config, "torch_dtype", "float16")).lower()
    model_dtype = torch.bfloat16 if "bfloat" in torch_dtype_str or "bf16" in torch_dtype_str else torch.float16
    device = CUDA_DEVICE if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=model_dtype, low_cpu_mem_usage=False)
    model = model.to(device)
    model.eval()
    return model


def _bench_transformers_llm_loaded(model, token_ids_list: list[int], num_steps: int,
                                   warmup: int, runs: int) -> dict:
    """Benchmark Transformers KV-cached greedy decode for LLM using a preloaded model."""
    import time
    import torch

    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids_list], dtype=torch.long, device=device)
    prefill_len = len(token_ids_list)

    def run_once():
        with torch.no_grad():
            out = model(input_ids, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        tok = out.logits[0, -1].argmax().item()
        decode_input = torch.tensor([[tok]], dtype=torch.long, device=device)
        for _ in range(num_steps - 1):
            with torch.no_grad():
                out = model(decode_input, past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            tok = out.logits[0, -1].argmax().item()
            decode_input = torch.tensor([[tok]], dtype=torch.long, device=device)

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    return _with_latency_summary({
        "avg_ms": avg * 1000,
        "prefill_tokens": prefill_len,
        "decode_tokens": num_steps,
        "total_tokens": prefill_len + num_steps,
        "tokens_per_sec": (prefill_len + num_steps) / avg,
        "decode_tokens_per_sec": num_steps / avg,
        "times_ms": [t * 1000 for t in times],
    })


def _bench_transformers_llm(model_path: str, token_ids_list: list[int], num_steps: int,
                            warmup: int, runs: int) -> dict:
    """Benchmark Transformers KV-cached greedy decode for LLM."""
    import torch

    model = _load_transformers_llm_model(model_path)
    try:
        return _bench_transformers_llm_loaded(model, token_ids_list, num_steps, warmup, runs)
    finally:
        del model
        torch.cuda.empty_cache()


def _load_transformers_vlm_model(model_path: str):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=False,
    )
    model = model.to(CUDA_DEVICE if torch.cuda.is_available() else "cpu")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 50176
    return model, processor


def _extend_vlm_input_ids(input_ids, image_token_id: int, target_prefill_len: int):
    import torch

    base_ids = input_ids[0].tolist()
    if len(base_ids) > target_prefill_len:
        raise ValueError(
            f"Base VLM prefill length {len(base_ids)} exceeds target prefill length {target_prefill_len}"
        )
    if len(base_ids) == target_prefill_len:
        return input_ids.clone()

    image_positions = [i for i, token_id in enumerate(base_ids) if token_id == image_token_id]
    if not image_positions:
        raise RuntimeError(f"Image token id {image_token_id} not found in VLM input_ids")

    tail_tokens = [tok for tok in base_ids[image_positions[-1] + 1:] if tok != image_token_id]
    if not tail_tokens:
        tail_tokens = [tok for tok in base_ids if tok != image_token_id]
    if not tail_tokens:
        raise RuntimeError("Unable to extend VLM benchmark input: no text-only tail tokens available")

    extended = list(base_ids)
    while len(extended) < target_prefill_len:
        need = target_prefill_len - len(extended)
        extended.extend(tail_tokens[:need])
    return torch.tensor([extended], dtype=input_ids.dtype)


def _prepare_vlm_bench_case(model, processor, prefill_len: int,
                            image_path: str | None = None,
                            prompt: str | None = None) -> dict:
    import inspect
    import torch
    from PIL import Image

    image_path = image_path or os.environ.get("EDGE_FM_BENCH_VLM_IMAGE_PATH", str(DEFAULT_VLM_IMAGE_PATH))
    prompt = prompt or os.environ.get("EDGE_FM_BENCH_VLM_PROMPT", DEFAULT_VLM_PROMPT)

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)

    vocab_size = model.config.text_config.vocab_size
    embed_token_id = getattr(
        model.config,
        "image_token_id",
        getattr(model.config.text_config, "image_token_id", vocab_size),
    )

    input_ids = _extend_vlm_input_ids(inputs["input_ids"], embed_token_id, prefill_len).to(CUDA_DEVICE)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(input_ids.device)
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(input_ids.device)

    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids.to(dtype=torch.int32)
        if mm_token_type_ids.shape[1] < prefill_len:
            extra = torch.zeros(
                (mm_token_type_ids.shape[0], prefill_len - mm_token_type_ids.shape[1]),
                dtype=mm_token_type_ids.dtype,
            )
            mm_token_type_ids = torch.cat([mm_token_type_ids, extra], dim=1)
        mm_token_type_ids = mm_token_type_ids[:, :prefill_len].to(input_ids.device)
    else:
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32, device=input_ids.device)
        mm_token_type_ids[input_ids == embed_token_id] = 1

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    embed_output = hidden_states[0]
    positions = [i for i in range(input_ids.shape[1]) if input_ids[0, i].item() == embed_token_id]
    if not positions:
        raise RuntimeError(f"Unable to locate image token positions for embed_token_id={embed_token_id}")
    image_embeddings = embed_output[0, positions, :].float().cpu().numpy()

    rope_sig = inspect.signature(model.model.get_rope_index)
    rope_kwargs = {
        "input_ids": input_ids,
        "image_grid_thw": image_grid_thw,
    }
    if "mm_token_type_ids" in rope_sig.parameters:
        rope_kwargs["mm_token_type_ids"] = mm_token_type_ids
    if "attention_mask" in rope_sig.parameters:
        rope_kwargs["attention_mask"] = attention_mask
    position_ids_3d, rope_deltas = model.model.get_rope_index(**rope_kwargs)
    position_ids_np = position_ids_3d[:, 0, :].cpu().numpy().astype(np.int32)

    edgefm_token_ids = input_ids[0].cpu().numpy().astype(np.int32)
    embed_counter = 0
    for i in range(len(edgefm_token_ids)):
        if edgefm_token_ids[i] == embed_token_id:
            edgefm_token_ids[i] = embed_token_id + embed_counter
            embed_counter += 1

    return {
        "prompt": prompt,
        "image_path": str(Path(image_path).resolve()),
        "prefill_tokens": int(input_ids.shape[1]),
        "input_ids": input_ids.detach().cpu(),
        "pixel_values": pixel_values.detach().cpu() if pixel_values is not None else None,
        "image_grid_thw": image_grid_thw.detach().cpu() if image_grid_thw is not None else None,
        "edgefm_token_ids": edgefm_token_ids.tolist(),
        "image_embeddings": image_embeddings.astype(np.float32),
        "embed_token_id": int(embed_token_id),
        "position_ids": position_ids_np,
        "rope_deltas": rope_deltas.detach().cpu(),
    }


def _trt_vlm_image_token_base(model_path: str) -> int:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"VLM config not found at {config_path}")
    config = json.loads(config_path.read_text())
    return int(config.get("vocab_size") or config.get("text_config", {}).get("vocab_size"))


def _build_trt_vlm_token_ids(prepared_inputs: dict, model_path: str) -> list[int]:
    trt_image_token_base = _trt_vlm_image_token_base(model_path)
    image_token_id = int(prepared_inputs["embed_token_id"])
    input_ids = prepared_inputs["input_ids"][0].tolist()

    trt_token_ids = []
    image_counter = 0
    for token_id in input_ids:
        if token_id == image_token_id:
            trt_token_ids.append(trt_image_token_base + image_counter)
            image_counter += 1
        else:
            trt_token_ids.append(int(token_id))
    return trt_token_ids


def _bench_transformers_vlm_loaded(model, prepared_inputs: dict,
                                   num_steps: int, warmup: int, runs: int) -> dict:
    """Benchmark Transformers KV-cached greedy decode for VLM using prepared image embeddings.

    This intentionally excludes visual encoder time so the timed region matches
    EdgeFM/TRT prepared-multimodal benchmarks.
    """
    import time
    import torch

    input_ids = prepared_inputs["input_ids"].to(CUDA_DEVICE)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=CUDA_DEVICE)
    image_grid_thw = prepared_inputs["image_grid_thw"].to(CUDA_DEVICE)
    image_embeddings = torch.from_numpy(prepared_inputs["image_embeddings"]).to(
        device=CUDA_DEVICE, dtype=model.dtype
    )
    embed_token_id = int(prepared_inputs["embed_token_id"])
    prefill_len = input_ids.shape[1]

    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings()(input_ids)
    image_positions = (input_ids[0] == embed_token_id).nonzero(as_tuple=False).flatten()
    if image_positions.numel() != image_embeddings.shape[0]:
        raise RuntimeError(
            "Prepared image embedding count does not match the number of image placeholder tokens "
            f"({image_embeddings.shape[0]} vs {image_positions.numel()})"
        )
    inputs_embeds[0, image_positions, :] = image_embeddings

    prefill_position_ids = torch.from_numpy(prepared_inputs["position_ids"]).to(
        device=CUDA_DEVICE, dtype=torch.long
    ).unsqueeze(1)
    rope_deltas = prepared_inputs["rope_deltas"].to(device=CUDA_DEVICE, dtype=torch.long)

    def run_once():
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                position_ids=prefill_position_ids,
                use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        tok = out.logits[0, -1].argmax().item()
        decode_input = torch.tensor([[tok]], dtype=torch.long, device=CUDA_DEVICE)
        cl = prefill_len
        for _ in range(num_steps - 1):
            cache_position = torch.arange(cl, cl + 1, dtype=torch.long, device=CUDA_DEVICE)
            mrope_pos = (cache_position + rope_deltas).view(1, 1, 1).expand(3, 1, 1).to(torch.long)
            text_pos = cache_position.view(1, 1, 1)
            position_ids = torch.cat([text_pos, mrope_pos], dim=0)
            with torch.no_grad():
                out = model(input_ids=decode_input, past_key_values=past_kv,
                            position_ids=position_ids, cache_position=cache_position,
                            use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            tok = out.logits[0, -1].argmax().item()
            decode_input = torch.tensor([[tok]], dtype=torch.long, device=CUDA_DEVICE)
            cl += 1

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    return _with_latency_summary({
        "avg_ms": avg * 1000,
        "prefill_tokens": prefill_len,
        "decode_tokens": num_steps,
        "total_tokens": prefill_len + num_steps,
        "tokens_per_sec": (prefill_len + num_steps) / avg,
        "decode_tokens_per_sec": num_steps / avg,
        "times_ms": [t * 1000 for t in times],
    })


def _bench_transformers_vlm(model_path: str, token_ids_list: list[int],
                            image_path: str, num_steps: int,
                            warmup: int, runs: int) -> dict:
    """Benchmark Transformers KV-cached greedy decode for VLM."""
    import torch

    model, processor = _load_transformers_vlm_model(model_path)
    try:
        prepared_inputs = _prepare_vlm_bench_case(
            model,
            processor,
            prefill_len=len(token_ids_list),
            image_path=image_path,
            prompt=os.environ.get("EDGE_FM_BENCH_VLM_PROMPT", DEFAULT_VLM_PROMPT),
        )
        return _bench_transformers_vlm_loaded(model, prepared_inputs, num_steps, warmup, runs)
    finally:
        del model
        torch.cuda.empty_cache()


def _bench_edgefm(engine, request_fn, num_steps: int, prefill_len: int,
                  warmup: int, runs: int) -> dict:
    """Benchmark EdgeFM engine.generate().

    The benchmark is only valid when each run produces the expected number of
    generated tokens. If CUDA graph or stop-token handling causes early exit,
    fail fast instead of reporting inflated throughput.
    """
    import torch
    import time

    warmup_counts = []
    for _ in range(warmup):
        response = engine.generate(request_fn())
        warmup_counts.append(len(response.token_ids()))
    torch.cuda.synchronize()

    if any(count != num_steps for count in warmup_counts):
        raise AssertionError(
            f"EdgeFM warmup generated unexpected token counts: {warmup_counts}, "
            f"expected every run to generate {num_steps} tokens"
        )

    times = []
    generated_counts = []
    stage_times = {
        "prefill_ms": [],
        "decode_ms": [],
        "total_stage_ms": [],
        "decode_step_avg_ms": [],
    }
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        response = engine.generate(request_fn())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        generated_counts.append(len(response.token_ids()))
        metrics = engine.last_generate_metrics()
        for key in stage_times:
            stage_times[key].append(float(metrics.get(key, 0.0)))

    if any(count != num_steps for count in generated_counts):
        raise AssertionError(
            f"EdgeFM timed runs generated unexpected token counts: {generated_counts}, "
            f"expected every run to generate {num_steps} tokens"
        )

    avg = sum(times) / len(times)
    actual_decode_tokens = generated_counts[0] if generated_counts else num_steps
    total_tokens = prefill_len + actual_decode_tokens
    stage_avg_ms = {
        key: (sum(values) / len(values) if values else 0.0)
        for key, values in stage_times.items()
    }
    return _with_latency_summary({
        "avg_ms": avg * 1000,
        "prefill_tokens": prefill_len,
        "decode_tokens": actual_decode_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / avg,
        "decode_tokens_per_sec": actual_decode_tokens / avg,
        "times_ms": [t * 1000 for t in times],
        "generated_counts": generated_counts,
        "stage_times_ms": stage_times,
        "stage_avg_ms": stage_avg_ms,
    })


def _print_bench_comparison(label: str, tf_result: dict, efm_result: dict):
    """Pretty-print benchmark comparison table."""
    print(f"\n{'='*70}")
    print(f"  Performance Benchmark: {label}")
    print(f"  prefill={tf_result['prefill_tokens']} tokens, "
          f"decode={tf_result['decode_tokens']} tokens")
    print(f"{'='*70}")
    print(f"  {'Metric':<30} {'Transformers':>15} {'EdgeFM':>15} {'Speedup':>10}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*10}")

    rows = [
        ("Total latency (ms)", "avg_ms", ".1f"),
        ("Total throughput (tok/s)", "tokens_per_sec", ".1f"),
        ("Decode throughput (tok/s)", "decode_tokens_per_sec", ".1f"),
    ]
    for name, key, fmt in rows:
        tf_val = tf_result[key]
        efm_val = efm_result[key]
        if "latency" in name.lower():
            speedup = tf_val / efm_val if efm_val > 0 else float("inf")
        else:
            speedup = efm_val / tf_val if tf_val > 0 else float("inf")
        print(f"  {name:<30} {tf_val:>15{fmt}} {efm_val:>15{fmt}} {speedup:>9.2f}x")

    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*10}")
    tf_runs = ", ".join(f"{t:.1f}" for t in tf_result["times_ms"])
    efm_runs = ", ".join(f"{t:.1f}" for t in efm_result["times_ms"])
    print(f"  Transformers runs (ms): [{tf_runs}]")
    print(f"  EdgeFM runs (ms):       [{efm_runs}]")
    tf_sum = tf_result.get("latency_summary", _summarize_times_ms(tf_result["times_ms"]))
    efm_sum = efm_result.get("latency_summary", _summarize_times_ms(efm_result["times_ms"]))
    print("  Transformers latency summary: "
          f"median={tf_sum['median_ms']:.1f} trimmed={tf_sum['trimmed_mean_drop_max_ms']:.1f} cv={tf_sum['cv_pct']:.2f}%")
    print("  EdgeFM latency summary:       "
          f"median={efm_sum['median_ms']:.1f} trimmed={efm_sum['trimmed_mean_drop_max_ms']:.1f} cv={efm_sum['cv_pct']:.2f}%")
    print(f"{'='*70}")


def _bench_trt_edgellm(
    engine_dir: Path,
    plugin_path: Path,
    token_ids_list: list[int],
    num_steps: int,
    prefill_len: int,
    warmup: int,
    runs: int,
    ignore_stop_tokens: bool = False,
) -> dict:
    """Benchmark TRT-Edge-LLM via edge_fm_trt (in-process only)."""
    if edge_fm_trt is None:
        raise RuntimeError(
            "edge_fm_trt not available. Build with BUILD_TRT_EDGELLM_PYBIND=ON "
            "and add its module path to PYTHONPATH."
        )
    return _bench_trt_edgellm_inprocess(
        engine_dir, plugin_path, token_ids_list, num_steps, prefill_len, warmup, runs, ignore_stop_tokens
    )


def _bench_trt_edgellm_inprocess(
    engine_dir: Path,
    plugin_path: Path,
    token_ids_list: list[int],
    num_steps: int,
    prefill_len: int,
    warmup: int,
    runs: int,
    ignore_stop_tokens: bool = False,
) -> dict:
    """Benchmark TRT-Edge-LLM via edge_fm_trt (in-process). Uses token_ids for same prefill as Edge-FM."""
    import time

    if not plugin_path.exists():
        raise FileNotFoundError(f"TRT-Edge-LLM plugin not found at {plugin_path}")
    os.environ["EDGELLM_PLUGIN_PATH"] = str(plugin_path)

    max_input_len = _trt_engine_max_input_len(engine_dir)
    if max_input_len and prefill_len > max_input_len:
        raise RuntimeError(
            f"TRT-Edge-LLM engine at {engine_dir} only supports max_input_len={max_input_len}, "
            f"but benchmark requested prefill_len={prefill_len}"
        )

    runtime = edge_fm_trt.TrtEdgeLlmRuntime(
        str(engine_dir), "", DEVICE_ID
    )

    # Warmup
    warmup_counts = []
    for _ in range(warmup):
        output_ids, _ = runtime.generate_from_token_ids(
            token_ids_list, num_steps, temperature=0.0, top_p=1.0, top_k=1, ignore_stop_tokens=ignore_stop_tokens
        )
        warmup_counts.append(len(output_ids[0]) if output_ids else 0)

    if any(count != num_steps for count in warmup_counts):
        raise AssertionError(
            f"TRT-Edge-LLM warmup generated unexpected token counts: {warmup_counts}, "
            f"expected every run to generate {num_steps} tokens"
        )

    # Timed runs
    times = []
    generated_counts = []
    stage_times = {
        "prefill_ms": [],
        "decode_ms": [],
        "total_stage_ms": [],
        "decode_step_avg_ms": [],
    }
    for _ in range(runs):
        t0 = time.perf_counter()
        output_ids, _ = runtime.generate_from_token_ids(
            token_ids_list, num_steps, temperature=0.0, top_p=1.0, top_k=1, ignore_stop_tokens=ignore_stop_tokens
        )
        times.append((time.perf_counter() - t0) * 1000)
        generated_counts.append(len(output_ids[0]) if output_ids else 0)
        metrics = runtime.last_generate_metrics()
        for key in stage_times:
            stage_times[key].append(float(metrics.get(key, 0.0)))

    if any(count != num_steps for count in generated_counts):
        raise AssertionError(
            f"TRT-Edge-LLM timed runs generated unexpected token counts: {generated_counts}, "
            f"expected every run to generate {num_steps} tokens"
        )

    avg_ms = sum(times) / len(times)
    avg_s = avg_ms / 1000.0
    actual_decode_tokens = generated_counts[0] if generated_counts else num_steps
    total_tokens = prefill_len + actual_decode_tokens
    stage_avg_ms = {
        key: (sum(values) / len(values) if values else 0.0)
        for key, values in stage_times.items()
    }
    return _with_latency_summary({
        "avg_ms": avg_ms,
        "prefill_tokens": prefill_len,
        "decode_tokens": actual_decode_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / avg_s,
        "decode_tokens_per_sec": actual_decode_tokens / avg_s,
        "times_ms": times,
        "generated_counts": generated_counts,
        "stage_times_ms": stage_times,
        "stage_avg_ms": stage_avg_ms,
    })


def _bench_trt_edgellm_vlm_prepared(
    runtime,
    token_ids_list: list[int],
    prepared_inputs: dict,
    num_steps: int,
    prefill_len: int,
    warmup: int,
    runs: int,
    ignore_stop_tokens: bool = False,
) -> dict:
    import time

    image_grid_thw = prepared_inputs.get("image_grid_thw")
    if image_grid_thw is None:
        raise RuntimeError("prepared_inputs.image_grid_thw is required for TRT VLM benchmarking")

    runtime.prepare_multimodal_from_token_ids(
        token_ids_list,
        prepared_inputs["image_embeddings"],
        image_grid_thw.tolist(),
    )

    warmup_counts = []
    for _ in range(warmup):
        output_ids, _ = runtime.generate_from_prepared_multimodal(
            num_steps, temperature=0.0, top_p=1.0, top_k=1, ignore_stop_tokens=ignore_stop_tokens
        )
        warmup_counts.append(len(output_ids[0]) if output_ids else 0)

    if any(count != num_steps for count in warmup_counts):
        raise AssertionError(
            f"TRT-Edge-LLM VLM warmup generated unexpected token counts: {warmup_counts}, "
            f"expected every run to generate {num_steps} tokens"
        )

    times = []
    generated_counts = []
    stage_times = {
        "prefill_ms": [],
        "decode_ms": [],
        "total_stage_ms": [],
        "decode_step_avg_ms": [],
    }
    for _ in range(runs):
        t0 = time.perf_counter()
        output_ids, _ = runtime.generate_from_prepared_multimodal(
            num_steps, temperature=0.0, top_p=1.0, top_k=1, ignore_stop_tokens=ignore_stop_tokens
        )
        times.append((time.perf_counter() - t0) * 1000)
        generated_counts.append(len(output_ids[0]) if output_ids else 0)
        metrics = runtime.last_generate_metrics()
        for key in stage_times:
            stage_times[key].append(float(metrics.get(key, 0.0)))

    if any(count != num_steps for count in generated_counts):
        raise AssertionError(
            f"TRT-Edge-LLM VLM timed runs generated unexpected token counts: {generated_counts}, "
            f"expected every run to generate {num_steps} tokens"
        )

    avg_ms = sum(times) / len(times)
    avg_s = avg_ms / 1000.0
    actual_decode_tokens = generated_counts[0] if generated_counts else num_steps
    total_tokens = prefill_len + actual_decode_tokens
    stage_avg_ms = {
        key: (sum(values) / len(values) if values else 0.0)
        for key, values in stage_times.items()
    }
    return _with_latency_summary({
        "avg_ms": avg_ms,
        "prefill_tokens": prefill_len,
        "decode_tokens": actual_decode_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / avg_s,
        "decode_tokens_per_sec": actual_decode_tokens / avg_s,
        "times_ms": times,
        "generated_counts": generated_counts,
        "stage_times_ms": stage_times,
        "stage_avg_ms": stage_avg_ms,
    })


def _print_bench_comparison_3way(
    label: str,
    tf_result: dict,
    efm_result: dict,
    trt_result: dict,
):
    """Pretty-print 3-way benchmark: Transformers vs EdgeFM vs TRT-Edge-LLM."""
    print(f"\n{'='*90}")
    print(f"  Performance Benchmark: {label}")
    print(f"  prefill={tf_result['prefill_tokens']} tokens, decode={tf_result['decode_tokens']} tokens")
    print(f"{'='*90}")
    print(f"  {'Metric':<28} {'Transformers':>14} {'EdgeFM':>14} {'TRT-Edge-LLM':>14} {'Speedup':>10}")
    print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")

    rows = [
        ("Total latency (ms)", "avg_ms", ".1f"),
        ("Total throughput (tok/s)", "tokens_per_sec", ".1f"),
        ("Decode throughput (tok/s)", "decode_tokens_per_sec", ".1f"),
    ]
    for name, key, fmt in rows:
        tf_val = tf_result[key]
        efm_val = efm_result[key]
        trt_val = trt_result[key]
        speedup = (tf_val / trt_val if "latency" in name.lower() else trt_val / tf_val) if (tf_val > 0 and trt_val > 0) else float("inf")
        print(f"  {name:<28} {tf_val:>14{fmt}} {efm_val:>14{fmt}} {trt_val:>14{fmt}} {speedup:>9.2f}x")

    print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")
    print(f"  Transformers runs (ms): [{', '.join(f'{t:.1f}' for t in tf_result['times_ms'])}]")
    print(f"  EdgeFM runs (ms):       [{', '.join(f'{t:.1f}' for t in efm_result['times_ms'])}]")
    print(f"  TRT-Edge-LLM runs (ms): [{', '.join(f'{t:.1f}' for t in trt_result['times_ms'])}]")
    tf_sum = tf_result.get("latency_summary", _summarize_times_ms(tf_result["times_ms"]))
    efm_sum = efm_result.get("latency_summary", _summarize_times_ms(efm_result["times_ms"]))
    trt_sum = trt_result.get("latency_summary", _summarize_times_ms(trt_result["times_ms"]))
    print("  Transformers latency summary: "
          f"median={tf_sum['median_ms']:.1f} trimmed={tf_sum['trimmed_mean_drop_max_ms']:.1f} cv={tf_sum['cv_pct']:.2f}%")
    print("  EdgeFM latency summary:       "
          f"median={efm_sum['median_ms']:.1f} trimmed={efm_sum['trimmed_mean_drop_max_ms']:.1f} cv={efm_sum['cv_pct']:.2f}%")
    print("  TRT-Edge-LLM latency summary: "
          f"median={trt_sum['median_ms']:.1f} trimmed={trt_sum['trimmed_mean_drop_max_ms']:.1f} cv={trt_sum['cv_pct']:.2f}%")
    if efm_result.get("stage_avg_ms") or trt_result.get("stage_avg_ms"):
        efm_stage = efm_result.get("stage_avg_ms", {})
        trt_stage = trt_result.get("stage_avg_ms", {})
        print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")
        print("  Stage avg latency (ms):")
        print(
            "  "
            f"{'Prefill':<28} {'-':>14} "
            f"{efm_stage.get('prefill_ms', 0.0):>14.1f} "
            f"{trt_stage.get('prefill_ms', 0.0):>14.1f} "
            f"{(efm_stage.get('prefill_ms', 0.0) - trt_stage.get('prefill_ms', 0.0)):>9.1f}"
        )
        print(
            "  "
            f"{'Decode':<28} {'-':>14} "
            f"{efm_stage.get('decode_ms', 0.0):>14.1f} "
            f"{trt_stage.get('decode_ms', 0.0):>14.1f} "
            f"{(efm_stage.get('decode_ms', 0.0) - trt_stage.get('decode_ms', 0.0)):>9.1f}"
        )
        print(
            "  "
            f"{'Stage total':<28} {'-':>14} "
            f"{efm_stage.get('total_stage_ms', 0.0):>14.1f} "
            f"{trt_stage.get('total_stage_ms', 0.0):>14.1f} "
            f"{(efm_stage.get('total_stage_ms', 0.0) - trt_stage.get('total_stage_ms', 0.0)):>9.1f}"
        )
    print(f"{'='*90}")


def _benchmark_llm_model(model_spec: dict, include_trt: bool = False) -> list[dict]:
    import torch
    from transformers import AutoTokenizer

    model_path = model_spec["model_path"]
    bench_cases = _resolve_bench_cases(DEFAULT_SEQ_LEN, BENCH_NUM_STEPS)
    plugin_path = None
    engine_dir = None

    if include_trt:
        requested_engine_dir = _resolve_trt_requested_engine_dir(model_spec["model_size"])
        try:
            engine_dir = _resolve_trt_engine_dir(
                max(prefill for prefill, _ in bench_cases),
                requested_engine_dir=requested_engine_dir,
                model_size=model_spec["model_size"],
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"\n[benchmark] skipping TRT for {model_spec['label']}: {exc}")
            return []

        _plugin = os.environ.get("TRT_EDGELLM_PLUGIN_PATH", "").strip()
        plugin_path = (
            Path(_plugin).resolve()
            if _plugin
            else (project_root / "third_party" / "TensorRT-Edge-LLM" / "build" / "libNvInfer_edgellm_plugin.so").resolve()
        )
        if not plugin_path.exists():
            print(f"\n[benchmark] skipping TRT for {model_spec['label']}: plugin missing at {plugin_path}")
            return []
        if edge_fm_trt is None:
            print(f"\n[benchmark] skipping TRT for {model_spec['label']}: edge_fm_trt module unavailable")
            return []

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tf_model = _load_transformers_llm_model(model_path)

    print(f"\n[benchmark] {model_spec['label']} cases:")
    for p, d in bench_cases:
        print(f"  - prefill={p}, decode={d}")

    reports = []
    try:
        for prefill_len, num_steps in bench_cases:
            token_ids_list = _build_llm_bench_token_ids(tokenizer, prefill_len)

            def make_request():
                req = edge_fm.Request(0, token_ids_list)
                req.set_ignore_stop_tokens(True)
                return req

            tf_result = _bench_transformers_llm_loaded(
                tf_model, token_ids_list, num_steps,
                warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS)

            trt_result = None
            if include_trt:
                trt_result = _bench_trt_edgellm(
                    engine_dir, plugin_path, token_ids_list, num_steps, prefill_len,
                    warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS,
                    ignore_stop_tokens=True,
                )
                torch.cuda.empty_cache()

            cfg_graph = _create_engine_config(
                model_path, prefill_len, num_steps, use_cuda_graph=True, generated_tokens_total=num_steps
            )
            engine_graph = edge_fm.EdgeFM(cfg_graph)
            try:
                efm_graph_result = _bench_edgefm(
                    engine_graph, make_request, num_steps, prefill_len,
                    warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS)
            finally:
                del engine_graph
                torch.cuda.empty_cache()

            label = (
                f"{model_spec['label']}: Transformers vs EdgeFM (cuda graph) vs TRT-Edge-LLM "
                f"(prefill={prefill_len}, decode={num_steps})"
                if include_trt
                else f"{model_spec['label']}: Transformers vs EdgeFM (cuda graph) "
                     f"(prefill={prefill_len}, decode={num_steps})"
            )
            if include_trt:
                _print_bench_comparison_3way(label, tf_result, efm_graph_result, trt_result)
            else:
                _print_bench_comparison(label, tf_result, efm_graph_result)

            reports.append({
                "config": {
                    "kind": "llm",
                    "model_size": model_spec["model_size"],
                    "model_label": model_spec["label"],
                    "model_path": model_path,
                    "device_id": DEVICE_ID,
                    "prefill_tokens": prefill_len,
                    "decode_tokens": num_steps,
                    "warmup_runs": BENCH_WARMUP_RUNS,
                    "timed_runs": BENCH_TIMED_RUNS,
                    "engine_dir": str(engine_dir) if engine_dir is not None else "",
                    "plugin_path": str(plugin_path) if plugin_path is not None else "",
                },
                "transformers": tf_result,
                "edgefm_cuda_graph": efm_graph_result,
                "trt_edgellm": trt_result,
            })
    finally:
        del tf_model
        torch.cuda.empty_cache()

    return reports


def _benchmark_vlm_model(model_spec: dict, include_trt: bool = False) -> list[dict]:
    import torch

    model_path = model_spec["model_path"]
    bench_cases = _resolve_bench_cases(1, BENCH_NUM_STEPS)
    tf_model, processor = _load_transformers_vlm_model(model_path)
    plugin_path = None
    engine_dir = None
    multimodal_engine_dir = None
    trt_runtime = None

    if include_trt:
        requested_engine_dir = _resolve_trt_requested_vlm_engine_dir(model_spec["model_size"])
        requested_multimodal_engine_dir = _resolve_trt_requested_vlm_multimodal_engine_dir(model_spec["model_size"])
        try:
            engine_dir, multimodal_engine_dir = _resolve_trt_vlm_engine_dirs(
                max(prefill for prefill, _ in bench_cases),
                requested_engine_dir=requested_engine_dir,
                requested_multimodal_engine_dir=requested_multimodal_engine_dir,
                model_size=model_spec["model_size"],
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"\n[benchmark] skipping TRT for {model_spec['label']}: {exc}")
            include_trt = False

        _plugin = os.environ.get("TRT_EDGELLM_PLUGIN_PATH", "").strip()
        plugin_path = (
            Path(_plugin).resolve()
            if _plugin
            else (project_root / "third_party" / "TensorRT-Edge-LLM" / "build" / "libNvInfer_edgellm_plugin.so").resolve()
        )
        if include_trt and not plugin_path.exists():
            print(f"\n[benchmark] skipping TRT for {model_spec['label']}: plugin missing at {plugin_path}")
            include_trt = False
        if include_trt and edge_fm_trt is None:
            print(f"\n[benchmark] skipping TRT for {model_spec['label']}: edge_fm_trt module unavailable")
            include_trt = False
        if include_trt:
            os.environ["EDGELLM_PLUGIN_PATH"] = str(plugin_path)
            trt_runtime = edge_fm_trt.TrtEdgeLlmRuntime(str(engine_dir), str(multimodal_engine_dir), DEVICE_ID)

    print(f"\n[benchmark] {model_spec['label']} cases:")
    for p, d in bench_cases:
        print(f"  - prefill={p}, decode={d}")

    reports = []
    try:
        for prefill_len, num_steps in bench_cases:
            prepared_inputs = _prepare_vlm_bench_case(tf_model, processor, prefill_len)
            token_ids_list = prepared_inputs["edgefm_token_ids"]
            image_embeddings = prepared_inputs["image_embeddings"]
            embed_token_id = prepared_inputs["embed_token_id"]
            position_ids = prepared_inputs["position_ids"]

            tf_result = _bench_transformers_vlm_loaded(
                tf_model,
                prepared_inputs,
                num_steps,
                warmup=BENCH_WARMUP_RUNS,
                runs=BENCH_TIMED_RUNS,
            )

            make_request = _make_vl_request_factory(
                token_ids_list,
                image_embeddings,
                embed_token_id,
                position_ids,
                ignore_stop_tokens=True,
            )
            cfg_graph = _create_engine_config(
                model_path,
                prefill_len,
                num_steps,
                use_cuda_graph=True,
                generated_tokens_total=num_steps,
                model_name="Qwen2.5-VL",
            )
            engine_graph = edge_fm.EdgeFM(cfg_graph)
            try:
                efm_graph_result = _bench_edgefm(
                    engine_graph, make_request, num_steps, prefill_len,
                    warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS)
            finally:
                del engine_graph
                torch.cuda.empty_cache()

            trt_result = None
            if include_trt:
                trt_token_ids_list = _build_trt_vlm_token_ids(prepared_inputs, model_path)
                trt_result = _bench_trt_edgellm_vlm_prepared(
                    trt_runtime,
                    trt_token_ids_list,
                    prepared_inputs,
                    num_steps,
                    prefill_len,
                    warmup=BENCH_WARMUP_RUNS,
                    runs=BENCH_TIMED_RUNS,
                    ignore_stop_tokens=True,
                )
                torch.cuda.empty_cache()

            label = (
                f"{model_spec['label']}: Transformers vs EdgeFM (cuda graph) vs TRT-Edge-LLM "
                f"(prefill={prefill_len}, decode={num_steps})"
                if trt_result is not None
                else f"{model_spec['label']}: Transformers vs EdgeFM (cuda graph) "
                     f"(prefill={prefill_len}, decode={num_steps})"
            )
            if trt_result is not None:
                _print_bench_comparison_3way(label, tf_result, efm_graph_result, trt_result)
            else:
                _print_bench_comparison(label, tf_result, efm_graph_result)
            reports.append({
                "config": {
                    "kind": "vlm",
                    "model_size": model_spec["model_size"],
                    "model_label": model_spec["label"],
                    "model_path": model_path,
                    "device_id": DEVICE_ID,
                    "prefill_tokens": prefill_len,
                    "decode_tokens": num_steps,
                    "warmup_runs": BENCH_WARMUP_RUNS,
                    "timed_runs": BENCH_TIMED_RUNS,
                    "prompt": prepared_inputs["prompt"],
                    "image_path": prepared_inputs["image_path"],
                    "engine_dir": str(engine_dir) if engine_dir is not None else "",
                    "multimodal_engine_dir": str(multimodal_engine_dir) if multimodal_engine_dir is not None else "",
                    "plugin_path": str(plugin_path) if plugin_path is not None else "",
                },
                "transformers": tf_result,
                "edgefm_cuda_graph": efm_graph_result,
                "trt_edgellm": trt_result,
            })
    finally:
        del tf_model
        del trt_runtime
        torch.cuda.empty_cache()

    return reports


def test_benchmark_llm():
    """LLM 性能基准：按模型矩阵比较 Transformers vs EdgeFM(cuda graph)。"""
    models, missing = _resolve_bench_model_specs("llm")
    _print_missing_bench_models("llm", missing)
    if not models:
        pytest.skip("No LLM model weights found for benchmark")

    reports = []
    for model_spec in models:
        reports.extend(_benchmark_llm_model(model_spec, include_trt=False))
    assert reports, "No LLM benchmark cases were executed"


def test_benchmark_trt_edgellm():
    """LLM 性能基准：按模型矩阵比较 Transformers vs EdgeFM(cuda graph) vs TRT-Edge-LLM。"""
    models, missing = _resolve_bench_model_specs("llm")
    _print_missing_bench_models("llm", missing)
    if not models:
        pytest.skip("No LLM model weights found for TRT benchmark")

    reports = []
    for model_spec in models:
        reports.extend(_benchmark_llm_model(model_spec, include_trt=True))
    if not reports:
        pytest.skip("No TRT-capable LLM benchmark cases were executed")


def test_benchmark_vlm():
    """VLM 性能基准：优先比较 Transformers vs EdgeFM(cuda graph) vs TRT-Edge-LLM。"""
    models, missing = _resolve_bench_model_specs("vlm")
    _print_missing_bench_models("vlm", missing)
    if not models:
        pytest.skip("No VLM model weights found for benchmark")

    reports = []
    for model_spec in models:
        reports.extend(_benchmark_vlm_model(model_spec, include_trt=True))
    assert reports, "No VLM benchmark cases were executed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
