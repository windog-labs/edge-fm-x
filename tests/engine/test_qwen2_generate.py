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
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
for _p in [project_root / "build" / "install" / "python", project_root / "build" / "python"]:
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
DEFAULT_SEQ_LEN = 6
DEFAULT_NUM_STEPS = 20
DEFAULT_SEED = 42

# GPU device：默认 1，避免占用 device 0；可通过环境变量 EDGE_FM_DEVICE_ID 覆盖
DEVICE_ID = int(os.environ.get("EDGE_FM_DEVICE_ID", "1"))
CUDA_DEVICE = f"cuda:{DEVICE_ID}"


# ---------------------------------------------------------------------------
# Dump generation (runs Transformers, only when dump is missing)
# ---------------------------------------------------------------------------

def _find_qwen_model_path() -> str | None:
    candidates = [
        os.environ.get("EDGE_FM_QWEN_MODEL_PATH"),
        str(project_root / "examples" / "qwen2.5-1.5b-instruct" / "qwen2.5-1.5b-instruct"),
        str(project_root / "examples" / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct"),
        str(project_root / "examples" / "qwen2.5-1.5b-instruct"),
        str(project_root / "examples" / "qwen2.5-0.5b-instruct"),
    ]
    for p in candidates:
        if p is None:
            continue
        path = Path(p)
        if path.exists() and (path / "config.json").exists() and (path / "model.safetensors").exists():
            return str(path.resolve())
    return None


def _find_qwen_vl_model_path() -> str | None:
    """查找 Qwen2.5-VL-3B-Instruct 模型路径（用于 VL 含图测试）。"""
    candidates = [
        os.environ.get("EDGE_FM_QWEN_VL_MODEL_PATH"),
        str(project_root / "examples" / "qwen2.5-vl-3b-instruct" / "qwen2.5-vl-3b-instruct"),
        str(project_root / "examples" / "qwen2.5-vl-3b-instruct"),
    ]
    for p in candidates:
        if p is None:
            continue
        path = Path(p)
        if not path.exists() or not (path / "config.json").exists():
            continue
        # VLM 可能是单文件或分片
        if (path / "model.safetensors").exists():
            return str(path.resolve())
        for f in path.glob("model-*.safetensors"):
            if f.exists():
                return str(path.resolve())
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


@pytest.fixture(scope="module")
def engine_and_dump_prefix_no_graph(dump_data):
    """Same as cuda_graph but use_cuda_graph=False: verify prefix path correctness."""
    manifest = dump_data["manifest"]
    model_path = dump_data["model_path"]
    num_steps = manifest["num_decode_steps"]
    seq_len = int(dump_data["token_ids"].size)
    token_ids_flat = dump_data["token_ids"].flatten()
    prefix = token_ids_flat[: min(4, len(token_ids_flat))].tolist()
    engine_config_path = _create_engine_config(
        model_path, seq_len, num_steps, use_cuda_graph=False, prefix_token_ids=prefix
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


def test_generate_token_alignment_prefix_no_graph(engine_and_dump_prefix_no_graph):
    """Prefix 路径（无 graph）：验证 cache_kv_len 等逻辑正确。"""
    import torch

    engine = engine_and_dump_prefix_no_graph["engine"]
    token_ids = engine_and_dump_prefix_no_graph["token_ids"]
    decode_tokens = engine_and_dump_prefix_no_graph["decode_tokens"]
    num_steps = engine_and_dump_prefix_no_graph["num_steps"]

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
            f"Prefix (no graph) token 不对齐：{len(mismatches)}/{num_steps} 步不一致\n{detail}"
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

BENCH_NUM_STEPS = 50
BENCH_WARMUP_RUNS = 3
BENCH_TIMED_RUNS = 5


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


def _resolve_bench_cases(default_prefill: int, default_decode: int) -> list[tuple[int, int]]:
    """Resolve benchmark cases from env.

    - EDGE_FM_BENCH_PREFILL_LIST: comma-separated prefill lengths
    - EDGE_FM_BENCH_DECODE_LIST: comma-separated decode lengths
    If both lists are provided and have equal length, run zip pairs; otherwise run cross-product.
    """
    p_list = _parse_bench_lengths("EDGE_FM_BENCH_PREFILL_LIST")
    d_list = _parse_bench_lengths("EDGE_FM_BENCH_DECODE_LIST")
    if p_list is None:
        p_list = [default_prefill]
    if d_list is None:
        d_list = [default_decode]
    if len(p_list) == len(d_list):
        return list(zip(p_list, d_list))
    return [(p, d) for p in p_list for d in d_list]


def _bench_transformers_llm(model_path: str, token_ids_list: list[int], num_steps: int,
                            warmup: int, runs: int) -> dict:
    """Benchmark Transformers KV-cached greedy decode for LLM."""
    import torch
    import time
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(model_path)
    torch_dtype_str = str(getattr(config, "torch_dtype", "float16")).lower()
    model_dtype = torch.bfloat16 if "bfloat" in torch_dtype_str or "bf16" in torch_dtype_str else torch.float16
    device = CUDA_DEVICE if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=model_dtype, low_cpu_mem_usage=False)
    model = model.to(device)
    model.eval()

    device = CUDA_DEVICE
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

    del model
    torch.cuda.empty_cache()

    avg = sum(times) / len(times)
    return {
        "avg_ms": avg * 1000,
        "prefill_tokens": prefill_len,
        "decode_tokens": num_steps,
        "total_tokens": prefill_len + num_steps,
        "tokens_per_sec": (prefill_len + num_steps) / avg,
        "decode_tokens_per_sec": num_steps / avg,
        "times_ms": [t * 1000 for t in times],
    }


def _bench_transformers_vlm(model_path: str, token_ids_list: list[int],
                            image_path: str, num_steps: int,
                            warmup: int, runs: int) -> dict:
    """Benchmark Transformers KV-cached greedy decode for VLM."""
    import torch
    import time
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from PIL import Image

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=CUDA_DEVICE)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 50176

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "What animal is on the candy?"},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(CUDA_DEVICE)
    prefill_len = inputs["input_ids"].shape[1]

    def run_once():
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        rope_deltas = out.rope_deltas
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

    del model
    torch.cuda.empty_cache()

    avg = sum(times) / len(times)
    return {
        "avg_ms": avg * 1000,
        "prefill_tokens": prefill_len,
        "decode_tokens": num_steps,
        "total_tokens": prefill_len + num_steps,
        "tokens_per_sec": (prefill_len + num_steps) / avg,
        "decode_tokens_per_sec": num_steps / avg,
        "times_ms": [t * 1000 for t in times],
    }


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
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        response = engine.generate(request_fn())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        generated_counts.append(len(response.token_ids()))

    if any(count != num_steps for count in generated_counts):
        raise AssertionError(
            f"EdgeFM timed runs generated unexpected token counts: {generated_counts}, "
            f"expected every run to generate {num_steps} tokens"
        )

    avg = sum(times) / len(times)
    actual_decode_tokens = generated_counts[0] if generated_counts else num_steps
    total_tokens = prefill_len + actual_decode_tokens
    return {
        "avg_ms": avg * 1000,
        "prefill_tokens": prefill_len,
        "decode_tokens": actual_decode_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / avg,
        "decode_tokens_per_sec": actual_decode_tokens / avg,
        "times_ms": [t * 1000 for t in times],
        "generated_counts": generated_counts,
    }


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
    for _ in range(runs):
        t0 = time.perf_counter()
        output_ids, _ = runtime.generate_from_token_ids(
            token_ids_list, num_steps, temperature=0.0, top_p=1.0, top_k=1, ignore_stop_tokens=ignore_stop_tokens
        )
        times.append((time.perf_counter() - t0) * 1000)
        generated_counts.append(len(output_ids[0]) if output_ids else 0)

    if any(count != num_steps for count in generated_counts):
        raise AssertionError(
            f"TRT-Edge-LLM timed runs generated unexpected token counts: {generated_counts}, "
            f"expected every run to generate {num_steps} tokens"
        )

    avg_ms = sum(times) / len(times)
    avg_s = avg_ms / 1000.0
    actual_decode_tokens = generated_counts[0] if generated_counts else num_steps
    total_tokens = prefill_len + actual_decode_tokens
    return {
        "avg_ms": avg_ms,
        "prefill_tokens": prefill_len,
        "decode_tokens": actual_decode_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / avg_s,
        "decode_tokens_per_sec": actual_decode_tokens / avg_s,
        "times_ms": times,
        "generated_counts": generated_counts,
    }


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
    print(f"{'='*90}")


def _print_bench_comparison_2way(
    label_a: str, result_a: dict, label_b: str, result_b: dict,
    prefill_tokens: int, decode_tokens: int,
):
    """Pretty-print 2-way benchmark comparison (e.g. EdgeFM no-graph vs cuda-graph)."""
    print(f"\n  --- {label_a} vs {label_b} ---")
    print(f"  {'Metric':<30} {label_a:>15} {label_b:>15} {'Speedup':>10}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*10}")
    rows = [
        ("Total latency (ms)", "avg_ms", ".1f"),
        ("Total throughput (tok/s)", "tokens_per_sec", ".1f"),
        ("Decode throughput (tok/s)", "decode_tokens_per_sec", ".1f"),
    ]
    for name, key, fmt in rows:
        va, vb = result_a[key], result_b[key]
        if "latency" in name.lower():
            speedup = va / vb if vb > 0 else float("inf")
        else:
            speedup = vb / va if va > 0 else float("inf")
        print(f"  {name:<30} {va:>15{fmt}} {vb:>15{fmt}} {speedup:>9.2f}x")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*10}")


def test_benchmark_llm(dump_data):
    """LLM 性能基准：Transformers vs EdgeFM（纯文本），以及 EdgeFM no-graph vs cuda-graph。"""
    import torch

    model_path = dump_data["model_path"]
    token_ids = dump_data["token_ids"]
    token_ids_list = token_ids.flatten().tolist()
    num_steps = BENCH_NUM_STEPS
    prefill_len = len(token_ids_list)

    tf_result = _bench_transformers_llm(
        model_path, token_ids_list, num_steps,
        warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS)

    engine_config_path = _create_engine_config(
        model_path, prefill_len, num_steps, generated_tokens_total=num_steps)
    engine = edge_fm.EdgeFM(engine_config_path)

    def make_request():
        req = edge_fm.Request(0, token_ids_list)
        req.set_ignore_stop_tokens(True)
        return req

    efm_result = _bench_edgefm(
        engine, make_request, num_steps, prefill_len,
        warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS)

    _print_bench_comparison("LLM (pure text)", tf_result, efm_result)

    # Fair decode-graph benchmark: do not use prefix_token_ids so graph capture
    # happens on the same request path as the no-graph baseline.
    cfg_graph = _create_engine_config(
        model_path, prefill_len, num_steps, use_cuda_graph=True, generated_tokens_total=num_steps
    )
    engine_graph = edge_fm.EdgeFM(cfg_graph)
    efm_graph_result = _bench_edgefm(
        engine_graph, make_request, num_steps, prefill_len,
        warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS)

    _print_bench_comparison_2way(
        "EdgeFM (no graph)", efm_result, "EdgeFM (cuda graph)", efm_graph_result,
        prefill_len, num_steps,
    )

    _print_bench_comparison("LLM: Transformers vs EdgeFM (cuda graph)", tf_result, efm_graph_result)


def test_benchmark_trt_edgellm(dump_data):
    """LLM 性能基准：Transformers vs EdgeFM(cuda graph) vs TRT-Edge-LLM（batch=1，同步计时）。"""
    import torch

    model_path = dump_data["model_path"]
    token_ids = dump_data["token_ids"]
    base_token_ids = token_ids.flatten().tolist()
    default_prefill_len = len(base_token_ids)
    default_decode_steps = BENCH_NUM_STEPS
    bench_cases = _resolve_bench_cases(default_prefill_len, default_decode_steps)

    _engine_dir = os.environ.get("TRT_EDGELLM_ENGINE_DIR", "").strip()
    engine_dir = (
        Path(_engine_dir).resolve()
        if _engine_dir
        else (project_root / "tests" / "data" / "trt_edgellm_workspace" / "qwen2.5-1.5b" / "engines").resolve()
    )
    _plugin = os.environ.get("TRT_EDGELLM_PLUGIN_PATH", "").strip()
    plugin_path = (
        Path(_plugin).resolve()
        if _plugin
        else (project_root / "third_party" / "TensorRT-Edge-LLM" / "build" / "libNvInfer_edgellm_plugin.so").resolve()
    )

    if not engine_dir.exists() or not (engine_dir / "llm.engine").exists():
        pytest.skip(
            f"TRT-Edge-LLM engine not found at {engine_dir}. "
            "Run: bash tests/scripts/setup_trt_edgellm_benchmark.sh"
        )
    if not plugin_path.exists():
        pytest.skip(
            f"TRT-Edge-LLM plugin not found at {plugin_path}. "
            "Build TensorRT-Edge-LLM first so libNvInfer_edgellm_plugin.so is available."
        )
    if edge_fm_trt is None:
        pytest.skip(
            "edge_fm_trt (in-process TRT runtime) not found. "
            "Build with BUILD_TRT_EDGELLM_PYBIND=ON and add its module path to PYTHONPATH."
        )

    print("\n[benchmark] test_benchmark_trt_edgellm cases:")
    for p, d in bench_cases:
        print(f"  - prefill={p}, decode={d}")

    for prefill_len, num_steps in bench_cases:
        token_ids_list = _build_prefill_token_ids(base_token_ids, prefill_len)

        def make_request():
            req = edge_fm.Request(0, token_ids_list)
            req.set_ignore_stop_tokens(True)  # 公平对比：固定生成 num_steps 个 token，不因 EOS 提前停止
            return req

        tf_result = _bench_transformers_llm(
            model_path, token_ids_list, num_steps,
            warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS)

        # Run TRT before EdgeFM to reduce peak memory overlap with EdgeFM runtime allocations.
        trt_result = _bench_trt_edgellm(
            engine_dir, plugin_path, token_ids_list, num_steps, prefill_len,
            warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS,
            ignore_stop_tokens=True,  # 公平对比：固定生成 num_steps 个 token
        )

        torch.cuda.empty_cache()

        cfg_no_graph = _create_engine_config(
            model_path, prefill_len, num_steps, generated_tokens_total=num_steps)
        engine = edge_fm.EdgeFM(cfg_no_graph)
        efm_no_graph_result = _bench_edgefm(
            engine, make_request, num_steps, prefill_len,
            warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS)
        del engine
        torch.cuda.empty_cache()

        cfg_graph = _create_engine_config(
            model_path, prefill_len, num_steps, use_cuda_graph=True, generated_tokens_total=num_steps
        )
        engine_graph = edge_fm.EdgeFM(cfg_graph)
        efm_graph_result = _bench_edgefm(
            engine_graph, make_request, num_steps, prefill_len,
            warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS)
        del engine_graph
        torch.cuda.empty_cache()

        _print_bench_comparison_2way(
            "EdgeFM (no graph)", efm_no_graph_result, "EdgeFM (cuda graph)", efm_graph_result,
            prefill_len, num_steps,
        )

        _print_bench_comparison_3way(
            f"LLM: Transformers vs EdgeFM (cuda graph) vs TRT-Edge-LLM (prefill={prefill_len}, decode={num_steps})",
            tf_result, efm_graph_result, trt_result,
        )


def test_benchmark_vlm(dump_data_vl):
    """VLM 性能基准：Transformers vs EdgeFM（图 + 文本），以及 EdgeFM no-graph vs cuda-graph。"""
    import torch

    model_path = dump_data_vl["model_path"]
    token_ids = dump_data_vl["token_ids"]
    image_embeddings = dump_data_vl["image_embeddings"]
    embed_token_id = dump_data_vl["embed_token_id"]
    position_ids = dump_data_vl["position_ids"]
    token_ids_list = token_ids.flatten().tolist()
    num_steps = BENCH_NUM_STEPS
    prefill_len = len(token_ids_list)

    image_path = str(project_root / "tests" / "data" / "candy.JPG")
    tf_result = _bench_transformers_vlm(
        model_path, token_ids_list, image_path, num_steps,
        warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS)

    engine_config_path = _create_engine_config(
        model_path, prefill_len, num_steps, generated_tokens_total=num_steps)
    engine = edge_fm.EdgeFM(engine_config_path)
    make_request = _make_vl_request_factory(
        token_ids_list,
        image_embeddings,
        embed_token_id,
        position_ids,
        ignore_stop_tokens=True,
    )

    efm_result = _bench_edgefm(
        engine, make_request, num_steps, prefill_len,
        warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS)

    _print_bench_comparison("VLM (image + text)", tf_result, efm_result)

    # Fair decode-graph benchmark: avoid prefix_token_ids so capture happens on
    # the full multimodal request instead of a text-only prefix warmup.
    cfg_graph = _create_engine_config(
        model_path, prefill_len, num_steps, use_cuda_graph=True,
        generated_tokens_total=num_steps, model_name="Qwen2.5-VL"
    )
    engine_graph = edge_fm.EdgeFM(cfg_graph)
    efm_graph_result = _bench_edgefm(
        engine_graph, make_request, num_steps, prefill_len,
        warmup=BENCH_WARMUP_RUNS, runs=BENCH_TIMED_RUNS)

    _print_bench_comparison_2way(
        "EdgeFM (no graph)", efm_result, "EdgeFM (cuda graph)", efm_graph_result,
        prefill_len, num_steps,
    )

    _print_bench_comparison("VLM: Transformers vs EdgeFM (cuda graph)", tf_result, efm_graph_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
