import json
import math

import torch
import torch.testing
from safetensors import safe_open

from ._test_utils import (
    DEFAULT_DEVICE_ID,
    QWEN_1P5B_MODEL_PATH,
    edge_fm_tensor_to_torch,
    edge_fm,
    ensure_cuda,
    make_engine_config,
    median_cuda_ms,
    reset_weight_loader,
    tensor_to_edge_fm_tensor,
    torch_device,
)


with open(QWEN_1P5B_MODEL_PATH / "config.json", "r") as f:
    MODEL_CONFIG = json.load(f)

HIDDEN_SIZE = MODEL_CONFIG["hidden_size"]
RMS_NORM_EPS = MODEL_CONFIG["rms_norm_eps"]
VOCAB_SIZE = MODEL_CONFIG["vocab_size"]


def _load_tensor(name: str, *, device: str) -> torch.Tensor:
    with safe_open(
        str(QWEN_1P5B_MODEL_PATH / "model.safetensors"), framework="pt", device=device
    ) as handle:
        return handle.get_tensor(name)


def _make_rmsnorm_layer():
    reset_weight_loader()
    engine_config_path = make_engine_config(
        QWEN_1P5B_MODEL_PATH,
        device_id=DEFAULT_DEVICE_ID,
    )
    return edge_fm.RMSNormLayer(layer_id=0, config_path=str(engine_config_path))


def test_rmsnorm_correctness_and_performance_smoke():
    ensure_cuda()
    device = torch_device()
    layer = _make_rmsnorm_layer()
    weight = _load_tensor("model.layers.0.input_layernorm.weight", device=device)

    torch.manual_seed(0)
    x = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    x_efm = tensor_to_edge_fm_tensor(x)
    y_efm = tensor_to_edge_fm_tensor(y)

    layer.forward_rmsnorm(x_efm, y_efm)
    torch.cuda.synchronize()
    y_torch = edge_fm_tensor_to_torch(y_efm)

    x_fp32 = x.float()
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + RMS_NORM_EPS)
    ref = (x_fp32 * rms * weight.float()).to(torch.bfloat16)
    torch.testing.assert_close(y_torch, ref, rtol=1e-2, atol=1e-2)

    median_ms = median_cuda_ms(lambda: layer.forward_rmsnorm(x_efm, y_efm), warmup=30, iters=200)
    assert math.isfinite(median_ms)
    assert median_ms < 0.5, f"rmsnorm latency regressed to {median_ms:.6f} ms"


def test_fused_add_rmsnorm_correctness_and_performance_smoke():
    ensure_cuda()
    device = torch_device()
    layer = _make_rmsnorm_layer()
    weight = _load_tensor("model.layers.0.input_layernorm.weight", device=device)

    torch.manual_seed(1)
    inout = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    residual = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    inout_ref = inout.float() + residual.float()
    residual_ref = inout_ref.clone()
    rms = torch.rsqrt(inout_ref.pow(2).mean(dim=-1, keepdim=True) + RMS_NORM_EPS)
    output_ref = (inout_ref * rms * weight.float()).to(torch.bfloat16)
    residual_ref = residual_ref.to(torch.bfloat16)

    inout_efm = tensor_to_edge_fm_tensor(inout)
    residual_efm = tensor_to_edge_fm_tensor(residual)
    layer.forward_fused_add_rmsnorm(inout_efm, residual_efm)
    torch.cuda.synchronize()
    inout_torch = edge_fm_tensor_to_torch(inout_efm)
    residual_torch = edge_fm_tensor_to_torch(residual_efm)

    torch.testing.assert_close(inout_torch, output_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(residual_torch, residual_ref, rtol=1e-2, atol=1e-2)

    median_ms = median_cuda_ms(
        lambda: layer.forward_fused_add_rmsnorm(inout_efm, residual_efm),
        warmup=30,
        iters=200,
    )
    assert math.isfinite(median_ms)
    assert median_ms < 0.5, f"fused_add_rmsnorm latency regressed to {median_ms:.6f} ms"


def test_sampler_greedy_correctness_and_performance_smoke():
    ensure_cuda()
    device = torch_device()
    engine_config_path = make_engine_config(
        QWEN_1P5B_MODEL_PATH,
        device_id=DEFAULT_DEVICE_ID,
        sampling={"temperature": 0.0, "seed": 0},
    )
    layer = edge_fm.SamplerLayer(str(engine_config_path))

    torch.manual_seed(0)
    logits = torch.randn(4, VOCAB_SIZE, device=device, dtype=torch.float32)
    token_ids = torch.zeros(4, device=device, dtype=torch.int32)
    logits_efm = tensor_to_edge_fm_tensor(logits)
    token_ids_efm = tensor_to_edge_fm_tensor(token_ids)

    layer.forward(logits_efm, token_ids_efm)
    torch.cuda.synchronize()

    expected = torch.argmax(logits, dim=-1).to(torch.int32)
    token_ids_torch = edge_fm_tensor_to_torch(token_ids_efm)
    assert torch.equal(token_ids_torch, expected)

    median_ms = median_cuda_ms(lambda: layer.forward(logits_efm, token_ids_efm), warmup=20, iters=100)
    assert math.isfinite(median_ms)
    assert median_ms < 2.0, f"sampler latency regressed to {median_ms:.6f} ms"
