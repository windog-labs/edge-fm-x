"""
Attention 层的正确性和性能测试（pytest 单元测试）

测试 edge_fm.AttentionLayer 的实现，使用 flashinfer 作为参考实现。
包括：
1. 正确性测试：分别测试 prefill 和 decode 模式
2. 性能测试：确保性能不低于 flashinfer 的 0.9 倍
"""

import sys
import json
import torch
import pytest
import tempfile
import os
from pathlib import Path

# 添加构建目录到路径
project_root = Path(__file__).parent.parent.parent
build_python = project_root / "build" / "install" / "python"
sys.path.insert(0, str(build_python))

import edge_fm
import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def create_attention_layer(num_qo_heads, num_kv_heads, hidden_size):
    """创建 AttentionLayer 实例"""
    # 创建临时目录用于存放 config.json
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "config.json")
    
    # 创建模型配置文件
    model_config = {
        "num_attention_heads": num_qo_heads,
        "num_key_value_heads": num_kv_heads,
        "hidden_size": hidden_size,
        "rope_theta": 1000000.0  # Qwen2.5 默认值，与 C++ 实现对齐
    }
    with open(config_path, "w") as f:
        json.dump(model_config, f)
    
    # 创建 engine_config.json 文件
    engine_config_dir = tempfile.mkdtemp()
    engine_config_path = os.path.join(engine_config_dir, "engine_config.json")
    engine_config = {
        "runtime": {
            "device": "cuda",
            "device_id": 0
        },
        "prefill_model_path": temp_dir
    }
    with open(engine_config_path, "w") as f:
        json.dump(engine_config, f)
    
    return edge_fm.AttentionLayer(engine_config_path)


def tensor_to_edge_fm_tensor(torch_tensor):
    """将 PyTorch tensor 转换为 edge_fm.Tensor（通过 DLPack）
    需确保 tensor 是 contiguous 且 row-major。对已 contiguous 的 tensor，
    .contiguous() 是空操作；对 view/transpose 等则会产生连续拷贝。
    """
    capsule = torch_tensor.contiguous().__dlpack__()
    return edge_fm.Tensor.from_dlpack(capsule)


# 测试参数
PREFILL_TEST_CASES = [
    {"qo_len": 128, "kv_len": 512, "num_qo_heads": 16, "num_kv_heads": 16, "head_dim": 128, "causal": True},
    {"qo_len": 256, "kv_len": 1024, "num_qo_heads": 16, "num_kv_heads": 16, "head_dim": 128, "causal": True},
    {"qo_len": 64, "kv_len": 256, "num_qo_heads": 16, "num_kv_heads": 16, "head_dim": 64, "causal": False},
]

DECODE_TEST_CASES = [
    {"kv_len": 512, "num_qo_heads": 4, "num_kv_heads": 4, "head_dim": 128},
    {"kv_len": 1024, "num_qo_heads": 4, "num_kv_heads": 4, "head_dim": 128},
    {"kv_len": 512, "num_qo_heads": 8, "num_kv_heads": 8, "head_dim": 128},
    {"kv_len": 1024, "num_qo_heads": 8, "num_kv_heads": 8, "head_dim": 128},
]

PREFILL_PERF_TEST_CASES = [
    {"qo_len": 128, "kv_len": 512, "num_qo_heads": 16, "num_kv_heads": 16, "head_dim": 128},
    {"qo_len": 512, "kv_len": 2048, "num_qo_heads": 16, "num_kv_heads": 8, "head_dim": 128},
    {"qo_len": 1024, "kv_len": 4096, "num_qo_heads": 16, "num_kv_heads": 16, "head_dim": 128},
]

DECODE_PERF_TEST_CASES = [
    {"kv_len": 512, "num_qo_heads": 16, "num_kv_heads": 16, "head_dim": 128},
    {"kv_len": 2048, "num_qo_heads": 16, "num_kv_heads": 8, "head_dim": 128},
    {"kv_len": 4096, "num_qo_heads": 16, "num_kv_heads": 16, "head_dim": 128},
    {"kv_len": 8192, "num_qo_heads": 16, "num_kv_heads": 16, "head_dim": 128},
]


class TestAttention:
    """Attention 层测试类（包括正确性和性能测试）"""
    
    @pytest.mark.parametrize("case", PREFILL_TEST_CASES)
    def test_prefill_correctness(self, case):
        """测试 Prefill 模式的正确性
        
        参数:
            case: 测试用例字典（pytest 参数化）
        """
        qo_len = case["qo_len"]
        kv_len = case["kv_len"]
        num_qo_heads = case["num_qo_heads"]
        num_kv_heads = case["num_kv_heads"]
        head_dim = case["head_dim"]
        causal = case["causal"]
        
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # 创建输入张量
        q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16)
        k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16)
        v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16)
        
        # 使用 FlashInfer 作为参考（使用 ROPE_LLAMA 模式，与 edge_fm 实现对齐）
        o_flashinfer = flashinfer.prefill.single_prefill_with_kv_cache(
            q, k, v, causal=causal, pos_encoding_mode="ROPE_LLAMA", 
            rope_theta=1000000.0, rope_scale=1.0, backend="fa2"
        )
        
        # 使用 edge_fm 实现
        layer = create_attention_layer(num_qo_heads, num_kv_heads, num_qo_heads * head_dim)
        
        q_efm = tensor_to_edge_fm_tensor(q)
        k_efm = tensor_to_edge_fm_tensor(k)
        v_efm = tensor_to_edge_fm_tensor(v)
        o_efm = tensor_to_edge_fm_tensor(torch.empty_like(q))
        
        # 执行 forward_prefill
        layer.forward_prefill(q_efm, k_efm, v_efm, o_efm, causal=causal)
        o_efm_torch = torch.from_dlpack(o_efm.to_dlpack())
        torch.cuda.synchronize()
        
        # 使用标准的 torch.testing.assert_close 进行断言
        torch.testing.assert_close(
            o_efm_torch, 
            o_flashinfer, 
            rtol=1e-3, 
            atol=1e-3,
            msg=f"Prefill 结果与 FlashInfer 不一致 (qo_len={qo_len}, kv_len={kv_len}, causal={causal})"
        )
    
    @pytest.mark.parametrize("case", DECODE_TEST_CASES)
    def test_decode_correctness(self, case):
        """测试 Decode 模式的正确性
        
        参数:
            case: 测试用例字典（pytest 参数化）
        """
        kv_len = case["kv_len"]
        num_qo_heads = case["num_qo_heads"]
        num_kv_heads = case["num_kv_heads"]
        head_dim = case["head_dim"]
        
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # 创建输入张量 (decode: q 的第一个维度为 1)
        q = torch.randn(1, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16)
        k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16)
        v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16)
        
        # 使用 FlashInfer 作为参考（使用 ROPE_LLAMA 模式，与 edge_fm 实现对齐）
        o_flashinfer = flashinfer.decode.single_decode_with_kv_cache(
            q.squeeze(0), k, v, pos_encoding_mode="ROPE_LLAMA",
            rope_theta=1000000.0, rope_scale=1.0
        )
        o_flashinfer = o_flashinfer.unsqueeze(0)  # [1, num_qo_heads, head_dim]
        
        # 使用 edge_fm 实现
        layer = create_attention_layer(num_qo_heads, num_kv_heads, num_qo_heads * head_dim)
        
        q_efm = tensor_to_edge_fm_tensor(q)
        k_efm = tensor_to_edge_fm_tensor(k)
        v_efm = tensor_to_edge_fm_tensor(v)
        o_efm = tensor_to_edge_fm_tensor(torch.empty_like(q))
        
        # 执行 forward_decode
        layer.forward_decode(q_efm, k_efm, v_efm, o_efm)
        o_efm_torch = torch.from_dlpack(o_efm.to_dlpack())
        torch.cuda.synchronize()
        
        # 使用标准的 torch.testing.assert_close 进行断言
        torch.testing.assert_close(
            o_efm_torch, 
            o_flashinfer, 
            rtol=1e-3, 
            atol=1e-3,
            msg=f"Decode 结果与 FlashInfer 不一致 (kv_len={kv_len})"
        )
    
    @pytest.mark.parametrize("case", PREFILL_PERF_TEST_CASES)
    def test_prefill_performance(self, case):
        """测试 Prefill 模式的性能
        
        确保 EdgeFM 的性能不低于 FlashInfer 的 0.9 倍。
        
        参数:
            case: 测试用例字典（pytest 参数化）
        """
        qo_len = case["qo_len"]
        kv_len = case["kv_len"]
        num_qo_heads = case["num_qo_heads"]
        num_kv_heads = case["num_kv_heads"]
        head_dim = case["head_dim"]
        
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # 创建输入张量
        q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16)
        k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16)
        v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16)
        
        # 测试 FlashInfer 性能（使用 ROPE_LLAMA 模式，与 edge_fm 实现对齐）
        def run_flashinfer():
            flashinfer.prefill.single_prefill_with_kv_cache(
                q, k, v, causal=True, pos_encoding_mode="ROPE_LLAMA",
                rope_theta=1000000.0, rope_scale=1.0, backend="fa2"
            )
        
        flashinfer_measurements = bench_gpu_time(run_flashinfer, repeat_time_ms=100)
        flashinfer_avg = sum(flashinfer_measurements) / len(flashinfer_measurements)
        
        # 测试 edge_fm 性能
        layer = create_attention_layer(num_qo_heads, num_kv_heads, num_qo_heads * head_dim)
        
        q_efm = tensor_to_edge_fm_tensor(q)
        k_efm = tensor_to_edge_fm_tensor(k)
        v_efm = tensor_to_edge_fm_tensor(v)
        o_efm = tensor_to_edge_fm_tensor(torch.empty_like(q))
        
        def run_edge_fm():
            layer.forward_prefill(q_efm, k_efm, v_efm, o_efm, causal=True)
        
        edge_fm_measurements = bench_gpu_time(run_edge_fm, repeat_time_ms=100)
        edge_fm_avg = sum(edge_fm_measurements) / len(edge_fm_measurements)
        
        # 计算性能比：FlashInfer 时间 / EdgeFM 时间
        performance_ratio = flashinfer_avg / edge_fm_avg if edge_fm_avg > 0 else 0.0
        
        # 断言：EdgeFM 性能不低于 FlashInfer 的 0.9 倍
        assert performance_ratio >= 0.9, (
            f"性能测试失败：EdgeFM 性能低于 FlashInfer 的 0.9 倍\n"
            f"  qo_len={qo_len}, kv_len={kv_len}\n"
            f"  FlashInfer 平均时间: {flashinfer_avg:.4f} ms\n"
            f"  EdgeFM 平均时间: {edge_fm_avg:.4f} ms\n"
            f"  性能比 (FlashInfer/EdgeFM): {performance_ratio:.4f}\n"
            f"  要求: performance_ratio >= 0.9"
        )
    
    @pytest.mark.parametrize("case", DECODE_PERF_TEST_CASES)
    def test_decode_performance(self, case):
        """测试 Decode 模式的性能
        
        确保 EdgeFM 的性能不低于 FlashInfer 的 0.9 倍。
        
        参数:
            case: 测试用例字典（pytest 参数化）
        """
        kv_len = case["kv_len"]
        num_qo_heads = case["num_qo_heads"]
        num_kv_heads = case["num_kv_heads"]
        head_dim = case["head_dim"]
        
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # 创建输入张量 (decode: q 的第一个维度为 1)
        q = torch.randn(1, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16)
        k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16)
        v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16)
        
        # 测试 FlashInfer 性能（使用 ROPE_LLAMA 模式，与 edge_fm 实现对齐）
        def run_flashinfer():
            flashinfer.decode.single_decode_with_kv_cache(
                q.squeeze(0), k, v, pos_encoding_mode="ROPE_LLAMA",
                rope_theta=1000000.0, rope_scale=1.0
            )
        
        flashinfer_measurements = bench_gpu_time(run_flashinfer, repeat_time_ms=100)
        flashinfer_avg = sum(flashinfer_measurements) / len(flashinfer_measurements)
        
        # 测试 edge_fm 性能
        layer = create_attention_layer(num_qo_heads, num_kv_heads, num_qo_heads * head_dim)
        
        q_efm = tensor_to_edge_fm_tensor(q)
        k_efm = tensor_to_edge_fm_tensor(k)
        v_efm = tensor_to_edge_fm_tensor(v)
        o_efm = tensor_to_edge_fm_tensor(torch.empty_like(q))
        
        def run_edge_fm():
            layer.forward_decode(q_efm, k_efm, v_efm, o_efm)
        
        edge_fm_measurements = bench_gpu_time(run_edge_fm, repeat_time_ms=100)
        edge_fm_avg = sum(edge_fm_measurements) / len(edge_fm_measurements)
        
        # 计算性能比：FlashInfer 时间 / EdgeFM 时间
        performance_ratio = flashinfer_avg / edge_fm_avg if edge_fm_avg > 0 else 0.0
        
        # 断言：EdgeFM 性能不低于 FlashInfer 的 0.9 倍
        assert performance_ratio >= 0.9, (
            f"性能测试失败：EdgeFM 性能低于 FlashInfer 的 0.9 倍\n"
            f"  kv_len={kv_len}\n"
            f"  FlashInfer 平均时间: {flashinfer_avg:.4f} ms\n"
            f"  EdgeFM 平均时间: {edge_fm_avg:.4f} ms\n"
            f"  性能比 (FlashInfer/EdgeFM): {performance_ratio:.4f}\n"
            f"  要求: performance_ratio >= 0.9"
        )


if __name__ == "__main__":
    # 支持直接运行：python test_attn.py
    pytest.main([__file__, "-v"])
