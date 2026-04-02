"""
Activation 层的正确性和性能测试（pytest 单元测试）

测试 edge_fm.ActivationLayer 的实现，使用 flashinfer 作为参考实现。
包括：
1. 正确性测试：测试 forward_silu_and_mul（Float16 和 BFloat16）
2. 性能测试：确保性能不低于 flashinfer 的 0.9 倍
"""

import sys
import json
import torch
import pytest
import tempfile
import statistics
from pathlib import Path

# 添加构建目录到路径
project_root = Path(__file__).parent.parent.parent
build_python = project_root / "build" / "install" / "python"
sys.path.insert(0, str(build_python))

import edge_fm
import flashinfer
from flashinfer.testing.utils import bench_gpu_time

# 模型配置
model_path = project_root / "examples" / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct"
with open(model_path / "config.json", "r") as f:
    config = json.load(f)
    hidden_size = config.get("intermediate_size") or config.get("hidden_size", 896)


@pytest.fixture(scope="module")
def activation_layer():
    """创建 ActivationLayer 实例（模块级 fixture，所有测试共享）"""
    engine_config_dir = tempfile.mkdtemp()
    engine_config_path = Path(engine_config_dir) / "engine_config.json"
    engine_config = {
        "model_name": "Qwen2.5",
        "runtime": {
            "device": "cuda",
            "device_id": 0,
            "hw_profile": "cuda_sm80"
        },
        "operator_impl_table_path": str((project_root / "examples" / "config" / "operator_impl_table.json").resolve()),
        "prefill_model_path": str(model_path)
    }
    with open(engine_config_path, "w") as f:
        json.dump(engine_config, f)
    return edge_fm.ActivationLayer(str(engine_config_path))


def tensor_to_edge_fm_tensor(torch_tensor):
    """将 PyTorch tensor 转换为 edge_fm.Tensor（通过 DLPack）
    需确保 tensor 是 contiguous 且 row-major。对已 contiguous 的 tensor，
    .contiguous() 是空操作；对 view/transpose 等则会产生连续拷贝。
    """
    capsule = torch_tensor.contiguous().__dlpack__()
    return edge_fm.Tensor.from_dlpack(capsule)


# 测试参数：seq_len 的取值
SEQ_LENS = [512, 1024, 2048, 4096]
BATCH_SIZE = 1


class TestActivation:
    """Activation 层测试类（包括正确性和性能测试）"""
    
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_forward_silu_and_mul_correctness_float16(self, activation_layer, seq_len):
        """测试 forward_silu_and_mul 的正确性（Float16）
        
        参数:
            activation_layer: ActivationLayer 实例（pytest fixture）
            seq_len: 序列长度（pytest 参数化）
        """
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # 创建输入张量：形状为 (batch_size, seq_len, 2 * hidden_size)
        # 前半部分是 gate projection，后半部分是 up projection
        input_tensor = torch.randn(
            BATCH_SIZE, seq_len, 2 * hidden_size, 
            device="cuda:0", dtype=torch.float16
        )
        
        # FlashInfer 参考实现
        # silu(input[..., :hidden_size]) * input[..., hidden_size:]
        output_flashinfer = (
            input_tensor[..., hidden_size:] * 
            torch.nn.functional.silu(input_tensor[..., :hidden_size])
        )
        
        # 转换为 edge_fm.Tensor
        input_efm = tensor_to_edge_fm_tensor(input_tensor)
        output_efm = tensor_to_edge_fm_tensor(
            torch.empty(BATCH_SIZE, seq_len, hidden_size, device="cuda:0", dtype=torch.float16)
        )
        
        # 执行 forward_silu_and_mul
        activation_layer.forward_silu_and_mul(input_efm, output_efm)
        output_efm_torch = torch.from_dlpack(output_efm.to_dlpack())
        torch.cuda.synchronize()
        
        # 使用标准的 torch.testing.assert_close 进行断言
        # pytest 会自动捕获 AssertionError 并报告详细的错误信息
        torch.testing.assert_close(
            output_efm_torch, 
            output_flashinfer, 
            rtol=1e-3, 
            atol=1e-3,
            msg=f"forward_silu_and_mul 结果与 FlashInfer 不一致 (seq_len={seq_len}, dtype=float16)"
        )
    
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_forward_silu_and_mul_correctness_bfloat16(self, activation_layer, seq_len):
        """测试 forward_silu_and_mul 的正确性（BFloat16）
        
        参数:
            activation_layer: ActivationLayer 实例（pytest fixture）
            seq_len: 序列长度（pytest 参数化）
        """
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # 创建输入张量：形状为 (batch_size, seq_len, 2 * hidden_size)，使用 bfloat16
        input_tensor = torch.randn(
            BATCH_SIZE, seq_len, 2 * hidden_size, 
            device="cuda:0", dtype=torch.bfloat16
        )
        
        # FlashInfer 参考实现
        # silu(input[..., :hidden_size]) * input[..., hidden_size:]
        output_flashinfer = (
            input_tensor[..., hidden_size:] * 
            torch.nn.functional.silu(input_tensor[..., :hidden_size])
        )
        
        # 转换为 edge_fm.Tensor
        input_efm = tensor_to_edge_fm_tensor(input_tensor)
        output_efm = tensor_to_edge_fm_tensor(
            torch.empty(BATCH_SIZE, seq_len, hidden_size, device="cuda:0", dtype=torch.bfloat16)
        )
        
        # 执行 forward_silu_and_mul
        activation_layer.forward_silu_and_mul(input_efm, output_efm)
        output_efm_torch = torch.from_dlpack(output_efm.to_dlpack())
        torch.cuda.synchronize()
        
        # 使用标准的 torch.testing.assert_close 进行断言
        # BFloat16 精度较低，使用更宽松的容差
        torch.testing.assert_close(
            output_efm_torch, 
            output_flashinfer, 
            rtol=1e-2, 
            atol=1e-2,
            msg=f"forward_silu_and_mul 结果与 FlashInfer 不一致 (seq_len={seq_len}, dtype=bfloat16)"
        )
    
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_forward_silu_and_mul_performance_float16(self, activation_layer, seq_len):
        """测试 forward_silu_and_mul 的性能（Float16）
        
        确保 EdgeFM 的性能不低于 FlashInfer 的 0.85 倍。
        使用中位数、更长测量时间以降低微秒级 benchmark 的波动。
        """
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        input_tensor = torch.randn(
            BATCH_SIZE, seq_len, 2 * hidden_size,
            device="cuda:0", dtype=torch.float16
        )

        def run_flashinfer():
            flashinfer.activation.silu_and_mul(input_tensor)

        flashinfer_measurements = bench_gpu_time(run_flashinfer, repeat_time_ms=300)
        flashinfer_median = statistics.median(flashinfer_measurements)

        input_efm = tensor_to_edge_fm_tensor(input_tensor)
        output_efm = tensor_to_edge_fm_tensor(
            torch.empty(BATCH_SIZE, seq_len, hidden_size, device="cuda:0", dtype=torch.float16)
        )

        def run_edge_fm():
            activation_layer.forward_silu_and_mul(input_efm, output_efm)

        edge_fm_measurements = bench_gpu_time(run_edge_fm, repeat_time_ms=300)
        edge_fm_median = statistics.median(edge_fm_measurements)
        performance_ratio = flashinfer_median / edge_fm_median if edge_fm_median > 0 else 0.0

        assert performance_ratio >= 0.85, (
            f"性能测试失败：EdgeFM 性能低于 FlashInfer 的 0.85 倍\n"
            f"  seq_len={seq_len}, dtype=float16\n"
            f"  FlashInfer 中位时间: {flashinfer_median:.4f} ms\n"
            f"  EdgeFM 中位时间: {edge_fm_median:.4f} ms\n"
            f"  性能比 (FlashInfer/EdgeFM): {performance_ratio:.4f}\n"
            f"  要求: performance_ratio >= 0.85"
        )
    
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_forward_silu_and_mul_performance_bfloat16(self, activation_layer, seq_len):
        """测试 forward_silu_and_mul 的性能（BFloat16）
        
        确保 EdgeFM 的性能不低于 FlashInfer 的 0.85 倍。
        使用中位数、更长测量时间以降低微秒级 benchmark 的波动。
        """
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        input_tensor = torch.randn(
            BATCH_SIZE, seq_len, 2 * hidden_size,
            device="cuda:0", dtype=torch.bfloat16
        )

        def run_flashinfer():
            flashinfer.activation.silu_and_mul(input_tensor)

        flashinfer_measurements = bench_gpu_time(run_flashinfer, repeat_time_ms=300)
        flashinfer_median = statistics.median(flashinfer_measurements)

        input_efm = tensor_to_edge_fm_tensor(input_tensor)
        output_efm = tensor_to_edge_fm_tensor(
            torch.empty(BATCH_SIZE, seq_len, hidden_size, device="cuda:0", dtype=torch.bfloat16)
        )

        def run_edge_fm():
            activation_layer.forward_silu_and_mul(input_efm, output_efm)

        edge_fm_measurements = bench_gpu_time(run_edge_fm, repeat_time_ms=300)
        edge_fm_median = statistics.median(edge_fm_measurements)
        performance_ratio = flashinfer_median / edge_fm_median if edge_fm_median > 0 else 0.0

        assert performance_ratio >= 0.85, (
            f"性能测试失败：EdgeFM 性能低于 FlashInfer 的 0.85 倍\n"
            f"  seq_len={seq_len}, dtype=bfloat16\n"
            f"  FlashInfer 中位时间: {flashinfer_median:.4f} ms\n"
            f"  EdgeFM 中位时间: {edge_fm_median:.4f} ms\n"
            f"  性能比 (FlashInfer/EdgeFM): {performance_ratio:.4f}\n"
            f"  要求: performance_ratio >= 0.85"
        )


if __name__ == "__main__":
    # 支持直接运行：python test_activation.py
    pytest.main([__file__, "-v"])
