"""
RMSNorm 层的正确性和性能测试（pytest 单元测试）

测试 edge_fm.RMSNormLayer 的实现，使用 flashinfer 作为参考实现。
包括：
1. 正确性测试：测试 forward_rmsnorm 和 forward_fused_add_rmsnorm
2. 性能测试：确保性能不低于 flashinfer 的 0.9 倍
"""

import sys
import json
import torch
import pytest
import tempfile
from pathlib import Path
from safetensors import safe_open

# 添加构建目录到路径
project_root = Path(__file__).parent.parent.parent
build_python = project_root / "build" / "install" / "python"
sys.path.insert(0, str(build_python))

import edge_fm
import flashinfer
from flashinfer.testing.utils import bench_gpu_time

# 模型配置
model_path = project_root / "examples" / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct"
safetensors_path = model_path / "model.safetensors"

with open(model_path / "config.json", "r") as f:
    config = json.load(f)
    hidden_size = config.get("hidden_size", 896)


def create_rmsnorm_layer(layer_id, model_path):
    """创建 RMSNormLayer 实例
    
    参数:
        layer_id: 层 ID（用于确定加载哪个层的权重）
        model_path: 模型路径（可选），如果提供则使用该路径，否则使用默认路径
                    默认路径为 examples/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct
    """
    loader = edge_fm.WeightLoader.instance()
    loader.clear_stage(edge_fm.ModelStage.Prefill)
    loader.clear_stage(edge_fm.ModelStage.Decode)

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
    return edge_fm.RMSNormLayer(layer_id=layer_id, config_path=str(engine_config_path))


@pytest.fixture(scope="module")
def rmsnorm_weight():
    """加载 RMSNorm 权重（模块级 fixture，直接从 safetensors 文件加载）"""
    possible_names = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ]
    
    with safe_open(str(safetensors_path), framework="pt", device="cuda:0") as f:
        for name in possible_names:
            if name in f.keys():
                weight_torch = f.get_tensor(name)
                
                if weight_torch.shape[0] != hidden_size:
                    pytest.skip(f"权重的 hidden_size ({weight_torch.shape[0]}) 与配置的 hidden_size ({hidden_size}) 不匹配")
                
                return weight_torch
    pytest.skip("无法在 safetensors 文件中找到所需的 RMSNorm 权重")


def tensor_to_edge_fm_tensor(torch_tensor):
    """将 PyTorch tensor 转换为 edge_fm.Tensor（通过 DLPack）
    需确保 tensor 是 contiguous 且 row-major。对已 contiguous 的 tensor，
    .contiguous() 是空操作；对 view/transpose 等则会产生连续拷贝。
    """
    capsule = torch_tensor.contiguous().__dlpack__()
    return edge_fm.Tensor.from_dlpack(capsule)


# 测试参数
BATCH_SIZES = [1, 32, 64, 128]
EPS = 1e-6
SPEED_RATIO = 0.9


class TestRMSNorm:
    """RMSNorm 层测试类（包括正确性和性能测试）"""
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_forward_rmsnorm_correctness(self, rmsnorm_weight, batch_size):
        """测试 forward_rmsnorm 的正确性
        
        参数:
            rmsnorm_weight: RMSNorm 权重（pytest fixture）
            batch_size: 批次大小（pytest 参数化）
        """
        # 在每个测试中创建新的 layer 实例
        rmsnorm_layer = create_rmsnorm_layer(layer_id=0, model_path=model_path)
        
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        weight_dtype = rmsnorm_weight.dtype
        input_tensor = torch.randn(batch_size, hidden_size, device="cuda:0", dtype=weight_dtype)
        
        # FlashInfer 参考实现
        output_flashinfer = flashinfer.rmsnorm(input_tensor, rmsnorm_weight, eps=EPS)
        
        # 转换为 edge_fm.Tensor
        input_efm = tensor_to_edge_fm_tensor(input_tensor)
        output_efm = tensor_to_edge_fm_tensor(torch.empty_like(input_tensor, dtype=weight_dtype))
        
        # 执行 forward_rmsnorm
        rmsnorm_layer.forward_rmsnorm(input_efm, output_efm)
        output_efm_torch = torch.from_dlpack(output_efm.to_dlpack())
        torch.cuda.synchronize()
        
        # 使用标准的 torch.testing.assert_close 进行断言
        torch.testing.assert_close(
            output_efm_torch, 
            output_flashinfer, 
            rtol=1e-3, 
            atol=1e-3,
            msg=f"forward_rmsnorm 结果与 FlashInfer 不一致 (batch_size={batch_size}, dtype={weight_dtype})"
        )
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_forward_fused_add_rmsnorm_correctness(self, rmsnorm_weight, batch_size):
        """测试 forward_fused_add_rmsnorm 的正确性
        
        参数:
            rmsnorm_weight: RMSNorm 权重（pytest fixture）
            batch_size: 批次大小（pytest 参数化）
        """
        # 在每个测试中创建新的 layer 实例
        rmsnorm_layer = create_rmsnorm_layer(layer_id=0, model_path=model_path)
        
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        weight_dtype = rmsnorm_weight.dtype
        input_tensor = torch.randn(batch_size, hidden_size, device="cuda:0", dtype=weight_dtype)
        residual_tensor = torch.randn(batch_size, hidden_size, device="cuda:0", dtype=weight_dtype)
        
        # FlashInfer 参考实现
        input_flashinfer = input_tensor.clone()
        residual_flashinfer = residual_tensor.clone()
        flashinfer.fused_add_rmsnorm(input_flashinfer, residual_flashinfer, rmsnorm_weight, eps=EPS)
        
        # 转换为 edge_fm.Tensor
        inout_efm = tensor_to_edge_fm_tensor(input_tensor.clone())
        residual_efm = tensor_to_edge_fm_tensor(residual_tensor.clone())
        
        # 执行 forward_fused_add_rmsnorm
        rmsnorm_layer.forward_fused_add_rmsnorm(inout_efm, residual_efm)
        inout_efm_torch = torch.from_dlpack(inout_efm.to_dlpack())
        residual_efm_torch = torch.from_dlpack(residual_efm.to_dlpack())
        torch.cuda.synchronize()
        
        # 使用标准的 torch.testing.assert_close 进行断言
        torch.testing.assert_close(
            inout_efm_torch, 
            input_flashinfer, 
            rtol=1e-3, 
            atol=1e-3,
            msg=f"forward_fused_add_rmsnorm inout 结果与 FlashInfer 不一致 (batch_size={batch_size}, dtype={weight_dtype})"
        )
        torch.testing.assert_close(
            residual_efm_torch, 
            residual_flashinfer, 
            rtol=1e-3, 
            atol=1e-3,
            msg=f"forward_fused_add_rmsnorm residual 结果与 FlashInfer 不一致 (batch_size={batch_size}, dtype={weight_dtype})"
        )
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES + [256])
    def test_forward_rmsnorm_performance(self, rmsnorm_weight, batch_size):
        """测试 forward_rmsnorm 的性能
        
        确保 EdgeFM 的性能不低于 FlashInfer 的 0.9 倍。
        
        参数:
            rmsnorm_weight: RMSNorm 权重（pytest fixture）
            batch_size: 批次大小（pytest 参数化）
        """
        # 在每个测试中创建新的 layer 实例
        rmsnorm_layer = create_rmsnorm_layer(layer_id=0, model_path=model_path)
        
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        weight_dtype = rmsnorm_weight.dtype
        input_tensor = torch.randn(batch_size, hidden_size, device="cuda:0", dtype=weight_dtype)
        
        # 测试 FlashInfer 性能
        def run_flashinfer():
            flashinfer.rmsnorm(input_tensor, rmsnorm_weight, eps=EPS)
        
        flashinfer_measurements = bench_gpu_time(run_flashinfer, repeat_time_ms=100)
        flashinfer_avg = sum(flashinfer_measurements) / len(flashinfer_measurements)
        
        # 测试 edge_fm 性能
        input_efm = tensor_to_edge_fm_tensor(input_tensor)
        output_efm = tensor_to_edge_fm_tensor(torch.empty_like(input_tensor, dtype=weight_dtype))
        
        def run_edge_fm():
            rmsnorm_layer.forward_rmsnorm(input_efm, output_efm)
        
        edge_fm_measurements = bench_gpu_time(run_edge_fm, repeat_time_ms=100)
        edge_fm_avg = sum(edge_fm_measurements) / len(edge_fm_measurements)
        
        # 计算性能比：FlashInfer 时间 / EdgeFM 时间
        performance_ratio = flashinfer_avg / edge_fm_avg if edge_fm_avg > 0 else 0.0
        
        # 断言：EdgeFM 性能不低于 FlashInfer 的 SPEED_RATIO 倍
        assert performance_ratio >= SPEED_RATIO, (
            f"性能测试失败：EdgeFM 性能低于 FlashInfer 的 {SPEED_RATIO} 倍\n"
            f"  batch_size={batch_size}, dtype={weight_dtype}\n"
            f"  FlashInfer 平均时间: {flashinfer_avg:.4f} ms\n"
            f"  EdgeFM 平均时间: {edge_fm_avg:.4f} ms\n"
            f"  性能比 (FlashInfer/EdgeFM): {performance_ratio:.4f}\n"
            f"  要求: performance_ratio >= {SPEED_RATIO}"
        )
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES + [256])
    def test_forward_fused_add_rmsnorm_performance(self, rmsnorm_weight, batch_size):
        """测试 forward_fused_add_rmsnorm 的性能
        
        确保 EdgeFM 的性能不低于 FlashInfer 的 0.8 倍。
        
        参数:
            rmsnorm_weight: RMSNorm 权重（pytest fixture）
            batch_size: 批次大小（pytest 参数化）
        """
        # 在每个测试中创建新的 layer 实例
        rmsnorm_layer = create_rmsnorm_layer(layer_id=0, model_path=model_path)
        
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        weight_dtype = rmsnorm_weight.dtype
        input_tensor = torch.randn(batch_size, hidden_size, device="cuda:0", dtype=weight_dtype)
        residual_tensor = torch.randn(batch_size, hidden_size, device="cuda:0", dtype=weight_dtype)
        
        # 测试 FlashInfer 性能
        def run_flashinfer():
            flashinfer.fused_add_rmsnorm(input_tensor, residual_tensor, rmsnorm_weight, eps=EPS)
        
        flashinfer_measurements = bench_gpu_time(run_flashinfer, repeat_time_ms=100)
        flashinfer_avg = sum(flashinfer_measurements) / len(flashinfer_measurements)
        
        # 测试 edge_fm 性能
        inout_efm = tensor_to_edge_fm_tensor(input_tensor)
        residual_efm = tensor_to_edge_fm_tensor(residual_tensor)
        
        def run_edge_fm():
            rmsnorm_layer.forward_fused_add_rmsnorm(inout_efm, residual_efm)
        
        edge_fm_measurements = bench_gpu_time(run_edge_fm, repeat_time_ms=100)
        edge_fm_avg = sum(edge_fm_measurements) / len(edge_fm_measurements)
        
        # 计算性能比：FlashInfer 时间 / EdgeFM 时间
        performance_ratio = flashinfer_avg / edge_fm_avg if edge_fm_avg > 0 else 0.0
        
        # 断言：EdgeFM 性能不低于 FlashInfer 的 SPEED_RATIO 倍
        assert performance_ratio >= SPEED_RATIO, (
            f"性能测试失败：EdgeFM 性能低于 FlashInfer 的 {SPEED_RATIO} 倍\n"
            f"  batch_size={batch_size}, dtype={weight_dtype}\n"
            f"  FlashInfer 平均时间: {flashinfer_avg:.4f} ms\n"
            f"  EdgeFM 平均时间: {edge_fm_avg:.4f} ms\n"
            f"  性能比 (FlashInfer/EdgeFM): {performance_ratio:.4f}\n"
            f"  要求: performance_ratio >= {SPEED_RATIO}"
        )


if __name__ == "__main__":
    # 支持直接运行：python test_layernorm.py
    pytest.main([__file__, "-v"])
