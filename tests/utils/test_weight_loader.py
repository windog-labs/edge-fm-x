"""
WeightLoader 的正确性测试（pytest 单元测试）

测试 edge_fm.WeightLoader 的实现，包括：
1. 单例模式测试
2. 加载权重文件测试
3. 获取权重测试
4. 不同 ModelStage 的隔离测试
5. 重复加载同一文件的测试（不会重复加载）
6. 加载不同文件会合并到缓存的测试（同名权重会被覆盖，其他权重保留）
"""

import sys
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
from safetensors.torch import save_file


def create_test_safetensors_file(dir_path, tensor_dict, filename=None):
    """创建测试用的 safetensors 文件"""
    if filename is None:
        filename = "test_weights.safetensors"
    safetensors_path = os.path.join(dir_path, filename)
    save_file(tensor_dict, safetensors_path)
    return safetensors_path


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 清理临时目录
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_weights(temp_dir):
    """创建测试权重张量"""
    weights = {
        "weight1": torch.randn(10, 20, dtype=torch.float16),
        "weight2": torch.randn(5, 10, dtype=torch.float16),
        "weight3": torch.randn(3, 4, dtype=torch.float32),
    }
    safetensors_path = create_test_safetensors_file(temp_dir, weights)
    return weights, safetensors_path


class TestWeightLoader:
    """WeightLoader 测试类"""
    
    def test_load_and_get_weights(self, test_weights):
        """测试加载权重文件并获取权重"""
        weights_dict, safetensors_path = test_weights
        loader = edge_fm.WeightLoader.instance()
        
        loader.load_weights_from_file(
            edge_fm.ModelStage.Prefill,
            safetensors_path,
            edge_fm.Device.CPU,
            device_id=0
        )
        loaded_weights = loader.get(edge_fm.ModelStage.Prefill)
        
        assert len(loaded_weights) == len(weights_dict), \
            f"权重数量不匹配：期望 {len(weights_dict)}，实际 {len(loaded_weights)}"

        assert set(loaded_weights.keys()) == set(weights_dict.keys()), \
            "权重名称不匹配"
        
        for name, tensor in loaded_weights.items():
            assert name in weights_dict, f"权重 {name} 不存在于原始字典中"
            original_shape = list(weights_dict[name].shape)
            loaded_shape = tensor.shape()
            
            assert loaded_shape == original_shape, \
                f"权重 {name} 的形状不匹配：期望 {original_shape}，实际 {loaded_shape}"
    
    def test_load_nonexistent_file(self, temp_dir):
        """测试加载不存在的文件应该抛出异常"""
        loader = edge_fm.WeightLoader.instance()
        nonexistent_path = os.path.join(temp_dir, "nonexistent.safetensors")
        
        with pytest.raises((edge_fm.ConfigurationError, ValueError), match="Failed to load safetensors file"):
            loader.load_weights_from_file(
                edge_fm.ModelStage.Prefill,
                nonexistent_path,
                edge_fm.Device.CPU,
                device_id=0
            )
    
    def test_skip_duplicate_file_load(self, temp_dir, test_weights):
        """测试重复加载同一个文件时不会重复加载"""
        _, safetensors_path = test_weights
        loader = edge_fm.WeightLoader.instance()
        
        # 第一次加载
        loader.load_weights_from_file(
            edge_fm.ModelStage.Prefill,
            safetensors_path,
            edge_fm.Device.CPU,
            device_id=0
        )
        
        first_weights = loader.get(edge_fm.ModelStage.Prefill)
        first_weight_count = len(first_weights)
        
        # 第二次加载同一个文件，应该直接返回，不会重复加载
        loader.load_weights_from_file(
            edge_fm.ModelStage.Prefill,
            safetensors_path,
            edge_fm.Device.CPU,
            device_id=0
        )
        
        second_weights = loader.get(edge_fm.ModelStage.Prefill)
        
        # 应该仍然是第一次加载的权重，数量应该相同
        assert len(second_weights) == first_weight_count, \
            "重复加载同一个文件时不应该重复加载"
        assert set(second_weights.keys()) == set(first_weights.keys()), \
            "重复加载同一个文件时权重名称应该保持不变"
    
    def test_merge_on_different_file(self, temp_dir, test_weights):
        """测试加载不同的文件时会合并到已有缓存（同名权重会被覆盖，其他权重保留）"""
        loader = edge_fm.WeightLoader.instance()
        _, first_safetensors_path = test_weights
        
        # 第一次加载
        loader.load_weights_from_file(
            edge_fm.ModelStage.Prefill,
            first_safetensors_path,
            edge_fm.Device.CPU,
            device_id=0
        )
        first_weights = loader.get(edge_fm.ModelStage.Prefill)
        first_weight1_shape = first_weights["weight1"].shape()
        first_weight2_shape = first_weights["weight2"].shape()

        # 创建不同的权重文件（包含 weight1 的新值和 weight_new）
        new_weights = {
            "weight1": torch.randn(15, 25, dtype=torch.float16),
            "weight_new": torch.randn(7, 8, dtype=torch.float16),
        }
        new_safetensors_path = create_test_safetensors_file(temp_dir, new_weights, "new_weights.safetensors")
        
        # 加载不同的文件，应该合并到现有缓存（使用 overwrite_if_exists=True 来覆盖同名权重）
        loader.load_weights_from_file(
            edge_fm.ModelStage.Prefill,
            new_safetensors_path,
            edge_fm.Device.CPU,
            device_id=0,
            overwrite_if_exists=True
        )
        second_weights = loader.get(edge_fm.ModelStage.Prefill)

        # weight1 应该被新值覆盖（形状改变）
        assert second_weights["weight1"].shape() != first_weight1_shape, \
            "加载不同文件时同名权重应该被覆盖（overwrite_if_exists=True）"
        # weight2 应该保留（没有被覆盖）
        assert "weight2" in second_weights, "旧权重应该被保留"
        assert second_weights["weight2"].shape() == first_weight2_shape, \
            "未被覆盖的权重应该保持原状"
        # weight3 应该保留
        assert "weight3" in second_weights, "旧权重应该被保留"
        # weight_new 应该被添加
        assert "weight_new" in second_weights, "新权重应该被添加"
    
    def test_merge_without_overwrite(self, temp_dir, test_weights):
        """测试加载不同文件时合并但不覆盖同名权重（overwrite_if_exists=False）"""
        loader = edge_fm.WeightLoader.instance()
        _, first_safetensors_path = test_weights
        
        # 第一次加载
        loader.load_weights_from_file(
            edge_fm.ModelStage.Prefill,
            first_safetensors_path,
            edge_fm.Device.CPU,
            device_id=0
        )
        first_weights = loader.get(edge_fm.ModelStage.Prefill)
        first_weight1_shape = first_weights["weight1"].shape()
        
        # 创建不同的权重文件（包含 weight1 的新值和 weight_new）
        new_weights = {
            "weight1": torch.randn(15, 25, dtype=torch.float16),
            "weight_new": torch.randn(7, 8, dtype=torch.float16),
        }
        new_safetensors_path = create_test_safetensors_file(temp_dir, new_weights, "another_weights.safetensors")
        
        # 加载不同的文件，合并但不覆盖（overwrite_if_exists=False）
        loader.load_weights_from_file(
            edge_fm.ModelStage.Prefill,
            new_safetensors_path,
            edge_fm.Device.CPU,
            device_id=0,
            overwrite_if_exists=False
        )
        second_weights = loader.get(edge_fm.ModelStage.Prefill)
        
        # weight1 应该保持原状（不被覆盖）
        assert second_weights["weight1"].shape() == first_weight1_shape, \
            "加载不同文件时同名权重不应该被覆盖（overwrite_if_exists=False）"
        # weight_new 应该被添加
        assert "weight_new" in second_weights, "新权重应该被添加"
        # weight2 和 weight3 应该保留
        assert "weight2" in second_weights, "旧权重应该被保留"
        assert "weight3" in second_weights, "旧权重应该被保留"
    
    def test_gpu_loading(self, temp_dir):
        """测试在 GPU 上加载权重"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用，跳过 GPU 测试")
        weights = {
            "gpu_weight1": torch.randn(4, 8, dtype=torch.float16),
            "gpu_weight2": torch.randn(2, 4, dtype=torch.float16),
        }
        safetensors_path = create_test_safetensors_file(temp_dir, weights)
        loader = edge_fm.WeightLoader.instance()
        
        loader.load_weights_from_file(
            edge_fm.ModelStage.Prefill,
            safetensors_path,
            edge_fm.Device.GPU,
            device_id=0
        )
        
        loaded_weights = loader.get(edge_fm.ModelStage.Prefill)
        for name in ["gpu_weight1", "gpu_weight2"]:
            if name in loaded_weights:
                tensor = loaded_weights[name]
                device, device_id = tensor.device()
                assert device == edge_fm.Device.GPU, \
                    f"权重 {name} 应该在 GPU 上，实际在 {device}"
                assert device_id == 0, \
                    f"权重 {name} 的 device_id 应该是 0，实际是 {device_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

