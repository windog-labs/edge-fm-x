"""
Tensor 类的 Python 接口测试（pytest 单元测试）

测试 pybind11 暴露出来的 Tensor 核心接口，包括：
- 工厂方法: view, adopt, clone_from
- DLPack 互操作: from_dlpack, to_dlpack
- 属性访问: shape, dtype, device, empty, data_ptr
- 数据导出: dump
"""

import os
import sys
import torch
import pytest
import tempfile
import numpy as np
from pathlib import Path

# 添加构建目录到路径
project_root = Path(__file__).parent.parent.parent
build_python = project_root / "build" / "install" / "python"
sys.path.insert(0, str(build_python))
import edge_fm


class TestTensorFactoryMethods:
    """测试 Tensor 工厂方法"""
    
    def test_tensor_view_cpu(self):
        """测试 Tensor::view 创建 CPU 非拥有视图"""
        # 创建 NumPy array
        np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        data_ptr = np_array.ctypes.data
        
        # 创建非拥有视图
        tensor = edge_fm.Tensor.view(
            data_ptr=data_ptr,
            shape=[2, 3],
            dtype=edge_fm.DType.Float32,
            device=edge_fm.Device.CPU
        )
        
        # 验证属性
        assert not tensor.empty()
        assert tensor.shape() == [2, 3]
        assert tensor.dtype() == edge_fm.DType.Float32
        device, device_id = tensor.device()
        assert device == edge_fm.Device.CPU
        assert tensor.data_ptr() == data_ptr
        
        print("\n✓ test_tensor_view_cpu passed")
    
    def test_tensor_clone_from_cpu_to_cpu(self):
        """测试 Tensor::clone_from CPU -> CPU 克隆"""
        # 创建源数据
        np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        src_ptr = np_array.ctypes.data
        
        # 克隆数据
        tensor = edge_fm.Tensor.clone_from(
            src_ptr=src_ptr,
            shape=[2, 2],
            dtype=edge_fm.DType.Float32,
            src_device=edge_fm.Device.CPU,
            src_device_id=0,
            dst_device=edge_fm.Device.CPU,
            dst_device_id=0,
            ownership=edge_fm.MemoryOwnership.OwnCpuMalloc
        )
        
        # 验证属性
        assert not tensor.empty()
        assert tensor.shape() == [2, 2]
        assert tensor.dtype() == edge_fm.DType.Float32
        device, _ = tensor.device()
        assert device == edge_fm.Device.CPU
        # 数据指针应该不同（已克隆）
        assert tensor.data_ptr() != src_ptr
        
        # 验证数据被正确复制
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            dump_path = f.name
        try:
            tensor.dump(dump_path)
            with open(dump_path, 'r') as f:
                content = f.read()
            assert "1 2 3 4" in content
        finally:
            if os.path.exists(dump_path):
                os.unlink(dump_path)
        
        print("\n✓ test_tensor_clone_from_cpu_to_cpu passed")
    
    def test_tensor_clone_from_cpu_to_gpu(self):
        """测试 Tensor::clone_from CPU -> GPU 克隆"""
        torch_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).contiguous()
        src_ptr = torch_tensor.data_ptr()
        
        # 克隆到 GPU
        tensor = edge_fm.Tensor.clone_from(
            src_ptr=src_ptr,
            shape=[4],
            dtype=edge_fm.DType.Float32,
            src_device=edge_fm.Device.CPU,
            src_device_id=0,
            dst_device=edge_fm.Device.GPU,
            dst_device_id=0,
            ownership=edge_fm.MemoryOwnership.OwnCudaMalloc
        )
        
        # 验证属性
        assert not tensor.empty()
        assert tensor.shape() == [4]
        assert tensor.dtype() == edge_fm.DType.Float32
        device, device_id = tensor.device()
        assert device == edge_fm.Device.GPU
        assert device_id == 0
        
        # 验证数据（会自动从 GPU 拷贝到 CPU 进行 dump）
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            dump_path = f.name
        try:
            tensor.dump(dump_path)
            with open(dump_path, 'r') as f:
                content = f.read()
            assert "1 2 3 4" in content
        finally:
            if os.path.exists(dump_path):
                os.unlink(dump_path)
        
        print("\n✓ test_tensor_clone_from_cpu_to_gpu passed")


class TestTensorDLPackInterop:
    """测试 Tensor 与 DLPack 的互操作"""
    
    def test_tensor_from_dlpack_torch(self):
        """测试从 PyTorch tensor 通过 DLPack 创建 Tensor"""
        torch_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
        capsule = torch_tensor.__dlpack__()
        tensor = edge_fm.Tensor.from_dlpack(capsule)

        # 验证基本属性
        assert not tensor.empty()
        assert tensor.dtype() == edge_fm.DType.Float32
        assert tensor.shape() == [2, 4]

        device, device_id = tensor.device()
        assert device == edge_fm.Device.CPU
        assert device_id == 0

        # 验证数据
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            dump_path = f.name
        try:
            tensor.dump(dump_path)
            with open(dump_path, 'r') as f:
                content = f.read()
            assert "1 2 3 4 5 6 7 8" in content
        finally:
            if os.path.exists(dump_path):
                os.unlink(dump_path)
        
        print("\n✓ test_tensor_from_dlpack_torch passed")

    def test_tensor_from_dlpack_numpy(self):
        """测试从 NumPy array 通过 DLPack 创建 Tensor"""
        np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        capsule = np_array.__dlpack__()
        tensor = edge_fm.Tensor.from_dlpack(capsule)
        
        assert not tensor.empty()
        assert tensor.dtype() == edge_fm.DType.Float32
        assert tensor.shape() == [2, 3]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            dump_path = f.name
        try:
            tensor.dump(dump_path)
            with open(dump_path, 'r') as f:
                content = f.read()
            assert "1 2 3 4 5 6" in content
        finally:
            if os.path.exists(dump_path):
                os.unlink(dump_path)
        
        print("\n✓ test_tensor_from_dlpack_numpy passed")
    
    def test_tensor_to_dlpack_and_back(self):
        """测试 Tensor 转换为 DLPack 并被 PyTorch 使用"""
        # 创建 Tensor
        np_array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        tensor = edge_fm.Tensor.clone_from(
            src_ptr=np_array.ctypes.data,
            shape=[4],
            dtype=edge_fm.DType.Float32,
            src_device=edge_fm.Device.CPU,
            src_device_id=0,
            dst_device=edge_fm.Device.CPU,
            dst_device_id=0,
            ownership=edge_fm.MemoryOwnership.OwnCpuMalloc
        )
        
        # 转换为 DLPack capsule
        capsule = tensor.to_dlpack()
        
        # PyTorch 从 capsule 创建 tensor（共享内存）
        torch_tensor = torch.from_dlpack(capsule)
        
        # 验证数据
        assert torch_tensor.shape == (4,)
        assert torch_tensor.dtype == torch.float32
        assert torch.allclose(torch_tensor, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        
        print("\n✓ test_tensor_to_dlpack_and_back passed")


class TestTensorProperties:
    """测试 Tensor 属性访问"""
    
    def test_tensor_empty_check(self):
        """测试空张量检查"""
        # 创建空张量
        np_array = np.array([], dtype=np.float32)
        tensor = edge_fm.Tensor.clone_from(
            src_ptr=np_array.ctypes.data,
            shape=[0],
            dtype=edge_fm.DType.Float32,
            src_device=edge_fm.Device.CPU,
            src_device_id=0,
            dst_device=edge_fm.Device.CPU,
            dst_device_id=0,
            ownership=edge_fm.MemoryOwnership.OwnCpuMalloc
        )
        
        # 应该是空的
        assert tensor.empty()
        assert tensor.shape() == [0]
        
        print("\n✓ test_tensor_empty_check passed")
    
    def test_tensor_dtype_variations(self):
        """测试不同数据类型的 Tensor"""
        dtypes_map = [
            (np.float32, edge_fm.DType.Float32),
            (np.int32, edge_fm.DType.Int32),
            (np.int64, edge_fm.DType.Int64),
        ]
        
        for np_dtype, edge_dtype in dtypes_map:
            np_array = np.array([1, 2, 3], dtype=np_dtype)
            tensor = edge_fm.Tensor.clone_from(
                src_ptr=np_array.ctypes.data,
                shape=[3],
                dtype=edge_dtype,
                src_device=edge_fm.Device.CPU,
                src_device_id=0,
                dst_device=edge_fm.Device.CPU,
                dst_device_id=0,
                ownership=edge_fm.MemoryOwnership.OwnCpuMalloc
            )
            assert tensor.dtype() == edge_dtype
            assert not tensor.empty()
        
        print("\n✓ test_tensor_dtype_variations passed")


if __name__ == "__main__":
    # 支持直接运行：python test_tensor.py
    pytest.main([__file__, "-v", "-s"])
