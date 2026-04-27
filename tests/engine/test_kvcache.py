"""
KVManager 的正确性测试（pytest 单元测试）

测试 edge_fm.KVManager 的实现，包括：
1. 基本功能测试：创建 KVManager、获取状态、检查请求有效性
2. MHA/GQA 模式测试：测试 common KV cache 分配
3. MLA 模式测试：测试 MLA KV cache 分配
4. 错误处理测试：测试无效 request_id 的处理
"""

import sys
import json
import pytest
import tempfile
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.edge_fm_build_paths import prepend_built_python_paths

prepend_built_python_paths(project_root)

import edge_fm


def create_engine_config(model_config, kvcache_config, temp_dir):
    """创建 engine_config.json 文件"""
    engine_config_dir = tempfile.mkdtemp()
    engine_config_path = os.path.join(engine_config_dir, "engine_config.json")
    
    # 创建模型配置文件
    config_path = os.path.join(temp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f)
    
    # 创建 engine_config.json
    engine_config = {
        "runtime": {
            "device": "cuda",
            "device_id": 0
        },
        "prefill_model_path": temp_dir,
        "kvcache": kvcache_config
    }
    with open(engine_config_path, "w") as f:
        json.dump(engine_config, f)
    
    return engine_config_path


# ============================================================================
# MHA/GQA 模式测试
# ============================================================================

def test_mha_kvcache_basic():
    """测试 MHA 模式的基本 KV cache 分配"""
    temp_dir = tempfile.mkdtemp()
    
    model_config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,  # MHA: num_attention_heads == num_key_value_heads
        "hidden_size": 1024
    }
    
    kvcache_config = {
        "attention_type": "mha",
        "dtype": "fp16",
        "requests": [
            {
                "request_id": 0,
                "max_tokens": 1024,
                "prefix_token_ids": []
            },
            {
                "request_id": 1,
                "max_tokens": 2048,
                "prefix_token_ids": [1, 2, 3]
            }
        ]
    }
    
    engine_config_path = create_engine_config(model_config, kvcache_config, temp_dir)
    kv_manager = edge_fm.KVManager(engine_config_path)
    
    # 测试基本功能
    assert kv_manager.is_request_valid(0), "Request 0 should be valid"
    assert kv_manager.is_request_valid(1), "Request 1 should be valid"
    assert not kv_manager.is_request_valid(999), "Request 999 should be invalid"
    
    # 测试获取状态
    status = kv_manager.get_status()
    assert status.device == edge_fm.Device.CPU, "Standalone KVManager should use the default host allocator"
    assert status.device_id == 0, "Device ID should be 0"
    assert len(status.slots) == 2, "Should have 2 slots"
    
    # 检查 slot 信息
    slot_0 = next(slot for slot in status.slots if slot.request_id == 0)
    assert slot_0.max_tokens == 1024, "Slot 0 max_tokens should be 1024"
    assert slot_0.prefix_size == 0, "Slot 0 prefix_size should be 0"
    assert slot_0.allocated_size > 0, "Slot 0 should have allocated memory"
    
    slot_1 = next(slot for slot in status.slots if slot.request_id == 1)
    assert slot_1.max_tokens == 2048, "Slot 1 max_tokens should be 2048"
    assert slot_1.prefix_size == 3, "Slot 1 prefix_size should be 3"
    assert slot_1.prefix_token_ids == [1, 2, 3], "Slot 1 prefix_token_ids should be [1, 2, 3]"
    assert slot_1.allocated_size > 0, "Slot 1 should have allocated memory"
    
    # 测试获取 KV cache 指针
    read_ptrs_0 = kv_manager.get_read_kvcache(0)
    write_ptrs_0 = kv_manager.get_write_kvcache(0)
    assert len(read_ptrs_0) == 2, "Should have 2 layers"
    assert len(write_ptrs_0) == 2, "Should have 2 layers"
    assert all(ptr > 0 for ptr in read_ptrs_0), "All read pointers should be valid"
    assert all(ptr > 0 for ptr in write_ptrs_0), "All write pointers should be valid"
    # 对于没有 prefix 的情况，read 和 write 指针应该相同
    assert read_ptrs_0 == write_ptrs_0, "For slot without prefix, read and write pointers should be the same"
    
    read_ptrs_1 = kv_manager.get_read_kvcache(1)
    write_ptrs_1 = kv_manager.get_write_kvcache(1)
    assert len(read_ptrs_1) == 2, "Should have 2 layers"
    assert len(write_ptrs_1) == 2, "Should have 2 layers"
    # 对于有 prefix 的情况，write 指针应该大于 read 指针
    assert all(write_ptr > read_ptr for read_ptr, write_ptr in zip(read_ptrs_1, write_ptrs_1)), \
        "For slot with prefix, write pointers should be greater than read pointers"


def test_gqa_kvcache():
    """测试 GQA 模式的 KV cache 分配"""
    temp_dir = tempfile.mkdtemp()
    
    model_config = {
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,  # GQA: num_attention_heads > num_key_value_heads
        "hidden_size": 2048
    }
    
    kvcache_config = {
        "attention_type": "gqa",
        "dtype": "fp16",
        "requests": [
            {
                "request_id": 0,
                "max_tokens": 512
            }
        ]
    }
    
    engine_config_path = create_engine_config(model_config, kvcache_config, temp_dir)
    kv_manager = edge_fm.KVManager(engine_config_path)
    
    status = kv_manager.get_status()
    assert len(status.slots) == 1, "Should have 1 slot"
    
    slot = status.slots[0]
    assert slot.max_tokens == 512, "Max tokens should be 512"
    assert slot.allocated_size > 0, "Should have allocated memory"
    
    read_ptrs = kv_manager.get_read_kvcache(0)
    write_ptrs = kv_manager.get_write_kvcache(0)
    assert len(read_ptrs) == 4, "Should have 4 layers"
    assert len(write_ptrs) == 4, "Should have 4 layers"


def test_mla_kvcache():
    """测试 MLA 模式的 KV cache 分配"""
    temp_dir = tempfile.mkdtemp()
    
    model_config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "hidden_size": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64
    }
    
    kvcache_config = {
        "attention_type": "mla",
        "dtype": "fp16",
        "requests": [
            {
                "request_id": 0,
                "max_tokens": 1024,
                "prefix_token_ids": [10, 20, 30]
            }
        ]
    }
    
    engine_config_path = create_engine_config(model_config, kvcache_config, temp_dir)
    kv_manager = edge_fm.KVManager(engine_config_path)
    
    status = kv_manager.get_status()
    assert len(status.slots) == 1, "Should have 1 slot"
    
    slot = status.slots[0]
    assert slot.max_tokens == 1024, "Max tokens should be 1024"
    assert slot.prefix_size == 3, "Prefix size should be 3"
    assert slot.prefix_token_ids == [10, 20, 30], "Prefix token IDs should match"
    assert slot.allocated_size > 0, "Should have allocated memory"
    
    read_ptrs = kv_manager.get_read_kvcache(0)
    write_ptrs = kv_manager.get_write_kvcache(0)
    assert len(read_ptrs) == 2, "Should have 2 layers"
    assert len(write_ptrs) == 2, "Should have 2 layers"
    # 对于有 prefix 的情况，write 指针应该大于 read 指针
    assert all(write_ptr > read_ptr for read_ptr, write_ptr in zip(read_ptrs, write_ptrs)), \
        "For slot with prefix, write pointers should be greater than read pointers"


def test_invalid_request_id():
    """测试无效 request_id 的错误处理"""
    temp_dir = tempfile.mkdtemp()
    
    model_config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_size": 1024
    }
    
    kvcache_config = {
        "attention_type": "mha",
        "dtype": "fp16",
        "requests": [
            {
                "request_id": 0,
                "max_tokens": 1024
            }
        ]
    }
    
    engine_config_path = create_engine_config(model_config, kvcache_config, temp_dir)
    kv_manager = edge_fm.KVManager(engine_config_path)
    
    # 测试无效 request_id
    with pytest.raises(edge_fm.InvalidRequestError):
        kv_manager.get_read_kvcache(999)
    
    with pytest.raises(edge_fm.InvalidRequestError):
        kv_manager.get_write_kvcache(999)
    
    # 测试 is_request_valid
    assert not kv_manager.is_request_valid(999), "Invalid request should return False"


def test_multiple_requests():
    """测试多个请求的管理"""
    temp_dir = tempfile.mkdtemp()
    
    model_config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_size": 1024
    }
    
    kvcache_config = {
        "attention_type": "mha",
        "dtype": "fp16",
        "requests": [
            {"request_id": 0, "max_tokens": 512},
            {"request_id": 1, "max_tokens": 1024},
            {"request_id": 2, "max_tokens": 2048, "prefix_token_ids": [1, 2, 3, 4, 5]}
        ]
    }
    
    engine_config_path = create_engine_config(model_config, kvcache_config, temp_dir)
    kv_manager = edge_fm.KVManager(engine_config_path)
    
    status = kv_manager.get_status()
    assert len(status.slots) == 3, "Should have 3 slots"
    
    # 验证每个请求都有独立的 KV cache
    for request_id in [0, 1, 2]:
        assert kv_manager.is_request_valid(request_id), f"Request {request_id} should be valid"
        read_ptrs = kv_manager.get_read_kvcache(request_id)
        write_ptrs = kv_manager.get_write_kvcache(request_id)
        assert len(read_ptrs) == 2, f"Request {request_id} should have 2 layers"
        assert len(write_ptrs) == 2, f"Request {request_id} should have 2 layers"
        
        # 验证不同请求的指针不同（至少应该不同）
        if request_id > 0:
            prev_read_ptrs = kv_manager.get_read_kvcache(request_id - 1)
            # 不同请求的指针应该不同（至少第一层应该不同）
            assert read_ptrs[0] != prev_read_ptrs[0], \
                f"Request {request_id} and {request_id - 1} should have different cache pointers"


def test_different_dtypes():
    """测试不同 dtype 的 KV cache 分配"""
    temp_dir = tempfile.mkdtemp()
    
    model_config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_size": 1024
    }
    
    for dtype in ["fp16", "bf16", "fp32"]:
        kvcache_config = {
            "attention_type": "mha",
            "dtype": dtype,
            "requests": [
                {"request_id": 0, "max_tokens": 512}
            ]
        }
        
        engine_config_path = create_engine_config(model_config, kvcache_config, temp_dir)
        kv_manager = edge_fm.KVManager(engine_config_path)
        
        status = kv_manager.get_status()
        assert len(status.slots) == 1, f"Should have 1 slot for dtype {dtype}"
        
        read_ptrs = kv_manager.get_read_kvcache(0)
        write_ptrs = kv_manager.get_write_kvcache(0)
        assert len(read_ptrs) == 2, f"Should have 2 layers for dtype {dtype}"
        assert len(write_ptrs) == 2, f"Should have 2 layers for dtype {dtype}"


if __name__ == "__main__":
    # 支持直接运行：python test_kvcache.py
    pytest.main([__file__, "-v"])
