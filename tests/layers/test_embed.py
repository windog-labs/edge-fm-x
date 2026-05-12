"""
EmbedHeadLayer 层的正确性和性能测试（pytest 单元测试）

测试 edge_fm.EmbedHeadLayer 的实现，使用 PyTorch 作为参考实现。
包括：
1. forward_for_tokens 的正确性测试
2. forward_for_embeddings 的正确性测试
3. forward_for_tokens 的性能测试（确保性能不低于 PyTorch 的 0.9 倍）
4. forward_for_embeddings 的性能测试（确保性能不低于 PyTorch 的 0.9 倍）
"""

import json
import torch
import pytest
import tempfile
import os
from safetensors.torch import save_file

from tests.layers._test_utils import make_layer_engine_config

import edge_fm
from flashinfer.testing.utils import bench_gpu_time


def create_embed_layer(vocab_size, hidden_size, dtype_str="float16"):
    """创建 EmbedHeadLayer 实例"""
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "config.json")
    
    model_config = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size
    }
    with open(config_path, "w") as f:
        json.dump(model_config, f)
    
    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    torch.manual_seed(42)
    embedding_weights = torch.randn(vocab_size, hidden_size, dtype=dtype)
    
    safetensors_path = os.path.join(temp_dir, "model.safetensors")
    save_file({"model.embed_tokens.weight": embedding_weights}, safetensors_path)
    
    engine_config_dir = tempfile.mkdtemp()
    engine_config_path = os.path.join(engine_config_dir, "engine_config.json")
    engine_config = make_layer_engine_config(temp_dir, with_operator_table=False)
    with open(engine_config_path, "w") as f:
        json.dump(engine_config, f)
    
    return edge_fm.EmbedHeadLayer(engine_config_path)


def tensor_to_edge_fm_tensor(torch_tensor):
    """将 PyTorch tensor 转换为 edge_fm.Tensor（通过 DLPack）
    需确保 tensor 是 contiguous 且 row-major。对已 contiguous 的 tensor，
    .contiguous() 是空操作；对 view/transpose 等则会产生连续拷贝。
    """
    capsule = torch_tensor.contiguous().__dlpack__()
    return edge_fm.Tensor.from_dlpack(capsule)


def create_embedding_weights(vocab_size, hidden_size, dtype):
    """创建 embedding 权重（用于测试）"""
    torch.manual_seed(42)
    weights = torch.randn(vocab_size, hidden_size, device="cuda:0", dtype=dtype)
    return weights


class TestEmbedHead:
    """EmbedHeadLayer 测试类"""
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    @pytest.mark.parametrize("vocab_size", [1000, 5000])  # 减小 vocab_size 以加快测试
    @pytest.mark.parametrize("hidden_size", [768, 1024])
    @pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
    def test_forward_for_tokens_correctness(self, batch_size, seq_len, vocab_size, hidden_size, dtype_str):
        """测试 forward_for_tokens 的正确性
        
        参数:
            batch_size: 批次大小
            seq_len: 序列长度
            vocab_size: 词汇表大小
            hidden_size: 隐藏层大小
            dtype_str: 数据类型字符串（"float16" 或 "bfloat16"）
        """
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        layer = create_embed_layer(vocab_size, hidden_size, dtype_str)
        
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda:0", dtype=torch.int32)
        
        loader = edge_fm.WeightLoader.instance()
        weights = loader.get(edge_fm.ModelStage.Prefill)
        embedding_table = weights["model.embed_tokens.weight"]
        
        # 确保 output tensor 的 dtype 与 embedding table 的 dtype 匹配
        embedding_table_torch = torch.from_dlpack(embedding_table.to_dlpack())
        embedding_dtype = embedding_table_torch.dtype
        
        token_ids_cpu = token_ids.cpu()
        expected_output = torch.nn.functional.embedding(token_ids_cpu, embedding_table_torch.cpu()).to("cuda:0")
        
        # 使用 embedding table 的 dtype，而不是根据 dtype_str 创建
        output = torch.zeros(batch_size, seq_len, hidden_size, device="cuda:0", dtype=embedding_dtype)
        
        # 转换为 edge_fm Tensor
        token_ids_efm = tensor_to_edge_fm_tensor(token_ids)
        output_efm = tensor_to_edge_fm_tensor(output)
        
        layer.forward_for_tokens(token_ids_efm, output_efm)
        output_torch = torch.from_dlpack(output_efm.to_dlpack())
        torch.cuda.synchronize()
        
        torch.testing.assert_close(
            output_torch,
            expected_output,
            rtol=1e-3,
            atol=1e-3,
            msg=f"forward_for_tokens 结果与 PyTorch 不一致 "
                f"(batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}, "
                f"hidden_size={hidden_size}, dtype={dtype_str})"
        )
    
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [32, 128])
    @pytest.mark.parametrize("vocab_size", [1000, 5000])  # 减小 vocab_size 以加快测试
    @pytest.mark.parametrize("hidden_size", [768])
    @pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
    def test_forward_for_embeddings_correctness(self, batch_size, seq_len, vocab_size, hidden_size, dtype_str):
        """测试 forward_for_embeddings 的正确性
        
        参数:
            batch_size: 批次大小
            seq_len: 序列长度
            vocab_size: 词汇表大小
            hidden_size: 隐藏层大小
            dtype_str: 数据类型字符串（"float16" 或 "bfloat16"）
        """
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        layer = create_embed_layer(vocab_size, hidden_size, dtype_str)
        
        loader = edge_fm.WeightLoader.instance()
        weights = loader.get(edge_fm.ModelStage.Prefill)
        embedding_table = weights["model.embed_tokens.weight"]
        
        # 确保所有 tensor 的 dtype 与 embedding table 的 dtype 匹配
        embedding_table_torch = torch.from_dlpack(embedding_table.to_dlpack())
        embedding_dtype = embedding_table_torch.dtype
        
        num_custom_embeddings = 4
        custom_embeddings = torch.randn(num_custom_embeddings, hidden_size, device="cuda:0", dtype=embedding_dtype)
        
        # 使用 embed_token_id 方式：选择 embed_token_id（应该 >= vocab_size 以避免冲突）
        embed_token_id = vocab_size  # 使用 vocab_size 作为起始 token ID
        
        # 生成 token_ids，其中一些位置使用 embed_token_id 范围的值
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda:0", dtype=torch.int32)
        
        # 随机选择一些位置替换为 embed_token_id 范围的值
        num_replacements = min(num_custom_embeddings, batch_size * seq_len)
        replacement_positions = torch.randperm(batch_size * seq_len, device="cuda:0")[:num_replacements]
        
        for i, pos in enumerate(replacement_positions):
            batch_idx = pos // seq_len
            seq_idx = pos % seq_len
            token_ids[batch_idx, seq_idx] = embed_token_id + i
        
        # 构建期望输出：
        # 1. 对于普通 token (token_id < vocab_size)，使用标准 embedding
        # 2. 对于自定义 embedding token (token_id >= embed_token_id)，使用 custom_embeddings
        token_ids_cpu = token_ids.cpu()
        expected_output = torch.zeros(batch_size, seq_len, hidden_size, device="cuda:0", dtype=embedding_dtype)
        
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = token_ids[b, s].item()
                if token_id < vocab_size:
                    # 使用标准 embedding
                    expected_output[b, s] = embedding_table_torch[token_id].to("cuda:0")
                elif token_id >= embed_token_id and token_id < embed_token_id + num_custom_embeddings:
                    # 使用自定义 embedding
                    custom_idx = token_id - embed_token_id
                    expected_output[b, s] = custom_embeddings[custom_idx]
                else:
                    # 超出范围，使用零向量
                    expected_output[b, s] = torch.zeros(hidden_size, device="cuda:0", dtype=embedding_dtype)
        
        output = torch.zeros(batch_size, seq_len, hidden_size, device="cuda:0", dtype=embedding_dtype)
        
        token_ids_efm = tensor_to_edge_fm_tensor(token_ids)
        custom_embeddings_efm = tensor_to_edge_fm_tensor(custom_embeddings)
        output_efm = tensor_to_edge_fm_tensor(output)
        
        # 使用新的 API：embed_token_id 而不是 embedding_indices
        layer.forward_for_embeddings(token_ids_efm, custom_embeddings_efm, output_efm, embed_token_id)
        output_torch = torch.from_dlpack(output_efm.to_dlpack())
        torch.cuda.synchronize()
        
        torch.testing.assert_close(
            output_torch,
            expected_output,
            rtol=1e-3,
            atol=1e-3,
            msg=f"forward_for_embeddings 结果与 PyTorch 不一致 "
                f"(batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}, "
                f"hidden_size={hidden_size}, dtype={dtype_str})"
        )
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    @pytest.mark.parametrize("vocab_size", [1000, 5000])
    @pytest.mark.parametrize("hidden_size", [768, 1024])
    @pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
    def test_forward_for_tokens_performance(self, batch_size, seq_len, vocab_size, hidden_size, dtype_str):
        """测试 forward_for_tokens 的性能
        
        确保 EdgeFM 的性能不低于 PyTorch 的 0.9 倍。
        
        参数:
            batch_size: 批次大小
            seq_len: 序列长度
            vocab_size: 词汇表大小
            hidden_size: 隐藏层大小
            dtype_str: 数据类型字符串（"float16" 或 "bfloat16"）
        """
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        layer = create_embed_layer(vocab_size, hidden_size, dtype_str)
        
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda:0", dtype=torch.int32)
        
        loader = edge_fm.WeightLoader.instance()
        weights = loader.get(edge_fm.ModelStage.Prefill)
        embedding_table = weights["model.embed_tokens.weight"]
        
        embedding_table_torch = torch.from_dlpack(embedding_table.to_dlpack())
        embedding_dtype = embedding_table_torch.dtype
        
        # 测试 PyTorch 性能
        def run_pytorch():
            token_ids_cpu = token_ids.cpu()
            output = torch.nn.functional.embedding(token_ids_cpu, embedding_table_torch.cpu()).to("cuda:0")
            torch.cuda.synchronize()
        
        pytorch_measurements = bench_gpu_time(run_pytorch, repeat_time_ms=100)
        pytorch_avg = sum(pytorch_measurements) / len(pytorch_measurements)
        
        # 测试 edge_fm 性能
        output = torch.zeros(batch_size, seq_len, hidden_size, device="cuda:0", dtype=embedding_dtype)
        token_ids_efm = tensor_to_edge_fm_tensor(token_ids)
        output_efm = tensor_to_edge_fm_tensor(output)
        
        def run_edge_fm():
            layer.forward_for_tokens(token_ids_efm, output_efm)
            torch.cuda.synchronize()
        
        edge_fm_measurements = bench_gpu_time(run_edge_fm, repeat_time_ms=100)
        edge_fm_avg = sum(edge_fm_measurements) / len(edge_fm_measurements)
        
        # 计算性能比：PyTorch 时间 / EdgeFM 时间
        performance_ratio = pytorch_avg / edge_fm_avg if edge_fm_avg > 0 else 0.0
        
        # 断言：EdgeFM 性能不低于 PyTorch 的 0.9 倍
        assert performance_ratio >= 0.9, (
            f"性能测试失败：EdgeFM 性能低于 PyTorch 的 0.9 倍\n"
            f"  batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}, "
            f"  hidden_size={hidden_size}, dtype={dtype_str}\n"
            f"  PyTorch 平均时间: {pytorch_avg:.4f} ms\n"
            f"  EdgeFM 平均时间: {edge_fm_avg:.4f} ms\n"
            f"  性能比 (PyTorch/EdgeFM): {performance_ratio:.4f}\n"
            f"  要求: performance_ratio >= 0.9"
        )
    
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [32, 128])
    @pytest.mark.parametrize("vocab_size", [1000, 5000])
    @pytest.mark.parametrize("hidden_size", [768])
    @pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
    def test_forward_for_embeddings_performance(self, batch_size, seq_len, vocab_size, hidden_size, dtype_str):
        """测试 forward_for_embeddings 的性能
        
        确保 EdgeFM 的性能不低于 PyTorch 的 0.9 倍。
        
        参数:
            batch_size: 批次大小
            seq_len: 序列长度
            vocab_size: 词汇表大小
            hidden_size: 隐藏层大小
            dtype_str: 数据类型字符串（"float16" 或 "bfloat16"）
        """
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        layer = create_embed_layer(vocab_size, hidden_size, dtype_str)
        
        loader = edge_fm.WeightLoader.instance()
        weights = loader.get(edge_fm.ModelStage.Prefill)
        embedding_table = weights["model.embed_tokens.weight"]
        
        embedding_table_torch = torch.from_dlpack(embedding_table.to_dlpack())
        embedding_dtype = embedding_table_torch.dtype
        
        num_custom_embeddings = 4
        custom_embeddings = torch.randn(num_custom_embeddings, hidden_size, device="cuda:0", dtype=embedding_dtype)
        
        # 使用 embed_token_id 方式
        embed_token_id = vocab_size
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda:0", dtype=torch.int32)
        num_replacements = min(num_custom_embeddings, batch_size * seq_len)
        replacement_positions = torch.randperm(batch_size * seq_len, device="cuda:0")[:num_replacements]
        for i, pos in enumerate(replacement_positions):
            batch_idx = pos // seq_len
            seq_idx = pos % seq_len
            token_ids[batch_idx, seq_idx] = embed_token_id + i
        
        # 测试 PyTorch 性能（模拟自定义 embeddings 插入）
        def run_pytorch():
            token_ids_cpu = token_ids.cpu()
            output = torch.zeros(batch_size, seq_len, hidden_size, device="cuda:0", dtype=embedding_dtype)
            for b in range(batch_size):
                for s in range(seq_len):
                    token_id = token_ids_cpu[b, s].item()
                    if token_id < vocab_size:
                        output[b, s] = embedding_table_torch[token_id].to("cuda:0")
                    elif token_id >= embed_token_id and token_id < embed_token_id + num_custom_embeddings:
                        custom_idx = token_id - embed_token_id
                        output[b, s] = custom_embeddings[custom_idx]
            torch.cuda.synchronize()
        
        pytorch_measurements = bench_gpu_time(run_pytorch, repeat_time_ms=100)
        pytorch_avg = sum(pytorch_measurements) / len(pytorch_measurements)
        
        # 测试 edge_fm 性能
        output = torch.zeros(batch_size, seq_len, hidden_size, device="cuda:0", dtype=embedding_dtype)
        token_ids_efm = tensor_to_edge_fm_tensor(token_ids)
        custom_embeddings_efm = tensor_to_edge_fm_tensor(custom_embeddings)
        output_efm = tensor_to_edge_fm_tensor(output)
        
        def run_edge_fm():
            layer.forward_for_embeddings(token_ids_efm, custom_embeddings_efm, output_efm, embed_token_id)
            torch.cuda.synchronize()
        
        edge_fm_measurements = bench_gpu_time(run_edge_fm, repeat_time_ms=100)
        edge_fm_avg = sum(edge_fm_measurements) / len(edge_fm_measurements)
        
        # 计算性能比：PyTorch 时间 / EdgeFM 时间
        performance_ratio = pytorch_avg / edge_fm_avg if edge_fm_avg > 0 else 0.0
        
        # 断言：EdgeFM 性能不低于 PyTorch 的 0.9 倍
        assert performance_ratio >= 0.9, (
            f"性能测试失败：EdgeFM 性能低于 PyTorch 的 0.9 倍\n"
            f"  batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}, "
            f"  hidden_size={hidden_size}, dtype={dtype_str}\n"
            f"  PyTorch 平均时间: {pytorch_avg:.4f} ms\n"
            f"  EdgeFM 平均时间: {edge_fm_avg:.4f} ms\n"
            f"  性能比 (PyTorch/EdgeFM): {performance_ratio:.4f}\n"
            f"  要求: performance_ratio >= 0.9"
        )


if __name__ == "__main__":
    # 支持直接运行：python test_embed.py
    pytest.main([__file__, "-v"])
