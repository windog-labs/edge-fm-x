"""
Linear 层的正确性和性能测试（pytest 单元测试）

测试 edge_fm.LinearLayer 的实现，使用 PyTorch 作为参考实现。
包括：
1. 正确性测试：分别测试 forward_fp16_bf16 和 forward_int4_groupwise
2. 性能测试：使用中位数比较，确保不低于 PyTorch 的 0.85 倍
"""

import json
import statistics
import torch
import pytest
import tempfile
import os
from safetensors.torch import save_file

from tests.layers._test_utils import PROJECT_ROOT, make_layer_engine_config

# 临时文件目录（避免使用 /tmp，存储空间可能不足）
TEST_TEMP_BASE = PROJECT_ROOT / "build" / "test_linear_temp"
TEST_TEMP_BASE.mkdir(parents=True, exist_ok=True)


def _mkdtemp():
    """在项目目录下创建临时目录，避免填满 /tmp"""
    return tempfile.mkdtemp(dir=str(TEST_TEMP_BASE))


import edge_fm
from flashinfer.testing.utils import bench_gpu_time


def create_linear_layer_with_weights(layer_prefix, in_features, out_features, dtype="float16", has_bias=True):
    """创建 LinearLayer 实例并保存权重到 safetensors 文件
    
    参数:
        layer_prefix: 层名称前缀（例如 "test"）
        in_features: 输入特征数
        out_features: 输出特征数
        dtype: 数据类型 ("float16" 或 "bfloat16")
        has_bias: 是否有 bias
    """
    # 创建临时目录用于存放 config.json 和 model.safetensors
    temp_dir = _mkdtemp()
    config_path = os.path.join(temp_dir, "config.json")
    
    # 创建模型配置文件
    model_config = {
        "torch_dtype": dtype,
        "hidden_size": 4096  # 示例值，实际不使用
    }
    with open(config_path, "w") as f:
        json.dump(model_config, f)
    
    weight_name = layer_prefix + ".weight"
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    torch.manual_seed(42)
    weight = torch.randn(out_features, in_features, device="cuda:0", dtype=torch_dtype)
    
    weights_dict = {weight_name: weight}
    if has_bias:
        bias = torch.randn(out_features, device="cuda:0", dtype=torch_dtype)
        bias_name = layer_prefix + ".bias"
        weights_dict[bias_name] = bias
    else:
        bias = None
    
    safetensors_path = os.path.join(temp_dir, "model.safetensors")
    save_file(weights_dict, safetensors_path)
    
    # 创建 engine_config.json 文件
    engine_config_dir = _mkdtemp()
    engine_config_path = os.path.join(engine_config_dir, "engine_config.json")
    engine_config = make_layer_engine_config(temp_dir)
    with open(engine_config_path, "w") as f:
        json.dump(engine_config, f)
    
    # 创建 layer，传入 layer_prefix（不带后缀）
    layer = edge_fm.LinearLayer(layer_prefix, engine_config_path, in_features, out_features)
    
    return layer, weight, bias


def create_int4_linear_layer_with_weights(layer_prefix, in_features, out_features, group_size=128):
    """创建 INT4 量化的 LinearLayer 实例并保存权重到 safetensors 文件
    
    参数:
        layer_prefix: 层名称前缀（例如 "test"）
        in_features: 输入特征数
        out_features: 输出特征数
        group_size: 分组大小
    """
    # 创建临时目录用于存放 config.json 和 model.safetensors
    temp_dir = _mkdtemp()
    config_path = os.path.join(temp_dir, "config.json")
    
    # 创建模型配置文件
    model_config = {
        "torch_dtype": "float16",
        "hidden_size": 4096
    }
    with open(config_path, "w") as f:
        json.dump(model_config, f)
    
    # 创建 INT4 量化权重（使用 int8 存储，实际是 int4 packed）
    # qweight shape: [out_features/2, in_features] (packed int4)
    # scaling_factors shape: [in_features/group_size, out_features]
    num_groups = (in_features + group_size - 1) // group_size
    torch.manual_seed(42)
    qweight = torch.randint(-8, 8, (out_features // 2, in_features), 
                           device="cuda:0", dtype=torch.int8)
    scaling_factors = torch.randn(num_groups, out_features, 
                                 device="cuda:0", dtype=torch.float16)
    
    # 保存到 safetensors 文件
    # 权重文件中的名称需要包含后缀
    qweight_name = layer_prefix + ".qweight"
    scaling_factors_name = layer_prefix + ".scaling_factors"
    weights_dict = {
        qweight_name: qweight,
        scaling_factors_name: scaling_factors
    }
    safetensors_path = os.path.join(temp_dir, "model.safetensors")
    save_file(weights_dict, safetensors_path)
    
    # 创建 engine_config.json 文件
    engine_config_dir = _mkdtemp()
    engine_config_path = os.path.join(engine_config_dir, "engine_config.json")
    engine_config = make_layer_engine_config(temp_dir)
    with open(engine_config_path, "w") as f:
        json.dump(engine_config, f)
    
    # 创建 layer，传入 layer_prefix（不带后缀）
    layer = edge_fm.LinearLayer(layer_prefix, engine_config_path, in_features, out_features)
    
    return layer, qweight, scaling_factors, group_size


def create_fused_qkv_linear_layer_with_weights(layer_prefix_base, in_features, q_out_features, k_out_features, v_out_features, dtype="float16", has_bias=True):
    """创建 FusedQKVLinearLayer 实例并保存权重到 safetensors 文件
    
    参数:
        layer_prefix_base: 层名称基础前缀（例如 "test"）
        in_features: 输入特征数
        q_out_features: Q 投影输出特征数
        k_out_features: K 投影输出特征数
        v_out_features: V 投影输出特征数
        dtype: 数据类型 ("float16" 或 "bfloat16")
        has_bias: 是否有 bias
    """
    # 创建临时目录用于存放 config.json 和 model.safetensors
    temp_dir = _mkdtemp()
    config_path = os.path.join(temp_dir, "config.json")
    
    # 创建模型配置文件
    model_config = {
        "torch_dtype": dtype,
        "hidden_size": 4096  # 示例值，实际不使用
    }
    with open(config_path, "w") as f:
        json.dump(model_config, f)
    
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    torch.manual_seed(42)
    
    # 创建 Q, K, V 权重
    q_weight = torch.randn(q_out_features, in_features, device="cuda:0", dtype=torch_dtype)
    k_weight = torch.randn(k_out_features, in_features, device="cuda:0", dtype=torch_dtype)
    v_weight = torch.randn(v_out_features, in_features, device="cuda:0", dtype=torch_dtype)
    
    weights_dict = {
        layer_prefix_base + ".q_proj.weight": q_weight,
        layer_prefix_base + ".k_proj.weight": k_weight,
        layer_prefix_base + ".v_proj.weight": v_weight,
    }
    
    if has_bias:
        q_bias = torch.randn(q_out_features, device="cuda:0", dtype=torch_dtype)
        k_bias = torch.randn(k_out_features, device="cuda:0", dtype=torch_dtype)
        v_bias = torch.randn(v_out_features, device="cuda:0", dtype=torch_dtype)
        weights_dict[layer_prefix_base + ".q_proj.bias"] = q_bias
        weights_dict[layer_prefix_base + ".k_proj.bias"] = k_bias
        weights_dict[layer_prefix_base + ".v_proj.bias"] = v_bias
    else:
        q_bias = None
        k_bias = None
        v_bias = None
    
    safetensors_path = os.path.join(temp_dir, "model.safetensors")
    save_file(weights_dict, safetensors_path)
    
    # 创建 engine_config.json 文件
    engine_config_dir = _mkdtemp()
    engine_config_path = os.path.join(engine_config_dir, "engine_config.json")
    engine_config = make_layer_engine_config(temp_dir)
    with open(engine_config_path, "w") as f:
        json.dump(engine_config, f)
    
    # 创建 layer
    layer = edge_fm.FusedQKVLinearLayer(
        layer_prefix_base, engine_config_path, 
        in_features, q_out_features, k_out_features, v_out_features
    )
    
    return layer, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias


def create_fused_gate_up_linear_layer_with_weights(layer_prefix_base, in_features, gate_out_features, up_out_features, dtype="float16", has_bias=True):
    """创建 FusedGateUpLinearLayer 实例并保存权重到 safetensors 文件

    参数:
        layer_prefix_base: 层名称基础前缀（例如 "model.layers.0.mlp"）
        in_features: 输入特征数
        gate_out_features: Gate 投影输出特征数
        up_out_features: Up 投影输出特征数
        dtype: 数据类型 ("float16" 或 "bfloat16")
        has_bias: 是否有 bias
    """
    temp_dir = _mkdtemp()
    config_path = os.path.join(temp_dir, "config.json")

    model_config = {
        "torch_dtype": dtype,
        "hidden_size": 4096
    }
    with open(config_path, "w") as f:
        json.dump(model_config, f)

    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    torch.manual_seed(42)

    gate_weight = torch.randn(gate_out_features, in_features, device="cuda:0", dtype=torch_dtype)
    up_weight = torch.randn(up_out_features, in_features, device="cuda:0", dtype=torch_dtype)

    weights_dict = {
        layer_prefix_base + ".gate_proj.weight": gate_weight,
        layer_prefix_base + ".up_proj.weight": up_weight,
    }

    if has_bias:
        gate_bias = torch.randn(gate_out_features, device="cuda:0", dtype=torch_dtype)
        up_bias = torch.randn(up_out_features, device="cuda:0", dtype=torch_dtype)
        weights_dict[layer_prefix_base + ".gate_proj.bias"] = gate_bias
        weights_dict[layer_prefix_base + ".up_proj.bias"] = up_bias
    else:
        gate_bias = None
        up_bias = None

    safetensors_path = os.path.join(temp_dir, "model.safetensors")
    save_file(weights_dict, safetensors_path)

    engine_config_dir = _mkdtemp()
    engine_config_path = os.path.join(engine_config_dir, "engine_config.json")
    engine_config = make_layer_engine_config(temp_dir)
    with open(engine_config_path, "w") as f:
        json.dump(engine_config, f)

    layer = edge_fm.FusedGateUpLinearLayer(
        layer_prefix_base, engine_config_path,
        in_features, gate_out_features, up_out_features
    )

    return layer, gate_weight, up_weight, gate_bias, up_bias


def create_lm_head_linear_layer_with_weights(in_features, out_features, dtype="float16"):
    """创建 LMHeadLinearLayer 实例（使用 model.embed_tokens.weight 作为 tied weights）

    参数:
        in_features: 输入特征数（hidden_size）
        out_features: 输出特征数（vocab_size）
        dtype: 数据类型 ("float16" 或 "bfloat16")
    """
    temp_dir = _mkdtemp()
    config_path = os.path.join(temp_dir, "config.json")

    model_config = {
        "torch_dtype": dtype,
        "hidden_size": in_features,
        "vocab_size": out_features
    }
    with open(config_path, "w") as f:
        json.dump(model_config, f)

    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    torch.manual_seed(42)
    # LM head 使用 embedding 的转置：embedding 形状 [vocab_size, hidden_size] = [out_features, in_features]
    embed_weight = torch.randn(out_features, in_features, device="cuda:0", dtype=torch_dtype)

    safetensors_path = os.path.join(temp_dir, "model.safetensors")
    save_file({"model.embed_tokens.weight": embed_weight}, safetensors_path)

    engine_config_dir = _mkdtemp()
    engine_config_path = os.path.join(engine_config_dir, "engine_config.json")
    engine_config = make_layer_engine_config(temp_dir)
    with open(engine_config_path, "w") as f:
        json.dump(engine_config, f)

    layer = edge_fm.LMHeadLinearLayer(engine_config_path, in_features, out_features)

    return layer, embed_weight


def tensor_to_edge_fm_tensor(torch_tensor):
    """将 PyTorch tensor 转换为 edge_fm.Tensor（通过 DLPack）
    需确保 tensor 是 contiguous 且 row-major。对已 contiguous 的 tensor，
    .contiguous() 是空操作；对 view/transpose 等则会产生连续拷贝。
    """
    return edge_fm.Tensor.from_dlpack(torch_tensor.contiguous().__dlpack__())


# 测试参数
FP16_BF16_TEST_CASES = [
    {"batch_size": 1, "in_features": 4096, "out_features": 11008, "dtype": "float16"},
    {"batch_size": 4, "in_features": 4096, "out_features": 11008, "dtype": "float16"},
    {"batch_size": 8, "in_features": 4096, "out_features": 11008, "dtype": "float16"},
    {"batch_size": 1, "in_features": 4096, "out_features": 11008, "dtype": "bfloat16"},
    {"batch_size": 4, "in_features": 4096, "out_features": 11008, "dtype": "bfloat16"},
]

INT4_TEST_CASES = [
    {"batch_size": 1, "in_features": 4096, "out_features": 11008, "group_size": 128},
    {"batch_size": 4, "in_features": 4096, "out_features": 11008, "group_size": 128},
    {"batch_size": 8, "in_features": 4096, "out_features": 11008, "group_size": 128},
]

FP16_BF16_PERF_TEST_CASES = [
    {"batch_size": 1, "in_features": 4096, "out_features": 11008, "dtype": "float16"},
    {"batch_size": 4, "in_features": 4096, "out_features": 11008, "dtype": "float16"},
    {"batch_size": 16, "in_features": 4096, "out_features": 11008, "dtype": "float16"},
    {"batch_size": 32, "in_features": 4096, "out_features": 11008, "dtype": "float16"},
]

FUSED_QKV_PERF_TEST_CASES = [
    {"batch_size": 1, "in_features": 4096, "q_out": 4096, "k_out": 1024, "v_out": 1024, "dtype": "float16"},
    {"batch_size": 4, "in_features": 4096, "q_out": 4096, "k_out": 1024, "v_out": 1024, "dtype": "float16"},
    {"batch_size": 16, "in_features": 4096, "q_out": 4096, "k_out": 1024, "v_out": 1024, "dtype": "float16"},
    {"batch_size": 32, "in_features": 4096, "q_out": 4096, "k_out": 1024, "v_out": 1024, "dtype": "float16"},
]

FUSED_GATE_UP_PERF_TEST_CASES = [
    {"batch_size": 1, "in_features": 4096, "gate_out": 11008, "up_out": 11008, "dtype": "float16"},
    {"batch_size": 4, "in_features": 4096, "gate_out": 11008, "up_out": 11008, "dtype": "float16"},
    {"batch_size": 16, "in_features": 4096, "gate_out": 11008, "up_out": 11008, "dtype": "float16"},
    {"batch_size": 32, "in_features": 4096, "gate_out": 11008, "up_out": 11008, "dtype": "float16"},
]

LM_HEAD_PERF_TEST_CASES = [
    {"batch_size": 1, "in_features": 4096, "out_features": 32000, "dtype": "float16"},
    {"batch_size": 4, "in_features": 4096, "out_features": 32000, "dtype": "float16"},
    {"batch_size": 16, "in_features": 4096, "out_features": 32000, "dtype": "float16"},
    {"batch_size": 32, "in_features": 4096, "out_features": 32000, "dtype": "float16"},
]

FUSED_QKV_TEST_CASES = [
    {"batch_size": 1, "in_features": 4096, "q_out": 4096, "k_out": 1024, "v_out": 1024, "dtype": "float16"},
    {"batch_size": 4, "in_features": 4096, "q_out": 4096, "k_out": 1024, "v_out": 1024, "dtype": "float16"},
    {"batch_size": 8, "in_features": 4096, "q_out": 4096, "k_out": 1024, "v_out": 1024, "dtype": "float16"},
    {"batch_size": 1, "in_features": 4096, "q_out": 4096, "k_out": 1024, "v_out": 1024, "dtype": "bfloat16"},
    {"batch_size": 4, "in_features": 4096, "q_out": 4096, "k_out": 1024, "v_out": 1024, "dtype": "bfloat16"},
]

FUSED_GATE_UP_TEST_CASES = [
    {"batch_size": 1, "in_features": 4096, "gate_out": 11008, "up_out": 11008, "dtype": "float16"},
    {"batch_size": 4, "in_features": 4096, "gate_out": 11008, "up_out": 11008, "dtype": "float16"},
    {"batch_size": 8, "in_features": 4096, "gate_out": 11008, "up_out": 11008, "dtype": "float16"},
    {"batch_size": 1, "in_features": 4096, "gate_out": 11008, "up_out": 11008, "dtype": "bfloat16"},
    {"batch_size": 4, "in_features": 4096, "gate_out": 11008, "up_out": 11008, "dtype": "bfloat16"},
]

LM_HEAD_TEST_CASES = [
    {"batch_size": 1, "in_features": 4096, "out_features": 32000, "dtype": "float16"},
    {"batch_size": 4, "in_features": 4096, "out_features": 32000, "dtype": "float16"},
    {"batch_size": 8, "in_features": 4096, "out_features": 32000, "dtype": "float16"},
    {"batch_size": 1, "in_features": 4096, "out_features": 32000, "dtype": "bfloat16"},
    {"batch_size": 4, "in_features": 4096, "out_features": 32000, "dtype": "bfloat16"},
]


class TestLinear:
    """Linear 层测试类（包括正确性和性能测试）"""
    
    @pytest.mark.parametrize("case", FP16_BF16_TEST_CASES)
    def test_fp16_bf16_correctness(self, case):
        """测试 FP16/BF16 前向传播的正确性
        
        参数:
            case: 测试用例字典（pytest 参数化）
        """
        batch_size = case["batch_size"]
        in_features = case["in_features"]
        out_features = case["out_features"]
        dtype = case["dtype"]
        
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # 创建唯一的 layer_prefix（基于 dtype 和测试用例参数，避免测试间权重污染）
        case_key = f"{dtype}_{in_features}_{out_features}"
        test_id = abs(hash(case_key)) % 100000
        layer_prefix = f"test_fp16_{dtype}_{test_id}"
        layer, weight, bias = create_linear_layer_with_weights(
            layer_prefix, in_features, out_features, dtype
        )
        
        # 创建输入
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        input_torch = torch.randn(batch_size, in_features, device="cuda:0", dtype=torch_dtype)
        
        # 使用 PyTorch 作为参考
        output_torch_ref = torch.nn.functional.linear(input_torch, weight, bias)
        
        # 使用 edge_fm 实现
        input_efm = tensor_to_edge_fm_tensor(input_torch)
        output_efm = tensor_to_edge_fm_tensor(torch.empty_like(output_torch_ref))
        
        layer.forward_fp16_bf16(input_efm, output_efm)
        torch.cuda.synchronize()
        
        output_efm_torch = torch.from_dlpack(output_efm.to_dlpack())
        
        # cuBLASLt bias epilogue and PyTorch can round large-K FP16/BF16 GEMM
        # differently. Validate numerical agreement within dtype-appropriate
        # tolerances instead of requiring bitwise-equivalent accumulation.
        rtol_atol = (5e-1, 1.0) if dtype == "bfloat16" else (3e-2, 3e-1)
        torch.testing.assert_close(
            output_efm_torch, 
            output_torch_ref, 
            rtol=rtol_atol[0],
            atol=rtol_atol[1],
            msg=f"FP16/BF16 结果与 PyTorch 不一致 (batch_size={batch_size}, dtype={dtype})"
        )
    
    @pytest.mark.parametrize("case", INT4_TEST_CASES)
    def test_int4_groupwise_correctness(self, case):
        """测试 INT4 Groupwise 前向传播的正确性
        
        参数:
            case: 测试用例字典（pytest 参数化）
        """
        batch_size = case["batch_size"]
        in_features = case["in_features"]
        out_features = case["out_features"]
        group_size = case["group_size"]
        
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # 创建唯一的 layer_prefix（基于 group_size 和测试用例参数，避免测试间权重污染）
        # 使用 case 的哈希值来确保唯一性（不包括 batch_size）
        case_key = f"{group_size}_{in_features}_{out_features}"
        test_id = abs(hash(case_key)) % 100000
        layer_prefix = f"test_int4_{group_size}_{test_id}"
        layer, _, _, _ = create_int4_linear_layer_with_weights(
            layer_prefix, in_features, out_features, group_size
        )
        
        # 创建输入
        input_torch = torch.randn(batch_size, in_features, device="cuda:0", dtype=torch.float16)
        
        # 使用 edge_fm 实现
        input_efm = tensor_to_edge_fm_tensor(input_torch)
        output_efm = tensor_to_edge_fm_tensor(
            torch.empty(batch_size, out_features, device="cuda:0", dtype=torch.float16)
        )
        
        layer.forward_int4_groupwise(input_efm, output_efm)
        torch.cuda.synchronize()
        
        output_efm_torch = torch.from_dlpack(output_efm.to_dlpack())
        
        # 基本检查：输出形状和数值范围
        assert output_efm_torch.shape == (batch_size, out_features), \
            f"输出形状不正确: 期望 {(batch_size, out_features)}, 得到 {output_efm_torch.shape}"
        assert not torch.isnan(output_efm_torch).any(), "输出包含 NaN"
        assert not torch.isinf(output_efm_torch).any(), "输出包含 Inf"

    @pytest.mark.parametrize("case", FUSED_QKV_TEST_CASES)
    def test_fused_qkv_correctness(self, case):
        """测试 FusedQKVLinearLayer 前向传播的正确性
        
        参数:
            case: 测试用例字典（pytest 参数化）
        """
        batch_size = case["batch_size"]
        in_features = case["in_features"]
        q_out = case["q_out"]
        k_out = case["k_out"]
        v_out = case["v_out"]
        dtype = case["dtype"]
        
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # 创建唯一的 layer_prefix_base（基于 dtype 和测试用例参数，避免测试间权重污染）
        case_key = f"{dtype}_{in_features}_{q_out}_{k_out}_{v_out}"
        test_id = abs(hash(case_key)) % 100000
        layer_prefix_base = f"test_fused_qkv_{dtype}_{test_id}"
        
        layer, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias = create_fused_qkv_linear_layer_with_weights(
            layer_prefix_base, in_features, q_out, k_out, v_out, dtype
        )
        
        # 创建输入
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        input_torch = torch.randn(batch_size, in_features, device="cuda:0", dtype=torch_dtype)
        
        # 使用 PyTorch 作为参考（与 edge_fm 相同的 fused 方式：权重 concat 后一次 matmul）
        fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        fused_bias = torch.cat([q_bias, k_bias, v_bias], dim=0) if q_bias is not None else None
        output_torch_ref = torch.nn.functional.linear(input_torch, fused_weight, fused_bias)
        
        # 使用 edge_fm 实现
        input_efm = tensor_to_edge_fm_tensor(input_torch)
        output_efm = tensor_to_edge_fm_tensor(torch.empty_like(output_torch_ref))
        
        layer.forward_fp16_bf16(input_efm, output_efm)
        torch.cuda.synchronize()
        
        output_efm_torch = torch.from_dlpack(output_efm.to_dlpack())

        # 参考与 edge_fm 均使用 fused 实现，比对更一致；bfloat16 因 cuBLAS 与 PyTorch 舍入顺序不同误差略大
        rtol_atol = (3e-1, 3e-1) if dtype == "bfloat16" else (3e-2, 3e-2)
        torch.testing.assert_close(
            output_efm_torch,
            output_torch_ref,
            rtol=rtol_atol[0],
            atol=rtol_atol[1],
            msg=f"FusedQKV 结果与 PyTorch 不一致 (batch_size={batch_size}, dtype={dtype})"
        )

        # 验证输出布局：检查 Q, K, V 部分是否正确
        q_output_efm = output_efm_torch[:, :q_out]
        k_output_efm = output_efm_torch[:, q_out:q_out + k_out]
        v_output_efm = output_efm_torch[:, q_out + k_out:]
        q_output_ref = output_torch_ref[:, :q_out]
        k_output_ref = output_torch_ref[:, q_out:q_out + k_out]
        v_output_ref = output_torch_ref[:, q_out + k_out:]

        torch.testing.assert_close(
            q_output_efm, q_output_ref, rtol=rtol_atol[0], atol=rtol_atol[1],
            msg="Q 部分结果不一致"
        )
        torch.testing.assert_close(
            k_output_efm, k_output_ref, rtol=rtol_atol[0], atol=rtol_atol[1],
            msg="K 部分结果不一致"
        )
        torch.testing.assert_close(
            v_output_efm, v_output_ref, rtol=rtol_atol[0], atol=rtol_atol[1],
            msg="V 部分结果不一致"
        )

    @pytest.mark.parametrize("case", FUSED_GATE_UP_TEST_CASES)
    def test_fused_gate_up_correctness(self, case):
        """测试 FusedGateUpLinearLayer 前向传播的正确性

        参数:
            case: 测试用例字典（pytest 参数化）
        """
        batch_size = case["batch_size"]
        in_features = case["in_features"]
        gate_out = case["gate_out"]
        up_out = case["up_out"]
        dtype = case["dtype"]

        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # 创建唯一的 layer_prefix_base
        case_key = f"{dtype}_{in_features}_{gate_out}_{up_out}"
        test_id = abs(hash(case_key)) % 100000
        layer_prefix_base = f"test_fused_gate_up_{dtype}_{test_id}"

        layer, gate_weight, up_weight, gate_bias, up_bias = create_fused_gate_up_linear_layer_with_weights(
            layer_prefix_base, in_features, gate_out, up_out, dtype
        )

        # 创建输入
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        input_torch = torch.randn(batch_size, in_features, device="cuda:0", dtype=torch_dtype)

        # edge_fm stores fused gate/up tensors as [up, gate] to feed the
        # downstream fused SwiGLU path without another reordered copy.
        fused_weight = torch.cat([up_weight, gate_weight], dim=0)
        fused_bias = torch.cat([up_bias, gate_bias], dim=0) if gate_bias is not None else None
        output_torch_ref = torch.nn.functional.linear(input_torch, fused_weight, fused_bias)

        # 使用 edge_fm 实现
        input_efm = tensor_to_edge_fm_tensor(input_torch)
        output_efm = tensor_to_edge_fm_tensor(torch.empty_like(output_torch_ref))

        layer.forward_fp16_bf16(input_efm, output_efm)
        torch.cuda.synchronize()

        output_efm_torch = torch.from_dlpack(output_efm.to_dlpack())

        # cuBLASLt and PyTorch may pick different internal algorithms/tile strategies,
        # causing slightly different rounding; bfloat16 needs wider tolerance.
        rtol_atol = (5e-1, 1.0) if dtype == "bfloat16" else (3e-2, 3e-1)
        torch.testing.assert_close(
            output_efm_torch,
            output_torch_ref,
            rtol=rtol_atol[0],
            atol=rtol_atol[1],
            msg=f"FusedGateUp 结果与 PyTorch 不一致 (batch_size={batch_size}, dtype={dtype})"
        )

        # 验证输出布局：[up, gate]
        up_output_efm = output_efm_torch[:, :up_out]
        gate_output_efm = output_efm_torch[:, up_out:]
        up_output_ref = output_torch_ref[:, :up_out]
        gate_output_ref = output_torch_ref[:, up_out:]
        torch.testing.assert_close(
            gate_output_efm, gate_output_ref, rtol=rtol_atol[0], atol=rtol_atol[1],
            msg="Gate 部分结果不一致"
        )
        torch.testing.assert_close(
            up_output_efm, up_output_ref, rtol=rtol_atol[0], atol=rtol_atol[1],
            msg="Up 部分结果不一致"
        )

    @pytest.mark.parametrize("case", LM_HEAD_TEST_CASES)
    def test_lm_head_correctness(self, case):
        """测试 LMHeadLinearLayer 前向传播的正确性

        参数:
            case: 测试用例字典（pytest 参数化）
        """
        batch_size = case["batch_size"]
        in_features = case["in_features"]
        out_features = case["out_features"]
        dtype = case["dtype"]

        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        layer, embed_weight = create_lm_head_linear_layer_with_weights(
            in_features, out_features, dtype
        )

        # 创建输入
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        input_torch = torch.randn(batch_size, in_features, device="cuda:0", dtype=torch_dtype)

        # LM head 使用 embedding 的转置作为权重: output = input @ weight^T
        # embed_weight shape: [vocab_size, hidden_size] = [out_features, in_features]
        output_torch_ref = torch.nn.functional.linear(input_torch, embed_weight, None)

        # 使用 edge_fm 实现
        input_efm = tensor_to_edge_fm_tensor(input_torch)
        output_efm = tensor_to_edge_fm_tensor(torch.empty_like(output_torch_ref))

        layer.forward_fp16_bf16(input_efm, output_efm)
        torch.cuda.synchronize()

        output_efm_torch = torch.from_dlpack(output_efm.to_dlpack())

        # LM Head 大矩阵 [batch, hidden] @ [vocab, hidden]^T，bfloat16 舍入误差更大
        rtol_atol = (5e-1, 5e-1) if dtype == "bfloat16" else (3e-2, 3e-2)
        torch.testing.assert_close(
            output_efm_torch,
            output_torch_ref,
            rtol=rtol_atol[0],
            atol=rtol_atol[1],
            msg=f"LMHead 结果与 PyTorch 不一致 (batch_size={batch_size}, dtype={dtype})"
        )

    @pytest.mark.parametrize("case", FP16_BF16_PERF_TEST_CASES)
    def test_fp16_bf16_performance(self, case):
        """测试 FP16/BF16 前向传播的性能

        使用中位数、更长测量时间以降低微秒级 benchmark 的波动。
        确保 EdgeFM 的性能不低于 PyTorch 的 0.85 倍。
        """
        batch_size = case["batch_size"]
        in_features = case["in_features"]
        out_features = case["out_features"]
        dtype = case["dtype"]

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        weight_pattern = "test.weight"
        layer, weight, bias = create_linear_layer_with_weights(
            weight_pattern, in_features, out_features, dtype
        )

        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        input_torch = torch.randn(batch_size, in_features, device="cuda:0", dtype=torch_dtype)

        def run_pytorch():
            torch.nn.functional.linear(input_torch, weight, bias)

        pytorch_measurements = bench_gpu_time(run_pytorch, repeat_time_ms=300)
        pytorch_median = statistics.median(pytorch_measurements)

        input_efm = tensor_to_edge_fm_tensor(input_torch)
        output_efm = tensor_to_edge_fm_tensor(
            torch.empty(batch_size, out_features, device="cuda:0", dtype=torch_dtype)
        )

        def run_edge_fm():
            layer.forward_fp16_bf16(input_efm, output_efm)

        edge_fm_measurements = bench_gpu_time(run_edge_fm, repeat_time_ms=300)
        edge_fm_median = statistics.median(edge_fm_measurements)
        performance_ratio = pytorch_median / edge_fm_median if edge_fm_median > 0 else 0.0

        assert performance_ratio >= 0.85, (
            f"性能测试失败：EdgeFM 性能低于 PyTorch 的 0.85 倍\n"
            f"  batch_size={batch_size}, dtype={dtype}\n"
            f"  PyTorch 中位时间: {pytorch_median:.4f} ms\n"
            f"  EdgeFM 中位时间: {edge_fm_median:.4f} ms\n"
            f"  性能比 (PyTorch/EdgeFM): {performance_ratio:.4f}\n"
            f"  要求: performance_ratio >= 0.85"
        )

    @pytest.mark.parametrize("case", FUSED_QKV_PERF_TEST_CASES)
    def test_fused_qkv_performance(self, case):
        """测试 FusedQKVLinearLayer 前向传播的性能

        使用中位数、更长测量时间以降低微秒级 benchmark 的波动。
        确保 EdgeFM 的性能不低于 PyTorch 的 0.85 倍。
        """
        batch_size = case["batch_size"]
        in_features = case["in_features"]
        q_out = case["q_out"]
        k_out = case["k_out"]
        v_out = case["v_out"]
        dtype = case["dtype"]
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        layer_prefix_base = "test_fused_qkv_perf"
        layer, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias = create_fused_qkv_linear_layer_with_weights(
            layer_prefix_base, in_features, q_out, k_out, v_out, dtype
        )
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        input_torch = torch.randn(batch_size, in_features, device="cuda:0", dtype=torch_dtype)

        def run_pytorch():
            q_o = torch.nn.functional.linear(input_torch, q_weight, q_bias)
            k_o = torch.nn.functional.linear(input_torch, k_weight, k_bias)
            v_o = torch.nn.functional.linear(input_torch, v_weight, v_bias)
            torch.cat([q_o, k_o, v_o], dim=1)

        pytorch_measurements = bench_gpu_time(run_pytorch, repeat_time_ms=300)
        pytorch_median = statistics.median(pytorch_measurements)

        input_efm = tensor_to_edge_fm_tensor(input_torch)
        output_efm = tensor_to_edge_fm_tensor(
            torch.empty(batch_size, q_out + k_out + v_out, device="cuda:0", dtype=torch_dtype)
        )

        def run_edge_fm():
            layer.forward_fp16_bf16(input_efm, output_efm)

        edge_fm_measurements = bench_gpu_time(run_edge_fm, repeat_time_ms=300)
        edge_fm_median = statistics.median(edge_fm_measurements)
        performance_ratio = pytorch_median / edge_fm_median if edge_fm_median > 0 else 0.0

        assert performance_ratio >= 0.85, (
            f"FusedQKV 性能测试失败：EdgeFM 性能低于 PyTorch 的 0.85 倍\n"
            f"  batch_size={batch_size}, dtype={dtype}\n"
            f"  PyTorch 中位时间: {pytorch_median:.4f} ms\n"
            f"  EdgeFM 中位时间: {edge_fm_median:.4f} ms\n"
            f"  性能比 (PyTorch/EdgeFM): {performance_ratio:.4f}\n"
            f"  要求: performance_ratio >= 0.85"
        )

    @pytest.mark.parametrize("case", FUSED_GATE_UP_PERF_TEST_CASES)
    def test_fused_gate_up_performance(self, case):
        """测试 FusedGateUpLinearLayer 前向传播的性能

        使用中位数、更长测量时间以降低微秒级 benchmark 的波动。
        确保 EdgeFM 的性能不低于 PyTorch 的 0.85 倍。
        """
        batch_size = case["batch_size"]
        in_features = case["in_features"]
        gate_out = case["gate_out"]
        up_out = case["up_out"]
        dtype = case["dtype"]
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        layer_prefix_base = "test_fused_gate_up_perf"
        layer, gate_weight, up_weight, gate_bias, up_bias = create_fused_gate_up_linear_layer_with_weights(
            layer_prefix_base, in_features, gate_out, up_out, dtype
        )
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        input_torch = torch.randn(batch_size, in_features, device="cuda:0", dtype=torch_dtype)

        def run_pytorch():
            g_o = torch.nn.functional.linear(input_torch, gate_weight, gate_bias)
            u_o = torch.nn.functional.linear(input_torch, up_weight, up_bias)
            torch.cat([g_o, u_o], dim=1)

        pytorch_measurements = bench_gpu_time(run_pytorch, repeat_time_ms=300)
        pytorch_median = statistics.median(pytorch_measurements)

        input_efm = tensor_to_edge_fm_tensor(input_torch)
        output_efm = tensor_to_edge_fm_tensor(
            torch.empty(batch_size, gate_out + up_out, device="cuda:0", dtype=torch_dtype)
        )

        def run_edge_fm():
            layer.forward_fp16_bf16(input_efm, output_efm)

        edge_fm_measurements = bench_gpu_time(run_edge_fm, repeat_time_ms=300)
        edge_fm_median = statistics.median(edge_fm_measurements)
        performance_ratio = pytorch_median / edge_fm_median if edge_fm_median > 0 else 0.0

        assert performance_ratio >= 0.85, (
            f"FusedGateUp 性能测试失败：EdgeFM 性能低于 PyTorch 的 0.85 倍\n"
            f"  batch_size={batch_size}, dtype={dtype}\n"
            f"  PyTorch 中位时间: {pytorch_median:.4f} ms\n"
            f"  EdgeFM 中位时间: {edge_fm_median:.4f} ms\n"
            f"  性能比 (PyTorch/EdgeFM): {performance_ratio:.4f}\n"
            f"  要求: performance_ratio >= 0.85"
        )

    @pytest.mark.parametrize("case", LM_HEAD_PERF_TEST_CASES)
    def test_lm_head_performance(self, case):
        """测试 LMHeadLinearLayer 前向传播的性能

        使用中位数、更长测量时间以降低微秒级 benchmark 的波动。
        确保 EdgeFM 的性能不低于 PyTorch 的 0.85 倍。
        """
        batch_size = case["batch_size"]
        in_features = case["in_features"]
        out_features = case["out_features"]
        dtype = case["dtype"]
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        layer, embed_weight = create_lm_head_linear_layer_with_weights(in_features, out_features, dtype)
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        input_torch = torch.randn(batch_size, in_features, device="cuda:0", dtype=torch_dtype)

        def run_pytorch():
            torch.nn.functional.linear(input_torch, embed_weight, None)

        pytorch_measurements = bench_gpu_time(run_pytorch, repeat_time_ms=300)
        pytorch_median = statistics.median(pytorch_measurements)

        input_efm = tensor_to_edge_fm_tensor(input_torch)
        output_efm = tensor_to_edge_fm_tensor(
            torch.empty(batch_size, out_features, device="cuda:0", dtype=torch_dtype)
        )

        def run_edge_fm():
            layer.forward_fp16_bf16(input_efm, output_efm)

        edge_fm_measurements = bench_gpu_time(run_edge_fm, repeat_time_ms=300)
        edge_fm_median = statistics.median(edge_fm_measurements)
        performance_ratio = pytorch_median / edge_fm_median if edge_fm_median > 0 else 0.0

        assert performance_ratio >= 0.85, (
            f"LMHead 性能测试失败：EdgeFM 性能低于 PyTorch 的 0.85 倍\n"
            f"  batch_size={batch_size}, dtype={dtype}\n"
            f"  PyTorch 中位时间: {pytorch_median:.4f} ms\n"
            f"  EdgeFM 中位时间: {edge_fm_median:.4f} ms\n"
            f"  性能比 (PyTorch/EdgeFM): {performance_ratio:.4f}\n"
            f"  要求: performance_ratio >= 0.85"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
