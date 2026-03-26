#!/usr/bin/env python3
"""
Qwen2.5-0.5B-Instruct 模型推理示例

此示例展示了如何使用 EdgeFM Python 接口进行 Qwen2.5-0.5B-Instruct 模型的推理。
支持纯文本推理和包含图像嵌入的多模态推理。
"""

import sys
import json
import torch
from pathlib import Path

# 添加项目根目录到路径（如果 edge_fm 模块不在系统路径中）
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "build" / "python"))

try:
    import edge_fm
except ImportError:
    print("错误: 无法导入 edge_fm 模块")
    print("请确保:")
    print("1. 已构建项目 (cmake && make)")
    print("2. Python 模块路径已添加到 PYTHONPATH")
    print("   例如: export PYTHONPATH=$PYTHONPATH:/path/to/edge-fm/build/python")
    sys.exit(1)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def example_text_only(engine: edge_fm.EdgeFM):
    """纯文本推理示例"""
    example_token_ids = [151643, 151644, 198, 2610, 525, 198]
    print(f"\n[示例 1] 纯文本推理")
    print(f"  输入: {example_token_ids}")
    
    request = edge_fm.Request(request_id=0, token_ids=example_token_ids)
    response = engine.generate(request)
    
    generated_tokens = response.token_ids()
    print(f"  生成 {len(generated_tokens)} 个 tokens: {generated_tokens}")


# def example_text_with_embedding(engine: edge_fm.EdgeFM, image_embedding, embedding_indices: list):
#     """文本+图像嵌入推理示例"""
#     example_token_ids = [151643, 151644, 198, 151644, 151645, 198, 2610, 525, 198]
#     print(f"\n[示例 2] 文本+图像嵌入推理")
#     print(f"  输入: {example_token_ids}")
#     print(f"  嵌入索引: {embedding_indices}")
    
#     capsule = image_embedding.__dlpack__()
#     embedding_tensor = edge_fm.Tensor.from_dlpack(capsule, copy_data=False)
    
#     request = edge_fm.Request(
#         request_id=1,
#         token_ids=example_token_ids,
#         embedding=embedding_tensor,
#         embedding_indices=embedding_indices
#     )
#     response = engine.generate(request)
    
#     generated_tokens = response.token_ids()
#     print(f"  生成 {len(generated_tokens)} 个 tokens: {generated_tokens}")


def main():
    """主函数"""
    script_dir = Path(__file__).parent
    config_path = script_dir / "engine_config.json"
    
    config = load_config(str(config_path))
    print(f"EdgeFM Qwen2.5-0.5B-Instruct - 配置: {config}")
    
    engine = edge_fm.EdgeFM(str(config_path))
    
    example_text_only(engine)
    
    embedding_indices = [2, 3]
    # embedding_indices 有 2 个索引，所以需要 2 个图像嵌入
    image_embedding = torch.randn(2, 4096, dtype=torch.float16, device='cuda')
    image_embedding = image_embedding.contiguous()
    example_text_with_embedding(engine, image_embedding, embedding_indices)


if __name__ == "__main__":
    main()
