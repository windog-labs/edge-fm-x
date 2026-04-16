#!/usr/bin/env python3
"""
Qwen2.5-3B-Instruct 模型推理示例

此示例展示了如何使用 EdgeFM Python 接口进行 Qwen2.5-3B-Instruct 模型的推理。
支持纯文本推理。
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.edge_fm_build_paths import prepend_built_python_paths

prepend_built_python_paths(project_root)

try:
    import edge_fm
except ImportError:
    print("错误: 无法导入 edge_fm 模块")
    print("请确保:")
    print("1. 已完成对应平台的 configure/build/install")
    print("2. 或将 Python 模块路径加入: build-a800/install/python")
    sys.exit(1)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def example_text_only(engine: edge_fm.EdgeFM):
    """纯文本推理示例"""
    example_token_ids = [151643, 151644, 198, 2610, 525, 198]
    print("\n[示例] 纯文本推理")
    print(f"  输入 token 数量: {len(example_token_ids)}")

    request = edge_fm.Request(request_id=0, token_ids=example_token_ids)
    response = engine.generate(request)

    generated_tokens = list(response.token_ids())
    print(f"  生成 {len(generated_tokens)} 个 tokens: {generated_tokens[:20]}{'...' if len(generated_tokens) > 20 else ''}")


def main():
    """主函数"""
    script_dir = Path(__file__).parent
    config_path = script_dir / "engine_config.json"

    if not config_path.exists():
        print(f"错误: 配置文件不存在 {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    print(f"EdgeFM Qwen2.5-3B-Instruct - 配置: prefill_model_path={config.get('prefill_model_path')}")

    engine = edge_fm.EdgeFM(str(config_path))
    example_text_only(engine)


if __name__ == "__main__":
    main()
