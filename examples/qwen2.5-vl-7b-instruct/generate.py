#!/usr/bin/env python3
"""
Qwen2.5-VL-7B-Instruct 示例（EdgeFM 仅跑 LLM，图像需预计算 embedding 后通过 Request 注入）。

带图推理请参考 tests/scripts/dump_qwen2_5_vl_decode.py 与 tests/engine/test_qwen2_generate.py 中的 VL 用例。
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "build" / "install" / "python"))

try:
    import edge_fm
except ImportError:
    print("错误: 无法导入 edge_fm，请先 build 并 make install")
    sys.exit(1)


def main():
    script_dir = Path(__file__).parent
    config_path = script_dir / "engine_config.json"
    if not config_path.exists():
        print(f"错误: 配置文件不存在 {config_path}")
        sys.exit(1)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"EdgeFM Qwen2.5-VL-7B 示例 - prefill_model_path={config.get('prefill_model_path')}")

    engine = edge_fm.EdgeFM(str(config_path))
    token_ids = [151643, 151644, 198, 2610, 525, 198]
    request = edge_fm.Request(request_id=0, token_ids=token_ids)
    response = engine.generate(request)
    tokens = list(response.token_ids())
    print(f"  生成 {len(tokens)} 个 tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")


if __name__ == "__main__":
    main()
