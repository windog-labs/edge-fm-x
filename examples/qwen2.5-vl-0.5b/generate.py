"""
Qwen2.5-VL-0.5B 示例（文本侧 smoke test）。

当前目录下模型是 Llava/CLIP 风格的多模态封装。EdgeFM 这里先验证
engine_config.json、模型路径和文本侧生成链路是否可用。
"""

import json
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
build_python = project_root / "build" / "install" / "python"
if build_python.exists():
    sys.path.insert(0, str(build_python))

try:
    import edge_fm
except ImportError:
    print("错误: 无法导入 edge_fm，请先 build 并 make install")
    print("同时确认当前 Python 版本与 build/install/python 下 edge_fm 扩展模块的 ABI 一致。")
    sys.exit(1)


def main():
    script_dir = Path(__file__).parent
    config_path = script_dir / "engine_config.json"
    if not config_path.exists():
        print(f"错误: 配置文件不存在 {config_path}")
        sys.exit(1)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(
        "EdgeFM Qwen2.5-VL-0.5B 示例 - prefill_model_path="
        f"{config.get('prefill_model_path')}"
    )

    engine = edge_fm.EdgeFM(str(config_path))
    token_ids = [151643, 151644, 198, 2610, 525, 198]
    request = edge_fm.Request(request_id=0, token_ids=token_ids)
    response = engine.generate(request)
    tokens = list(response.token_ids())
    print(f"  生成 {len(tokens)} 个 tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")


if __name__ == "__main__":
    main()
