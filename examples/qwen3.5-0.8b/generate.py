#!/usr/bin/env python3
"""
Qwen3.5-0.8B Transformers smoke test.

当前 EdgeFM 还没有 Qwen3.5 linear-attention loader；这个脚本用于验证
download.sh 下载出的 Hugging Face checkpoint 是否能被 Transformers 正常加载。
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Qwen3.5-0.8B Transformers smoke test.")
    parser.add_argument(
        "--model-dir",
        default=str(Path(__file__).resolve().parent / "qwen3.5-0.8b"),
        help="Local model directory produced by download.sh.",
    )
    parser.add_argument("--prompt", default="用一句话介绍 EdgeFM。", help="Text prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum generated tokens.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"错误: 模型目录不存在 {model_dir}")
        print("请先在 examples/qwen3.5-0.8b 下执行: bash download.sh")
        return 1

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        print(f"错误: 缺少 Transformers smoke test 依赖: {exc}")
        print("请安装较新的 transformers、torch 后重试。")
        return 1

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    print(processor.decode(generated, skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
