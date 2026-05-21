#!/usr/bin/env python3
"""
Generate a Transformers reference dump for Qwen3.5 text-only greedy decode.

The dump is intentionally small and token-focused: EdgeFM v1 should match the
prefill sample token and every following greedy token exactly before we spend
time on kernel fusion or CUDA graph capture.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TRANSFORMERS = Path("/home/zhangzimo/Repos/public/transformers/src")
if LOCAL_TRANSFORMERS.exists():
    sys.path.insert(0, str(LOCAL_TRANSFORMERS))

DEFAULT_MODEL_PATH = PROJECT_ROOT / "examples" / "qwen3.5-0.8b" / "qwen3.5-0.8b"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "data" / "decode_dump_qwen3_5_0p8b"
DEFAULT_PROMPT = "用一句话介绍 EdgeFM。"


def parse_args():
    parser = argparse.ArgumentParser(description="Dump Qwen3.5 text-only decode reference")
    parser.add_argument("--model-path", default=os.environ.get("EDGE_FM_QWEN3_5_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path)
    if not (model_path / "config.json").exists():
        print(f"missing Qwen3.5 model config: {model_path / 'config.json'}", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    generated_tokens = []
    logits_argmax = []
    next_input_ids = input_ids
    past_key_values = None

    with torch.inference_mode():
        outputs = model(
            input_ids=next_input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens.append(int(next_token.item()))
        logits_argmax.append(int(next_token.item()))
        past_key_values = outputs.past_key_values

        for _ in range(1, args.num_steps):
            next_input_ids = next_token[:, None]
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)],
                dim=-1,
            )
            outputs = model(
                input_ids=next_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_tokens.append(int(next_token.item()))
            logits_argmax.append(int(next_token.item()))
            past_key_values = outputs.past_key_values

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "input_ids.npy", input_ids.detach().cpu().numpy().astype(np.int32))
    np.save(output_dir / "decode_tokens.npy", np.array(generated_tokens, dtype=np.int32))
    metadata = {
        "model_path": str(model_path),
        "prompt": args.prompt,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "input_length": int(input_ids.shape[-1]),
        "decode_tokens": generated_tokens,
        "transformers_path": str(LOCAL_TRANSFORMERS) if LOCAL_TRANSFORMERS.exists() else None,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote Qwen3.5 decode dump to {output_dir}")
    print("tokens:", generated_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
