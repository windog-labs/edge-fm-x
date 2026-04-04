#!/usr/bin/env python3
"""
生成 Qwen2.5-VL 的 decode 对齐用 dump：VIT+projector 后的 image embedding、token_ids、
prefill/decode 中间结果。使用 KV-cached 逐步 greedy decode 作为参考（与 EdgeFM 行为一致）。

固定输入：tests/data/candy.JPG，文本 "What animal is on the candy?"
输出目录：tests/data/decode_dump_vl/

用法（在项目根目录）：
  python tests/scripts/dump_qwen2_5_vl_decode.py
  python tests/scripts/dump_qwen2_5_vl_decode.py --model_path /path/to/qwen2.5-vl-3b-instruct
"""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_IMAGE_PATH = project_root / "tests" / "data" / "candy.JPG"
DEFAULT_PROMPT = "What animal is on the candy?"
DEFAULT_NUM_STEPS = 20
DEFAULT_SEED = 42


def main():
    parser = argparse.ArgumentParser(description="Dump Qwen2.5-VL prefill+decode for alignment")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "tests" / "data" / "decode_dump_vl"),
    )
    parser.add_argument("--image_path", type=str, default=str(DEFAULT_IMAGE_PATH))
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    model_path = args.model_path or os.environ.get("EDGE_FM_QWEN_VL_MODEL_PATH")
    if not model_path:
        model_path = str(project_root / "examples" / "qwen2.5-vl-3b-instruct" / "qwen2.5-vl-3b-instruct")
    if not Path(model_path).exists() or not (Path(model_path) / "config.json").exists():
        print(f"错误: 模型路径不存在或缺少 config.json: {model_path}")
        sys.exit(1)

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from PIL import Image

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=False,
    )
    model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 50176

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"错误: 图像不存在: {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
    ).to(device)

    input_ids = inputs["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None and not isinstance(mm_token_type_ids, torch.Tensor):
        mm_token_type_ids = torch.tensor(mm_token_type_ids, dtype=torch.int32, device=device)

    vocab_size = model.config.text_config.vocab_size
    embed_token_id = getattr(
        model.config, "image_token_id",
        getattr(model.config.text_config, "image_token_id", vocab_size),
    )
    ids_flat = input_ids.cpu().numpy().ravel()
    num_image_tokens = int((ids_flat == embed_token_id).sum())

    # ========== Prefill: get image embeddings + KV cache + first token ==========
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )

    # Extract image embeddings from the embed layer output
    hidden_states = outputs.hidden_states
    embed_output = hidden_states[0]
    seq_len_total = embed_output.shape[1]
    positions = [i for i in range(seq_len_total) if input_ids[0, i].item() == embed_token_id]
    positions = positions[:num_image_tokens]
    if not positions:
        raise RuntimeError(f"无法确定图像 token 位置, embed_token_id={embed_token_id}")
    image_embeddings = embed_output[0, positions, :].float().cpu().numpy()

    # Compute M-RoPE 3D position_ids and rope_deltas.
    # Transformers changed Qwen2.5-VL get_rope_index() across versions:
    # older builds accepted mm_token_type_ids, newer ones derive multimodal
    # metadata from attention_mask / grid args only.
    if mm_token_type_ids is None:
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32, device=input_ids.device)
        mm_token_type_ids[input_ids == embed_token_id] = 1
    rope_sig = inspect.signature(model.model.get_rope_index)
    rope_kwargs = {
        "input_ids": input_ids,
        "image_grid_thw": image_grid_thw,
    }
    if "mm_token_type_ids" in rope_sig.parameters:
        rope_kwargs["mm_token_type_ids"] = mm_token_type_ids
    if "video_grid_thw" in rope_sig.parameters and inputs.get("video_grid_thw") is not None:
        rope_kwargs["video_grid_thw"] = inputs.get("video_grid_thw")
    if "second_per_grid_ts" in rope_sig.parameters and inputs.get("second_per_grid_ts") is not None:
        rope_kwargs["second_per_grid_ts"] = inputs.get("second_per_grid_ts")
    if "attention_mask" in rope_sig.parameters and inputs.get("attention_mask") is not None:
        rope_kwargs["attention_mask"] = inputs.get("attention_mask")
    position_ids_3d, rope_deltas_from_rope = model.model.get_rope_index(**rope_kwargs)
    position_ids_np = position_ids_3d[:, 0, :].cpu().numpy().astype(np.int32)

    # Prepare token_ids with incremented image token IDs
    token_ids = input_ids[0].cpu().numpy().astype(np.int32)
    embed_counter = 0
    for i in range(len(token_ids)):
        if token_ids[i] == embed_token_id:
            token_ids[i] = embed_token_id + embed_counter
            embed_counter += 1

    # Prefill logits and first decoded token
    prefill_logits = outputs.logits.float().cpu().numpy()
    past_kv = outputs.past_key_values
    rope_deltas = (
        rope_deltas_from_rope
        if rope_deltas_from_rope is not None
        else getattr(model.model, "rope_deltas", None)
        or torch.zeros(1, dtype=torch.long, device=device)
    )
    first_token = outputs.logits[0, -1].argmax().item()

    print(f"[dump] rope_deltas: {rope_deltas}", flush=True)
    print(f"[dump] first_token: {first_token}", flush=True)

    # ========== KV-cached greedy decode (no repetition_penalty) ==========
    input_len = input_ids.shape[1]
    cache_len = input_len
    decode_tokens = [first_token]
    decode_logits_list = []
    decode_input = torch.tensor([[first_token]], dtype=torch.long, device=device)

    for step in range(args.num_steps):
        cache_position = torch.arange(cache_len, cache_len + 1, dtype=torch.long, device=device)
        mrope_pos = (cache_position + rope_deltas).view(1, 1, 1).expand(3, 1, 1).to(torch.long)
        text_pos = cache_position.view(1, 1, 1)
        position_ids_4d = torch.cat([text_pos, mrope_pos], dim=0)  # [4, 1, 1]

        with torch.no_grad():
            out = model(
                input_ids=decode_input,
                past_key_values=past_kv,
                position_ids=position_ids_4d,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
        past_kv = out.past_key_values
        step_logits = out.logits[0, -1]  # [vocab]
        next_tok = step_logits.argmax().item()

        decode_logits_list.append(step_logits.float().cpu().numpy())
        decode_tokens.append(next_tok)
        decode_input = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        cache_len += 1

    # ========== Save ==========
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(min(args.num_steps, len(decode_tokens) - 1)):
        np.savez(
            out_dir / f"step_{step}.npz",
            input_token_id=np.int32(decode_tokens[step]),
            logits=decode_logits_list[step].astype(np.float32),
            next_token_id=np.int32(decode_tokens[step + 1]),
        )

    np.save(str(out_dir / "token_ids.npy"), token_ids)
    np.save(str(out_dir / "prefill_logits.npy"), prefill_logits)
    np.save(str(out_dir / "image_embeddings.npy"), image_embeddings.astype(np.float32))
    np.save(str(out_dir / "decode_tokens.npy"), np.array(decode_tokens, dtype=np.int32))
    np.save(str(out_dir / "position_ids.npy"), position_ids_np)

    manifest = {
        "model_path": str(Path(model_path).resolve()),
        "seed": args.seed,
        "prompt": args.prompt,
        "image_path": str(image_path.resolve()),
        "seq_len": int(token_ids.size),
        "num_decode_steps": args.num_steps,
        "vocab_size": vocab_size,
        "embed_token_id": embed_token_id,
        "image_embeddings_shape": list(image_embeddings.shape),
        "token_ids_shape": list(token_ids.shape),
        "decode_tokens": decode_tokens,
        "position_ids_shape": list(position_ids_np.shape),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[dump] Saved VL decode dump to {out_dir}")
    print(f"  prompt: {args.prompt!r}")
    print(f"  embed_token_id={embed_token_id}, image_embeddings shape={image_embeddings.shape}")
    print(f"  {args.num_steps} steps, decode_tokens={decode_tokens}")


if __name__ == "__main__":
    main()
