"""Trace exact explicit-state Qwen3.5 generation and prepare native runner inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from vlaforge.adapters.qwen3_5.qwen3_5_state import (
    Qwen3_5ExplicitPrefill,
    Qwen3_5GenerateExact,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_of(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def generation_reference_payloads(
    tokens: torch.Tensor,
    logits: torch.Tensor,
) -> dict[str, bytes]:
    """Write references using the TorchScript tuple index and typed names."""
    return {
        "direct-0.bin": bytes_of(tokens),
        "eager-0.bin": bytes_of(tokens),
        "direct.bin": bytes_of(logits),
        "eager.bin": bytes_of(logits),
    }


def write(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def graph_audit(traced) -> dict[str, object]:
    graph = traced.inlined_graph
    kinds = [node.kind() for node in graph.nodes()]
    forbidden_random = ("rand", "bernoulli", "dropout", "multinomial", "normal", "poisson")
    forbidden_io = ("read", "write", "socket", "print", "open")
    return {
        "python_op": any("PythonOp" in kind for kind in kinds),
        "random_ops": sorted({kind for kind in kinds if any(token in kind.lower() for token in forbidden_random)}),
        "external_io_ops": sorted({kind for kind in kinds if any(token in kind.lower() for token in forbidden_io)}),
        "node_count": len(kinds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument(
        "--prompt",
        default="Describe the visible objects and the robot interaction in one short sentence.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.output / "data"
    data.mkdir()
    manifest = json.loads((args.images / "manifest.json").read_text())
    if manifest.get("status") != "verified_lossless_rgb_images":
        raise ValueError("input image manifest is not accepted")
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    ).eval().requires_grad_(False)
    traced = None
    records = []
    for index, sample in enumerate(manifest["samples"]):
        image_path = args.images / sample["image"]
        if sha256(image_path) != sample["image_sha256"]:
            raise ValueError(f"image identity differs: {image_path}")
        text = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": args.prompt}
            ]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        with Image.open(image_path) as image:
            values = processor(
                text=[text],
                images=[image.convert("RGB")],
                return_tensors="pt",
            )
        values = {name: value.to("cuda:0") for name, value in values.items()}
        with torch.inference_mode():
            position_ids, rope_deltas = model.model.get_rope_index(
                values["input_ids"],
                image_grid_thw=values["image_grid_thw"],
                attention_mask=values["attention_mask"],
                mm_token_type_ids=values["mm_token_type_ids"],
            )
            official = model.generate(
                **values,
                max_new_tokens=args.new_tokens,
                min_new_tokens=args.new_tokens,
                do_sample=False,
                use_cache=True,
            )[:, -args.new_tokens:]
        prefill = Qwen3_5ExplicitPrefill(
            model,
            max_sequence_length=values["input_ids"].shape[1] + args.new_tokens - 1,
            position_ids=position_ids,
            text_position_ids=position_ids[:1],
            image_grid_thw=values["image_grid_thw"],
            rope_deltas=rope_deltas,
        )
        generate = Qwen3_5GenerateExact(
            prefill,
            prompt_length=values["input_ids"].shape[1],
            new_tokens=args.new_tokens,
        ).eval()
        inputs = (
            values["input_ids"].contiguous(),
            values["attention_mask"].contiguous(),
            values["pixel_values"].contiguous(),
        )
        with torch.inference_mode():
            eager_tokens, eager_logits = generate(*inputs)
        if traced is None:
            traced = torch.jit.trace(
                generate, inputs, strict=False, check_trace=False
            )
            traced.save(str(args.output / "qwen_generate_exact.pt"))
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            traced_tokens, traced_logits = traced(*inputs)
        if not torch.equal(official.cpu(), eager_tokens.cpu()):
            raise ValueError(f"eager exact tokens differ from official at sample {index}")
        if not torch.equal(eager_tokens, traced_tokens) or not torch.equal(
            eager_logits, traced_logits
        ):
            raise ValueError(f"traced output differs from eager at sample {index}")
        folder = data / str(index)
        folder.mkdir()
        payloads = {
            "0.bin": bytes_of(inputs[0]),
            "1.bin": bytes_of(inputs[1]),
            "2.bin": bytes_of(inputs[2]),
            "3.bin": bytes_of(torch.ones((1,), dtype=torch.bool, device="cuda:0")),
            **generation_reference_payloads(eager_tokens, eager_logits),
        }
        for name, payload in payloads.items():
            (folder / name).write_bytes(payload)
        records.append({
            "index": index,
            "image": sample["image"],
            "official_tokens": official.cpu().reshape(-1).tolist(),
            "eager_tokens": eager_tokens.cpu().reshape(-1).tolist(),
            "traced_tokens": traced_tokens.cpu().reshape(-1).tolist(),
            "tokens_equal": True,
            "logits_exact": True,
            "files": {
                name: {"sha256": sha256(folder / name),
                       "size_bytes": (folder / name).stat().st_size}
                for name in payloads
            },
        })
    artifact = args.output / "qwen_generate_exact.pt"
    before = tuple(value.clone() for value in inputs)
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        traced(*inputs)
    input_unchanged = all(
        torch.equal(left, right) for left, right in zip(before, inputs, strict=True)
    )
    audit = graph_audit(traced)
    if audit["python_op"] or audit["random_ops"] or audit["external_io_ops"] or not input_unchanged:
        raise ValueError(f"traced graph effect audit failed: {audit}, unchanged={input_unchanged}")
    report = {
        "schema": "edgefm.qwen35.native_trace/1",
        "status": "passed",
        "model": str(args.model),
        "images_manifest_sha256": sha256(args.images / "manifest.json"),
        "artifact": str(artifact),
        "artifact_sha256": sha256(artifact),
        "new_tokens": args.new_tokens,
        "artifact_outputs": [
            {"index": 0, "name": "tokens", "dtype": "i64", "shape": list(eager_tokens.shape)},
            {"index": 1, "name": "logits", "dtype": "bf16", "shape": list(eager_logits.shape)},
        ],
        "graph_audit": audit,
        "inputs_unchanged": input_unchanged,
        "samples": records,
    }
    write(args.output / "report.json", report)
    print(json.dumps({
        "status": "passed",
        "artifact": str(artifact),
        "artifact_sha256": report["artifact_sha256"],
        "graph_audit": audit,
        "samples": len(records),
    }, indent=2))


if __name__ == "__main__":
    main()
