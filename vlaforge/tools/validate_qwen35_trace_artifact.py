"""Validate a saved Qwen3.5 TorchScript artifact against official token output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


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


def graph_audit(traced) -> dict[str, object]:
    graph = traced.inlined_graph
    kinds = [node.kind() for node in graph.nodes()]
    forbidden_random = ("rand", "bernoulli", "dropout", "multinomial", "normal", "poisson")
    forbidden_io = ("read", "write", "socket", "print", "open")
    return {
        "python_op": any("PythonOp" in kind for kind in kinds),
        "random_ops": sorted({
            kind for kind in kinds
            if any(token in kind.lower() for token in forbidden_random)
        }),
        "external_io_ops": sorted({
            kind for kind in kinds
            if any(token in kind.lower() for token in forbidden_io)
        }),
        "node_count": len(kinds),
    }


def write(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
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
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    ).eval().requires_grad_(False)
    traced = torch.jit.load(str(args.artifact), map_location="cuda:0")
    records = []
    artifact_outputs = None
    input_unchanged = True
    for index, sample in enumerate(manifest["samples"]):
        image_path = args.images / sample["image"]
        if sha256(image_path) != sample["image_sha256"]:
            raise ValueError("image identity changed")
        text = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": args.prompt}
            ]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        with Image.open(image_path) as image:
            values = processor(
                text=[text], images=[image.convert("RGB")], return_tensors="pt"
            )
        values = {name: value.to("cuda:0") for name, value in values.items()}
        inputs = (
            values["input_ids"].contiguous(),
            values["attention_mask"].contiguous(),
            values["pixel_values"].contiguous(),
        )
        before = tuple(value.clone() for value in inputs)
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            official = model.generate(
                **values,
                max_new_tokens=args.new_tokens,
                min_new_tokens=args.new_tokens,
                do_sample=False,
                use_cache=True,
            )[:, -args.new_tokens:]
            traced_tokens, traced_logits = traced(*inputs)
        input_unchanged = input_unchanged and all(
            torch.equal(left, right) for left, right in zip(before, inputs, strict=True)
        )
        if not torch.equal(official.cpu(), traced_tokens.cpu()):
            raise ValueError(f"traced tokens differ from official at sample {index}")
        current_outputs = [
            {"index": 0, "name": "tokens", "dtype": "i64", "shape": list(traced_tokens.shape)},
            {"index": 1, "name": "logits", "dtype": "bf16", "shape": list(traced_logits.shape)},
        ]
        if artifact_outputs is None:
            artifact_outputs = current_outputs
        elif artifact_outputs != current_outputs:
            raise ValueError("artifact output signature changed across samples")
        folder = data / str(index)
        folder.mkdir()
        payloads = {
            "0.bin": bytes_of(inputs[0]),
            "1.bin": bytes_of(inputs[1]),
            "2.bin": bytes_of(inputs[2]),
            "3.bin": bytes_of(torch.ones((1,), dtype=torch.bool, device="cuda:0")),
            **generation_reference_payloads(traced_tokens, traced_logits),
        }
        for name, payload in payloads.items():
            (folder / name).write_bytes(payload)
        records.append({
            "index": index,
            "image": sample["image"],
            "tokens": official.cpu().reshape(-1).tolist(),
            "tokens_equal_official": True,
            "files": {
                name: {"sha256": sha256(folder / name),
                       "size_bytes": (folder / name).stat().st_size}
                for name in payloads
            },
        })
    audit = graph_audit(traced)
    if audit["python_op"] or audit["random_ops"] or audit["external_io_ops"] or not input_unchanged:
        raise ValueError(
            f"traced graph effect audit failed: {audit}, unchanged={input_unchanged}"
        )
    report = {
        "schema": "edgefm.qwen35.trace_artifact_validation/1",
        "status": "passed",
        "model": str(args.model),
        "artifact_sha256": sha256(args.artifact),
        "artifact_outputs": artifact_outputs,
        "graph_audit": audit,
        "inputs_unchanged": input_unchanged,
        "samples": records,
        "all_tokens_equal_official": True,
    }
    write(args.output / "report.json", report)
    print(json.dumps({
        "status": "passed",
        "samples": len(records),
        "artifact_sha256": report["artifact_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
