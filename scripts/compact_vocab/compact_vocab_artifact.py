#!/usr/bin/env python3
"""Build and validate EdgeFM compact-vocab artifacts.

The packaging flow mirrors TensorRT-Edge-LLM's reduced-vocabulary shape:
`new_to_old` is the vocab map, and vocab-sized model tensors are cropped by
selecting rows from that map. EdgeFM additionally writes the inverse
`old_to_new` map because runtime requests arrive in original tokenizer ids.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


FORMAT = "edgefm.compact_vocab.v1"
VOCAB_TENSOR_NAMES = {
    "lm_head.weight",
    "model.lm_head.weight",
    "model.embed_tokens.weight",
}
TOKENIZER_FILE_PREFIXES = ("tokenizer",)
TOKENIZER_FILE_NAMES = {
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
}
VOCAB_MAP_TENSOR_NAME = "vocab_map"


def _as_int_list(values: Iterable[int], name: str) -> list[int]:
    out: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{name} must contain integers, got bool")
        out.append(int(value))
    return out


def build_mapping_from_new_to_old(
    *,
    original_vocab_size: int,
    new_to_old: Iterable[int],
    special_token_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Build an EdgeFM v1 mapping from a TRT-Edge-LLM style vocab_map."""
    new_to_old_list = _as_int_list(new_to_old, "new_to_old")
    special_ids = _as_int_list(special_token_ids or [], "special_token_ids")
    if original_vocab_size <= 0:
        raise ValueError("original_vocab_size must be positive")
    if not new_to_old_list:
        raise ValueError("new_to_old must not be empty")

    old_to_new = [-1] * original_vocab_size
    seen: set[int] = set()
    for new_id, old_id in enumerate(new_to_old_list):
        if old_id < 0 or old_id >= original_vocab_size:
            raise ValueError(f"new_to_old[{new_id}]={old_id} is out of range")
        if old_id in seen:
            raise ValueError(f"new_to_old contains duplicate original id {old_id}")
        seen.add(old_id)
        old_to_new[old_id] = new_id

    for token_id in special_ids:
        if token_id < 0 or token_id >= original_vocab_size:
            raise ValueError(f"special token {token_id} is out of range")
        if old_to_new[token_id] < 0:
            raise ValueError(f"special token {token_id} is pruned by the compact vocab map")

    mapping = {
        "format": FORMAT,
        "original_vocab_size": original_vocab_size,
        "compact_vocab_size": len(new_to_old_list),
        "old_to_new": old_to_new,
        "new_to_old": new_to_old_list,
        "special_token_ids": special_ids,
    }
    validate_mapping(mapping)
    return mapping


def validate_mapping(mapping: dict[str, Any]) -> None:
    if mapping.get("format") != FORMAT:
        raise ValueError(f"mapping format must be {FORMAT}")
    original_vocab_size = int(mapping["original_vocab_size"])
    compact_vocab_size = int(mapping["compact_vocab_size"])
    old_to_new = _as_int_list(mapping["old_to_new"], "old_to_new")
    new_to_old = _as_int_list(mapping["new_to_old"], "new_to_old")
    special_ids = _as_int_list(mapping.get("special_token_ids", []), "special_token_ids")

    if len(old_to_new) != original_vocab_size:
        raise ValueError("old_to_new length must equal original_vocab_size")
    if len(new_to_old) != compact_vocab_size:
        raise ValueError("new_to_old length must equal compact_vocab_size")

    for old_id, new_id in enumerate(old_to_new):
        if new_id < -1 or new_id >= compact_vocab_size:
            raise ValueError(f"old_to_new[{old_id}]={new_id} is out of range")
        if new_id >= 0 and new_to_old[new_id] != old_id:
            raise ValueError(f"old_to_new/new_to_old mismatch at original id {old_id}")

    seen: set[int] = set()
    for new_id, old_id in enumerate(new_to_old):
        if old_id < 0 or old_id >= original_vocab_size:
            raise ValueError(f"new_to_old[{new_id}]={old_id} is out of range")
        if old_id in seen:
            raise ValueError(f"new_to_old contains duplicate original id {old_id}")
        seen.add(old_id)
        if old_to_new[old_id] != new_id:
            raise ValueError(f"new_to_old/old_to_new mismatch at compact id {new_id}")

    for token_id in special_ids:
        if token_id < 0 or token_id >= original_vocab_size:
            raise ValueError(f"special token {token_id} is out of range")
        if old_to_new[token_id] < 0:
            raise ValueError(f"special token {token_id} is pruned")


def _copy_tokenizer_files(input_model_dir: Path, output_dir: Path) -> None:
    for path in input_model_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in TOKENIZER_FILE_NAMES or path.name.startswith(TOKENIZER_FILE_PREFIXES):
            shutil.copy2(path, output_dir / path.name)


def _update_vocab_size(config: dict[str, Any], compact_vocab_size: int) -> dict[str, Any]:
    updated = dict(config)
    if isinstance(updated.get("text_config"), dict) and "vocab_size" in updated["text_config"]:
        updated["text_config"] = dict(updated["text_config"])
        updated["text_config"]["vocab_size"] = compact_vocab_size
    if "vocab_size" in updated or "text_config" not in updated:
        updated["vocab_size"] = compact_vocab_size
    updated["edgefm_compact_vocab_format"] = FORMAT
    return updated


def _read_config_vocab_sizes(config: dict[str, Any]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    if "vocab_size" in config:
        sizes["vocab_size"] = int(config["vocab_size"])
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and "vocab_size" in text_config:
        sizes["text_config.vocab_size"] = int(text_config["vocab_size"])
    return sizes


def _crop_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    original_vocab_size: int,
    new_to_old: torch.Tensor,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        should_crop = name in VOCAB_TENSOR_NAMES and tensor.ndim >= 2 and tensor.shape[0] == original_vocab_size
        out[name] = tensor.index_select(0, new_to_old).contiguous() if should_crop else tensor
    return out


def materialize_compact_vocab_artifact(
    *,
    input_model_dir: str | Path,
    output_dir: str | Path,
    mapping: dict[str, Any],
) -> dict[str, Any]:
    input_model_dir = Path(input_model_dir)
    output_dir = Path(output_dir)
    validate_mapping(mapping)

    config_path = input_model_dir / "config.json"
    weights_path = input_model_dir / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {input_model_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {input_model_dir}")

    original_vocab_size = int(mapping["original_vocab_size"])
    compact_vocab_size = int(mapping["compact_vocab_size"])
    new_to_old = torch.tensor(mapping["new_to_old"], dtype=torch.long)

    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    (output_dir / "config.json").write_text(
        json.dumps(_update_vocab_size(config, compact_vocab_size), indent=2),
        encoding="utf-8",
    )
    (output_dir / "compact_vocab.json").write_text(
        json.dumps(mapping, indent=2),
        encoding="utf-8",
    )
    save_file(
        {VOCAB_MAP_TENSOR_NAME: new_to_old.to(torch.int32)},
        str(output_dir / "vocab_map.safetensors"),
    )

    tensors = load_file(str(weights_path), device="cpu")
    cropped = _crop_tensors(
        tensors,
        original_vocab_size=original_vocab_size,
        new_to_old=new_to_old,
    )
    save_file(cropped, str(output_dir / "model.safetensors"))
    _copy_tokenizer_files(input_model_dir, output_dir)

    return {
        "format": FORMAT,
        "input_model_dir": str(input_model_dir),
        "output_dir": str(output_dir),
        "original_vocab_size": original_vocab_size,
        "compact_vocab_size": compact_vocab_size,
        "vocab_map_path": str(output_dir / "vocab_map.safetensors"),
        "cropped_tensor_names": sorted(name for name in tensors if name in VOCAB_TENSOR_NAMES),
    }


def validate_compact_vocab_artifact(*, artifact_dir: str | Path) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    mapping_path = artifact_dir / "compact_vocab.json"
    config_path = artifact_dir / "config.json"
    weights_path = artifact_dir / "model.safetensors"
    vocab_map_path = artifact_dir / "vocab_map.safetensors"

    if not mapping_path.exists():
        raise FileNotFoundError(f"compact_vocab.json not found in {artifact_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {artifact_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {artifact_dir}")

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    validate_mapping(mapping)
    compact_vocab_size = int(mapping["compact_vocab_size"])
    original_vocab_size = int(mapping["original_vocab_size"])

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_vocab_sizes = _read_config_vocab_sizes(config)
    if not config_vocab_sizes:
        raise ValueError("config.json must define vocab_size or text_config.vocab_size")
    for key, value in config_vocab_sizes.items():
        if value != compact_vocab_size:
            raise ValueError(
                f"config {key} must equal compact_vocab_size={compact_vocab_size}, got {value}"
            )

    if not vocab_map_path.exists():
        raise FileNotFoundError(f"vocab_map.safetensors not found in {artifact_dir}")
    vocab_map = load_file(str(vocab_map_path), device="cpu")
    if VOCAB_MAP_TENSOR_NAME not in vocab_map:
        raise KeyError(f"vocab_map.safetensors must contain tensor '{VOCAB_MAP_TENSOR_NAME}'")
    expected_vocab_map = torch.tensor(mapping["new_to_old"], dtype=torch.int32)
    actual_vocab_map = vocab_map[VOCAB_MAP_TENSOR_NAME].to(torch.int32).reshape(-1)
    if not torch.equal(actual_vocab_map, expected_vocab_map):
        raise ValueError("vocab_map.safetensors tensor does not match compact_vocab.json new_to_old")

    checked_vocab_tensor_names: list[str] = []
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if name not in VOCAB_TENSOR_NAMES:
                continue
            shape = handle.get_slice(name).get_shape()
            if len(shape) >= 2:
                if int(shape[0]) != compact_vocab_size:
                    raise ValueError(
                        f"{name} first dimension must equal compact_vocab_size={compact_vocab_size}, got {shape[0]}"
                    )
                checked_vocab_tensor_names.append(name)
    if not checked_vocab_tensor_names:
        raise ValueError(
            "model.safetensors must contain at least one known vocab tensor "
            f"({', '.join(sorted(VOCAB_TENSOR_NAMES))})"
        )

    tokenizer_files = [
        path.name
        for path in artifact_dir.iterdir()
        if path.is_file()
        and (path.name in TOKENIZER_FILE_NAMES or path.name.startswith(TOKENIZER_FILE_PREFIXES))
    ]

    return {
        "format": FORMAT,
        "artifact_dir": str(artifact_dir),
        "valid": True,
        "original_vocab_size": original_vocab_size,
        "compact_vocab_size": compact_vocab_size,
        "config_vocab_sizes": config_vocab_sizes,
        "checked_vocab_tensor_names": sorted(checked_vocab_tensor_names),
        "tokenizer_files": sorted(tokenizer_files),
        "vocab_map_path": str(vocab_map_path),
    }


def _load_mapping(args: argparse.Namespace) -> dict[str, Any]:
    if args.mapping_json:
        mapping = json.loads(Path(args.mapping_json).read_text(encoding="utf-8"))
        validate_mapping(mapping)
        return mapping
    if args.new_to_old_json:
        if args.original_vocab_size is None:
            raise ValueError("--original-vocab-size is required with --new-to-old-json")
        new_to_old = json.loads(Path(args.new_to_old_json).read_text(encoding="utf-8"))
        return build_mapping_from_new_to_old(
            original_vocab_size=args.original_vocab_size,
            new_to_old=new_to_old,
            special_token_ids=args.special_token_id,
        )
    if args.vocab_map_safetensors:
        if args.original_vocab_size is None:
            raise ValueError("--original-vocab-size is required with --vocab-map-safetensors")
        tensors = load_file(str(args.vocab_map_safetensors), device="cpu")
        if VOCAB_MAP_TENSOR_NAME not in tensors:
            raise KeyError(f"vocab_map_safetensors must contain tensor '{VOCAB_MAP_TENSOR_NAME}'")
        return build_mapping_from_new_to_old(
            original_vocab_size=args.original_vocab_size,
            new_to_old=tensors[VOCAB_MAP_TENSOR_NAME].to(torch.int64).tolist(),
            special_token_ids=args.special_token_id,
        )
    raise ValueError("one of --mapping-json, --new-to-old-json, or --vocab-map-safetensors is required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--validate-artifact-dir")
    parser.add_argument("--mapping-json")
    parser.add_argument("--new-to-old-json")
    parser.add_argument("--vocab-map-safetensors")
    parser.add_argument("--original-vocab-size", type=int)
    parser.add_argument("--special-token-id", type=int, action="append", default=[])
    parser.add_argument("--json", action="store_true", help="print machine-readable summary")
    args = parser.parse_args()

    if args.validate_artifact_dir:
        result = validate_compact_vocab_artifact(artifact_dir=args.validate_artifact_dir)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(
                f"Validated compact vocab artifact at {result['artifact_dir']} "
                f"({result['original_vocab_size']} -> {result['compact_vocab_size']})"
            )
        return

    if not args.input_model_dir or not args.output_dir:
        parser.error("--input-model-dir and --output-dir are required unless --validate-artifact-dir is set")

    mapping = _load_mapping(args)
    result = materialize_compact_vocab_artifact(
        input_model_dir=args.input_model_dir,
        output_dir=args.output_dir,
        mapping=mapping,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"Wrote compact vocab artifact to {result['output_dir']} "
            f"({result['original_vocab_size']} -> {result['compact_vocab_size']})"
        )


if __name__ == "__main__":
    main()
