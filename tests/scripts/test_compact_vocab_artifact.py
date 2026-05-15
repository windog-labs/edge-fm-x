import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.compact_vocab.compact_vocab_artifact import (
    build_mapping_from_new_to_old,
    materialize_compact_vocab_artifact,
    validate_compact_vocab_artifact,
    validate_mapping,
)


def test_build_mapping_from_new_to_old_validates_inverse():
    mapping = build_mapping_from_new_to_old(
        original_vocab_size=6,
        new_to_old=[3, 4, 1],
        special_token_ids=[1, 3],
    )

    assert mapping["format"] == "edgefm.compact_vocab.v1"
    assert mapping["old_to_new"] == [-1, 2, -1, 0, 1, -1]
    assert mapping["new_to_old"] == [3, 4, 1]
    validate_mapping(mapping)


def test_build_mapping_rejects_pruned_special_token():
    with pytest.raises(ValueError, match="special token"):
        build_mapping_from_new_to_old(
            original_vocab_size=5,
            new_to_old=[0, 2, 4],
            special_token_ids=[1],
        )


def test_materialize_compact_vocab_artifact_crops_vocab_tensors_and_updates_config(tmp_path):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "compact"
    model_dir.mkdir()

    (model_dir / "config.json").write_text(
        json.dumps({"vocab_size": 6, "hidden_size": 4, "torch_dtype": "float16"}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    embed = torch.arange(24, dtype=torch.float16).reshape(6, 4)
    lm_head = embed + 100
    other = torch.arange(8, dtype=torch.float16).reshape(2, 4)
    save_file(
        {
            "model.embed_tokens.weight": embed,
            "lm_head.weight": lm_head,
            "model.layers.0.weight": other,
        },
        str(model_dir / "model.safetensors"),
    )

    mapping = build_mapping_from_new_to_old(
        original_vocab_size=6,
        new_to_old=[3, 4, 1],
        special_token_ids=[1, 3],
    )
    result = materialize_compact_vocab_artifact(
        input_model_dir=model_dir,
        output_dir=output_dir,
        mapping=mapping,
    )

    assert result["compact_vocab_size"] == 3
    out_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert out_config["vocab_size"] == 3
    out_mapping = json.loads((output_dir / "compact_vocab.json").read_text(encoding="utf-8"))
    assert out_mapping["old_to_new"] == [-1, 2, -1, 0, 1, -1]
    assert (output_dir / "tokenizer.json").exists()
    vocab_map = load_file(str(output_dir / "vocab_map.safetensors"))
    assert torch.equal(vocab_map["vocab_map"], torch.tensor([3, 4, 1], dtype=torch.int32))

    tensors = load_file(str(output_dir / "model.safetensors"))
    expected_rows = torch.tensor([3, 4, 1], dtype=torch.long)
    assert torch.equal(tensors["model.embed_tokens.weight"], embed.index_select(0, expected_rows))
    assert torch.equal(tensors["lm_head.weight"], lm_head.index_select(0, expected_rows))
    assert torch.equal(tensors["model.layers.0.weight"], other)

    validation = validate_compact_vocab_artifact(artifact_dir=output_dir)
    assert validation["valid"] is True
    assert validation["checked_vocab_tensor_names"] == ["lm_head.weight", "model.embed_tokens.weight"]
    assert validation["tokenizer_files"] == ["tokenizer.json"]


def test_validate_compact_vocab_artifact_rejects_config_vocab_mismatch(tmp_path):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "compact"
    model_dir.mkdir()

    (model_dir / "config.json").write_text(
        json.dumps({"vocab_size": 6, "hidden_size": 2}),
        encoding="utf-8",
    )
    save_file(
        {"model.embed_tokens.weight": torch.arange(12, dtype=torch.float16).reshape(6, 2)},
        str(model_dir / "model.safetensors"),
    )

    mapping = build_mapping_from_new_to_old(
        original_vocab_size=6,
        new_to_old=[0, 2, 4],
        special_token_ids=[0],
    )
    materialize_compact_vocab_artifact(
        input_model_dir=model_dir,
        output_dir=output_dir,
        mapping=mapping,
    )
    (output_dir / "config.json").write_text(
        json.dumps({"vocab_size": 6, "hidden_size": 2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="compact_vocab_size"):
        validate_compact_vocab_artifact(artifact_dir=output_dir)
