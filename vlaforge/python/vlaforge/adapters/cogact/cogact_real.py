"""Strict real-checkpoint CogACT reference helpers, isolated from runtime core.

The public OpenVLA configuration route is explicitly a dependency candidate,
not evidence that the gated original Meta configuration was acquired.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

CHECKPOINT_REVISION = "6550bf0992f162fc5d74f14ffee30771a9433363"
CHECKPOINT_SHA256 = "1d35ec754c5c1ed7ab9ac22c9aba478e7269bab7db5b40df2703b2dc1e1009c0"
CHECKPOINT_SIZE = 30521280578
TOKENIZER_MODEL_SHA256 = "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"
PUBLIC_LLM_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
PUBLIC_LLM_FILES = {
    "config.json": "edd5c5cf6d7927e07465cf086ebe41f7b3ec8f3b128a51f71d6db14dad7ad8b1",
    "tokenizer.json": "8f5e2869e1807b8bb3c7717a294539c37e9de5d728ad60e31ef57e83cc5ea527",
    "tokenizer.model": TOKENIZER_MODEL_SHA256,
    "tokenizer_config.json": "f5f1d3ed015ebb71cf11686ff00bc8e0c25957bbf3952f95312cecc9b168fbea",
    "special_tokens_map.json": "cb90ee5cf5793aa444039af9795b00db59d3fef8942e7eca9cf1990bab370d61",
    "added_tokens.json": "ab43123267b190cb7990bc9f2ae4ad32dcb7ed029fb0fccb1b4e062c6f54a2a1",
    "configuration_prismatic.py": "68cc5ae34f1b46af3168d8d479cb81bb776965653453fd904aa8eefb6c8f9f68",
}
REQUIRED_GROUPS = ("action_model", "projector", "vision_backbone", "llm_backbone")
VISION_ASSETS = {
    "vit_large_patch14_reg4_dinov2.lvd142m": (
        "dino/pytorch_model.bin", "78971dc00a0c488f2b2dff17d6dcb7ebe787af70a703d8212b38fc6a33dbcdd4"),
    "vit_so400m_patch14_siglip_224": (
        "siglip/open_clip_pytorch_model.bin", "19d5c4b9a50a301af4de8e51383abf4c8b0315196969efd646a3e51c1f929ffd"),
}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_complete_groups(state: Any) -> None:
    if not isinstance(state, dict):
        raise TypeError("checkpoint model must be a state dictionary")
    missing = [name for name in REQUIRED_GROUPS if not isinstance(state.get(name), dict) or not state[name]]
    if missing:
        raise ValueError(f"incomplete checkpoint; random fallback forbidden: {missing}")


def validate_public_llm_lock(lock: dict[str, Any]) -> None:
    if (lock.get("repo") != "openvla/openvla-7b" or lock.get("revision") != PUBLIC_LLM_REVISION
            or lock.get("status") != "verified" or lock.get("gated") is not False
            or lock.get("official_metadata") is not True):
        raise ValueError("verified fixed official public OpenVLA dependency lock required")
    records = lock.get("files", [])
    if len(records) != len(PUBLIC_LLM_FILES) or {item.get("path"): item.get("sha256") for item in records} != PUBLIC_LLM_FILES:
        raise ValueError("public dependency lock must contain every exact pinned file")
    if any(item.get("verified") is not True or item.get("revision") != PUBLIC_LLM_REVISION for item in records):
        raise ValueError("public dependency contains an unverified or mixed revision file")


def prepare_public_llm_candidate(
    source: Path,
    lock_path: Path,
    output: Path,
    *,
    diagnostic_max_position_embeddings: int | None = None,
) -> dict[str, Any]:
    """Materialize the official OpenVLA text configuration without invented fields."""
    import importlib.metadata
    import shutil

    from transformers import AutoTokenizer, LlamaConfig

    if importlib.metadata.version("transformers") != "4.40.1":
        raise ValueError("official dependency candidate requires transformers==4.40.1")
    lock = json.loads(lock_path.read_text())
    validate_public_llm_lock(lock)
    for record in lock["files"]:
        if file_sha256(source / record["path"]) != record["sha256"]:
            raise ValueError("public dependency changed since verification")
    if file_sha256(source / "tokenizer.model") != TOKENIZER_MODEL_SHA256:
        raise ValueError("tokenizer vocabulary differs from official Meta LFS identity")
    config = json.loads((source / "config.json").read_text())
    if config["llm_backbone_id"] != "llama2-7b-pure":
        raise ValueError("public dependency is not the matching Llama backbone")
    text = LlamaConfig(**config["text_config"])
    source_resolved_fields = text.to_dict()
    if diagnostic_max_position_embeddings is not None:
        if type(diagnostic_max_position_embeddings) is not int or diagnostic_max_position_embeddings != 4096:
            raise ValueError("bounded diagnostic only supports an explicit max_position_embeddings=4096")
        text.max_position_embeddings = diagnostic_max_position_embeddings
    output.mkdir(parents=True, exist_ok=False)
    text.save_pretrained(output)
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
        shutil.copyfile(source / name, output / name)
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True, model_max_length=2048, padding_side="right")
    if (len(tokenizer), tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (32001, 1, 2, 32000):
        raise ValueError("public tokenizer special-token contract differs")
    if tokenizer.convert_tokens_to_ids("\u2581") != 29871:
        raise ValueError("CogACT cognition suffix token differs")
    report = {
        "schema": "cogact.public-llm-dependency-candidate/1", "strict_original_meta_config": False,
        "public_repo": lock["repo"], "public_revision": lock["revision"],
        "tokenizer_model_matches_official_meta_lfs_sha256": True,
        "text_config_constructor": "transformers==4.40.1 LlamaConfig(**official_openvla.text_config)",
        "explicit_source_fields": config["text_config"], "resolved_fields": text.to_dict(),
        "source_resolved_fields": source_resolved_fields,
        "diagnostic_overrides": ({"max_position_embeddings": diagnostic_max_position_embeddings}
                                  if diagnostic_max_position_embeddings is not None else {}),
        "diagnostic_is_original_meta_configuration_claim": False,
        "candidate_max_position_embeddings": text.max_position_embeddings,
        "original_meta_config_available": False,
        "unverified_difference": "Original Meta configuration is gated. Full field equivalence remains unverified.",
        "tokenizer": {"length": len(tokenizer), "vocabulary_size": tokenizer.vocab_size,
                      "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id,
                      "pad": tokenizer.pad_token_id, "cognition_suffix": [29871, 2],
                      "add_bos_token": tokenizer.add_bos_token, "add_eos_token": tokenizer.add_eos_token},
        "files": {path.name: file_sha256(path) for path in output.iterdir() if path.is_file()},
    }
    (output / "candidate.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_candidate_assets(checkpoint_root: Path, llm_root: Path, vision_root: Path):
    """Verify the complete fixed checkpoint and processor dependency files."""
    checkpoint = checkpoint_root / "checkpoints/CogACT-Base.pt"
    if checkpoint.stat().st_size != CHECKPOINT_SIZE or file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("complete pinned CogACT checkpoint identity mismatch")
    for filename, digest in {
        "config.json": "1895f14441591fab96fe477928a21a194fd76618009a89a5383294aa33d1acb7",
        "dataset_statistics.json": "b9f87dcc2b4449ca552066aecb90337507848810a9cf3d0cf18b9f86f03be61c",
    }.items():
        if file_sha256(checkpoint_root / filename) != digest:
            raise ValueError("checkpoint configuration/statistics identity mismatch")
    candidate = json.loads((llm_root / "candidate.json").read_text())
    if candidate["schema"] != "cogact.public-llm-dependency-candidate/1" or candidate["strict_original_meta_config"]:
        raise ValueError("explicit public dependency candidate label required")
    for name, digest in candidate["files"].items():
        if file_sha256(llm_root / name) != digest:
            raise ValueError("LLM candidate file changed")
    for filename, digest in VISION_ASSETS.values():
        if file_sha256(vision_root / filename) != digest:
            raise ValueError("pinned official vision asset identity mismatch")
    config = json.loads((checkpoint_root / "config.json").read_text())
    if (config["diffusion_model_type"], config["future_action_window_size"], config["past_action_window_size"]) != ("DiT-B", 15, 0):
        raise ValueError("unexpected official CogACT checkpoint configuration")
    return candidate


def load_verified_candidate(checkpoint_root: Path, llm_root: Path, vision_root: Path):
    """Call the untouched official loader using verified local dependency paths."""
    import timm
    import torch
    from prismatic.models.backbones.llm.llama2 import LLAMA2_MODELS
    from vla.load import load_vla

    candidate = verify_candidate_assets(checkpoint_root, llm_root, vision_root)
    checkpoint = checkpoint_root / "checkpoints/CogACT-Base.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["model"]
    require_complete_groups(state)
    create_model = timm.create_model

    def local_vision(name, *args, **kwargs):
        if name not in VISION_ASSETS or kwargs.get("pretrained") is not True:
            raise ValueError("unexpected official vision factory request")
        kwargs["pretrained_cfg_overlay"] = {"file": str(vision_root / VISION_ASSETS[name][0]), "hf_hub_id": None}
        return create_model(name, *args, **kwargs)

    entry = dict(LLAMA2_MODELS["llama2-7b-pure"], hf_hub_path=str(llm_root))
    with patch.dict(LLAMA2_MODELS, {"llama2-7b-pure": entry}), patch.object(timm, "create_model", local_vision):
        model = load_vla(checkpoint, load_for_training=False, action_model_type="DiT-B",
                         future_action_window_size=15, past_action_window_size=0, use_ema=False)
    modules = {"action_model": model.action_model, "projector": model.vlm.projector,
               "vision_backbone": model.vlm.vision_backbone, "llm_backbone": model.vlm.llm_backbone}
    coverage = {}
    for name, module in modules.items():
        actual = module.state_dict()
        if actual.keys() != state[name].keys():
            raise ValueError(f"strict key coverage mismatch for {name}")
        unequal = [key for key in actual if not torch.equal(actual[key], state[name][key])]
        if unequal:
            raise ValueError(f"full installed weights differ for {name}: {unequal[:3]}")
        coverage[name] = {"tensors": len(actual), "all_installed_weights_equal": True,
                          "parameters": sum(value.numel() for value in module.parameters()),
                          "buffers": sum(value.numel() for value in module.buffers())}
    model.eval()
    return model, {"checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_revision": CHECKPOINT_REVISION,
                   "coverage": coverage, "random_fallback": False, "dependency_candidate": candidate,
                   "parameter_count": sum(value.numel() for value in model.parameters())}


def rng_snapshot() -> dict[str, Any]:
    import random

    import numpy as np
    import torch

    return {"torch_cpu": torch.get_rng_state().clone(), "torch_cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(state: dict[str, Any]) -> None:
    import random

    import numpy as np
    import torch

    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


@contextmanager
def record_official_scheduler(model):
    """Observe the original scheduler, including RNG consumed when DDIM eta=0."""
    import torch

    scheduler = model.action_model.ddim_diffusion
    if scheduler is None or scheduler.num_timesteps != 10:
        raise ValueError("initialize the official N10 DDIM scheduler before recording")
    trace: dict[str, Any] = {"random_draws": [], "steps": [], "cfg_calls": [], "rotary_positions": []}
    randn, randn_like = torch.randn, torch.randn_like
    ddim_sample = scheduler.ddim_sample
    cfg = model.action_model.net.forward_with_cfg

    def recorded_randn(*args, **kwargs):
        value = randn(*args, **kwargs)
        trace["random_draws"].append({"kind": "randn", "value": value.detach().cpu().clone()})
        return value

    def recorded_randn_like(*args, **kwargs):
        value = randn_like(*args, **kwargs)
        trace["random_draws"].append({"kind": "randn_like", "value": value.detach().cpu().clone()})
        return value

    def recorded_cfg(x, timestep, z, cfg_scale):
        output = cfg(x, timestep, z, cfg_scale)
        trace["cfg_calls"].append({"x": x.detach().cpu().clone(), "timestep": timestep.detach().cpu().clone(),
                                    "z": z.detach().cpu().clone(), "cfg_scale": cfg_scale,
                                    "output": output.detach().cpu().clone()})
        return output

    def recorded_step(*args, **kwargs):
        before = rng_snapshot()
        result = ddim_sample(*args, **kwargs)
        trace["steps"].append({"rng_before": before, "rng_after": rng_snapshot(),
                               "output": {key: value.detach().cpu().clone() for key, value in result.items()}})
        return result

    def record_positions(_module, args):
        trace["rotary_positions"].append(args[1].detach().cpu().clone())

    rotary = model.vlm.llm_backbone.llm.model.layers[0].self_attn.rotary_emb
    hook = rotary.register_forward_pre_hook(record_positions)
    try:
        with patch.object(torch, "randn", recorded_randn), patch.object(torch, "randn_like", recorded_randn_like), \
                patch.object(scheduler, "ddim_sample", recorded_step), \
                patch.object(model.action_model.net, "forward_with_cfg", recorded_cfg):
            yield trace
    finally:
        hook.remove()
