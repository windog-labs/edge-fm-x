"""Pinned OpenPI source and strict, offline checkpoint conversion gates.

This module does not download weights or silently instantiate a substitute model.
Conversion requires the complete, checksum-verified official GCS object inventory.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Any, Mapping


OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
OPENPI_REPOSITORY = "https://github.com/Physical-Intelligence/openpi"
TRANSFORMERS_VERSION = "4.53.2"
_UNUSED_EXPERT_HEAD = "paligemma_with_expert.gemma_expert.lm_head.weight"


def file_digest(path: Path, *, include_crc32c: bool = False) -> dict[str, object]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    crc = None
    if include_crc32c:
        import google_crc32c

        crc = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(block)
            sha256.update(block)
            md5.update(block)
            if crc is not None:
                crc.update(block)
    result = {
        "size_bytes": size,
        "sha256": sha256.hexdigest(),
        "md5_base64": base64.b64encode(md5.digest()).decode("ascii"),
    }
    if crc is not None:
        result["crc32c_base64"] = base64.b64encode(crc.digest()).decode("ascii")
    return result


def gcs_checksum_type(item: Mapping[str, Any]) -> str:
    """Accept ordinary MD5 metadata or explicitly identified GCS composites."""
    if (
        int(item["size"]) < 0
        or not str(item.get("generation", "")).isdigit()
        or int(item["generation"]) <= 0
    ):
        raise ValueError("GCS object requires frozen size and positive generation")
    component_count = item.get("componentCount")
    if "componentCount" in item:
        if type(component_count) is not int or component_count < 1:
            raise ValueError("invalid composite componentCount")
        if "md5Hash" in item:
            raise ValueError("mixed composite/MD5 object metadata is invalid")
        if not item.get("crc32c"):
            raise ValueError("composite object requires an official CRC32C")
        kind = "crc32c"
    else:
        if not item.get("md5Hash"):
            raise ValueError("ordinary object requires an official MD5")
        kind = "md5"
    for field, size in (("md5Hash", 16), ("crc32c", 4)):
        if field in item:
            try:
                checksum = base64.b64decode(item[field], validate=True)
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid GCS {field} encoding") from error
            if len(checksum) != size:
                raise ValueError(f"invalid GCS {field} length")
    return kind


def verify_gcs_object(path: Path, item: Mapping[str, Any]) -> dict[str, Any]:
    kind = gcs_checksum_type(item)
    actual = file_digest(path, include_crc32c="crc32c" in item)
    if actual["size_bytes"] != int(item["size"]):
        raise ValueError(f"checkpoint size mismatch: {item['name']}")
    if kind == "md5" and actual["md5_base64"] != item["md5Hash"]:
        raise ValueError(f"checkpoint MD5 mismatch: {item['name']}")
    if "crc32c" in item and actual["crc32c_base64"] != item["crc32c"]:
        raise ValueError(f"checkpoint CRC32C mismatch: {item['name']}")
    return {
        "object": dict(item),
        "local": actual,
        "checksum_type": kind,
        "md5_verified": kind == "md5",
        "crc32c_verified": "crc32c" in item,
    }


def verify_openpi_source(source_root: str | Path) -> dict[str, object]:
    root = Path(source_root).resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != OPENPI_REVISION:
        raise ValueError(f"OpenPI revision mismatch: {revision}")
    subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD"], check=True)
    paths = (
        "src/openpi/models_pytorch/pi0_pytorch.py",
        "src/openpi/models_pytorch/gemma_pytorch.py",
        "src/openpi/models_pytorch/preprocessing_pytorch.py",
        "src/openpi/models/pi0_config.py",
        "src/openpi/models/tokenizer.py",
        "src/openpi/policies/policy.py",
        "src/openpi/policies/policy_config.py",
        "src/openpi/training/config.py",
        "examples/convert_jax_model_to_pytorch.py",
        "uv.lock",
    )
    return {
        "repository": OPENPI_REPOSITORY,
        "revision": revision,
        "root": str(root),
        "files": {name: file_digest(root / name) for name in paths},
    }


def import_openpi(source_root: str | Path):
    """Use the verified checkout, refusing a previously imported foreign OpenPI."""
    root = Path(source_root).resolve(strict=True)
    verify_openpi_source(root)
    existing = sys.modules.get("openpi")
    if existing is not None:
        locations = tuple(Path(p).resolve() for p in existing.__path__)
        if locations != (root / "src/openpi",):
            raise ValueError(f"another OpenPI package is already imported: {locations}")
    for relative in ("packages/openpi-client/src", "src"):
        path = str(root / relative)
        if path not in sys.path:
            sys.path.insert(0, path)
    transformers = importlib.import_module("transformers")
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError(f"OpenPI requires transformers {TRANSFORMERS_VERSION}")
    installed = Path(transformers.__file__).resolve().parent
    patches = root / "src/openpi/models_pytorch/transformers_replace"
    for patch in sorted(patches.rglob("*.py")):
        relative = patch.relative_to(patches)
        target = installed / relative
        if not target.is_file() or file_digest(patch) != file_digest(target):
            raise ValueError(f"OpenPI transformers patch mismatch: {relative}")
    return importlib.import_module("openpi.training.config")


def verify_gcs_checkpoint(
    checkpoint_dir: str | Path, inventory_path: str | Path, *, checkpoint_name: str
) -> dict[str, object]:
    """Verify frozen size and provider checksum before Orbax consumes any bytes."""
    root = Path(checkpoint_dir).resolve(strict=True)
    inventory = Path(inventory_path)
    manifest = json.loads(inventory.read_text())
    if manifest.get("nextPageToken") or not manifest.get("items"):
        raise ValueError("GCS inventory is empty or incomplete (pagination remains)")
    prefix = f"checkpoints/{checkpoint_name}/"
    records = []
    seen = set()
    for item in manifest["items"]:
        name = item["name"]
        if not name.startswith(prefix):
            raise ValueError(f"object outside checkpoint prefix: {name}")
        relative = PurePosixPath(name.removeprefix(prefix))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"invalid object path: {name}")
        if str(relative) in seen:
            raise ValueError(f"duplicate object: {name}")
        seen.add(str(relative))
        path = (root / str(relative)).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"object escapes checkpoint directory: {name}")
        records.append(verify_gcs_object(path, item))
    if not any(name.startswith("params/") for name in seen) or not any(
        name.startswith("assets/") and name.endswith("/norm_stats.json")
        for name in seen
    ):
        raise ValueError("checkpoint requires both params and normalization assets")
    return {"inventory": file_digest(inventory), "objects": records}


def load_converted_state_dict(
    model: Any,
    state_dict: Mapping[str, Any],
    *,
    unused_parameters: tuple[str, ...] = (),
) -> dict[str, object]:
    """Resolve genuine tied aliases and explicitly zero unused parameters only.

    All executable parameter keys and shapes must be present. This is deliberately
    stricter than the upstream converter's unchecked ``strict=False`` load.
    """
    import torch

    expected = model.state_dict()
    unexpected = sorted(set(state_dict) - set(expected))
    if unexpected:
        raise ValueError(f"unexpected converted parameters: {unexpected}")
    parameters = dict(model.named_parameters(remove_duplicate=False))
    supplied_aliases: dict[int, str] = {}
    for name, tensor in state_dict.items():
        if name in parameters:
            identity = id(parameters[name])
            previous = supplied_aliases.get(identity)
            if previous is not None and not torch.equal(tensor, state_dict[previous]):
                raise ValueError(
                    f"conflicting tied converted parameters: {previous}, {name}"
                )
            supplied_aliases[identity] = name
    unknown_unused = set(unused_parameters) - set(parameters)
    if unknown_unused:
        raise ValueError(
            f"unused parameter declaration mismatch: {sorted(unknown_unused)}"
        )
    complete = dict(state_dict)
    aliases: dict[str, str] = {}
    inactive: dict[str, int] = {}
    missing = []
    for name in set(expected) - set(complete):
        same = next(
            (
                key
                for key in state_dict
                if name in parameters
                and key in parameters
                and parameters[key] is parameters[name]
            ),
            None,
        )
        if same is not None:
            complete[name] = complete[same]
            aliases[name] = same
        elif name in unused_parameters:
            complete[name] = torch.zeros_like(expected[name])
            inactive[name] = expected[name].numel()
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"missing executable converted parameters: {sorted(missing)}")
    model.load_state_dict(complete, strict=True)
    return {
        "strict": True,
        "missing_required": [],
        "unexpected": [],
        "resolved_tied_aliases": aliases,
        "zeroed_unused_parameters": inactive,
    }


def convert_openpi_checkpoint(
    *,
    source_root: str | Path,
    checkpoint_dir: str | Path,
    inventory_path: str | Path,
    checkpoint_name: str,
    config_name: str,
    output_dir: str | Path,
) -> dict[str, object]:
    """CPU conversion using official tensor transforms with strict load/assets gates."""
    import dataclasses
    import numpy as np
    import safetensors.torch
    import torch

    source_root, checkpoint_dir = Path(source_root), Path(checkpoint_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"conversion output already exists: {output}")
    source = verify_openpi_source(source_root)
    inputs = verify_gcs_checkpoint(
        checkpoint_dir, inventory_path, checkpoint_name=checkpoint_name
    )
    configs = import_openpi(source_root)
    config = dataclasses.replace(
        configs.get_config(config_name).model, pytorch_compile_mode=None
    )
    if config.pi05 != (checkpoint_name == "pi05_base"):
        raise ValueError("checkpoint family does not match selected config")
    if checkpoint_name not in ("pi0_base", "pi05_base"):
        raise ValueError("conversion supports the two inventoried base checkpoints")
    if (
        config.paligemma_variant != "gemma_2b"
        or config.action_expert_variant != "gemma_300m"
    ):
        raise ValueError("base checkpoint architecture mismatch")

    converter_path = source_root / "examples/convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location(
        "_vf_openpi_conversion", converter_path
    )
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    pi0 = importlib.import_module("openpi.models_pytorch.pi0_pytorch")
    gemma = importlib.import_module("openpi.models.gemma")
    initial = converter.slice_initial_orbax_checkpoint(str(checkpoint_dir), "float32")
    with torch.device("cpu"):
        model = pi0.PI0Pytorch(config).eval()
    prefix, expert = converter.slice_paligemma_state_dict(
        initial["paligemma_params"], model.paligemma_with_expert.paligemma.config
    )
    # The pinned upstream slicer checks the path string despite accepting pi05.
    # Give it only a family label; it does not perform I/O using this argument.
    expert = converter.slice_gemma_state_dict(
        expert,
        gemma.get_config(config.action_expert_variant),
        num_expert=1,
        checkpoint_dir="pi05" if config.pi05 else "pi0",
        pi05=config.pi05,
    )
    projections = (
        ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
        if config.pi05
        else (
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        )
    )
    converted = {**prefix, **expert}
    for name in projections:
        for origin, target in (("kernel", "weight"), ("bias", "bias")):
            value = initial["projection_params"][name][origin]
            if isinstance(value, dict):
                value = value["value"]
            tensor = torch.from_numpy(np.array(value))
            converted[f"{name}.{target}"] = tensor.T if target == "weight" else tensor
    gate = load_converted_state_dict(
        model, converted, unused_parameters=(_UNUSED_EXPERT_HEAD,)
    )
    model = model.to(torch.bfloat16)
    output.mkdir(parents=True)
    weight_path = output / "model.safetensors"
    safetensors.torch.save_model(model, str(weight_path))
    # Use checkpoint/assets, not checkpoint.parent/assets from the upstream script.
    shutil.copytree(checkpoint_dir / "assets", output / "assets")
    assets = {
        str(path.relative_to(output)): file_digest(path)
        for path in sorted((output / "assets").rglob("*.json"))
    }
    if not assets:
        raise ValueError("converted checkpoint has no normalization assets")
    report = {
        "schema": "vlaforge.openpi_conversion/1",
        "evidence_level": "converted-checkpoint-not-inference",
        "source": source,
        "inputs": inputs,
        "config_name": config_name,
        "model_config": dataclasses.asdict(config),
        "load_gate": gate,
        "checkpoint": file_digest(weight_path),
        "assets": assets,
        "torch_version": torch.__version__,
        "source_restore_precision": "float32",
        "converted_storage_precision": "bfloat16",
        "casting_policy": "pinned-upstream-converter-default",
        "jax_pytorch_numerical_equivalence": "not-assessed",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "zeroed_unused_parameter_count": sum(gate["zeroed_unused_parameters"].values()),
    }
    (output / "vlaforge_conversion.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--inventory-path", required=True)
    parser.add_argument(
        "--checkpoint-name", choices=("pi0_base", "pi05_base"), required=True
    )
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = convert_openpi_checkpoint(**vars(args))
    print(
        json.dumps(
            {"checkpoint": report["checkpoint"], "load_gate": report["load_gate"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
