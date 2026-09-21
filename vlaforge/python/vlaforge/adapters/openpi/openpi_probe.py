"""Reproducible CPU OpenPI environment probe without allocating model weights."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import platform
import sys

from vlaforge.adapters.openpi.openpi_assets import verify_openpi_tokenizer
from vlaforge.adapters.openpi.openpi_checkpoint import (
    file_digest,
    import_openpi,
    verify_openpi_source,
)


def probe_environment(source_root: str | Path, *, require_cpu: bool = False) -> dict:
    import jax
    import torch
    import transformers

    root = Path(source_root).resolve(strict=True)
    configs = import_openpi(root)
    devices = jax.devices()
    if require_cpu and (
        torch.cuda.is_available() or any(d.platform != "cpu" for d in devices)
    ):
        raise ValueError("this environment probe requires CPU-only execution")
    spec = importlib.util.spec_from_file_location(
        "_vf_openpi_conversion_probe", root / "examples/convert_jax_model_to_pytorch.py"
    )
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    patch_root = root / "src/openpi/models_pytorch/transformers_replace"
    return {
        "evidence_level": "dependency/source/processor/config probe only; no checkpoint conversion or model inference",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "torch": torch.__version__,
        "jax": jax.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "jax_devices": [str(d) for d in devices],
        "source": verify_openpi_source(root),
        "tokenizer": verify_openpi_tokenizer(),
        "converter_import": "passed",
        "configs": {
            name: asdict(configs.get_config(name).model)
            for name in ("pi0_aloha", "pi05_aloha")
        },
        "transformers_patches": {
            str(p.relative_to(patch_root)): file_digest(p)
            for p in sorted(patch_root.rglob("*.py"))
        },
        "adapter_source": {
            p.name: file_digest(p)
            for p in sorted(Path(__file__).parent.glob("openpi_*.py"))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--require-cpu", action="store_true")
    print(
        json.dumps(
            probe_environment(**vars(parser.parse_args())), indent=2, default=str
        )
    )


if __name__ == "__main__":
    main()
