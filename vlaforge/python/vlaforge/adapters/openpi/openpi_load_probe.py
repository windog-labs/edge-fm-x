"""CPU-only strict real-checkpoint Policy load; no action inference or capture."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import time

from vlaforge.adapters.openpi.openpi_frontend import OpenPIConfig, load_openpi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config-name", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    loaded = load_openpi(OpenPIConfig(**vars(args), device="cpu"))
    parameters = tuple(loaded.model.parameters())
    if any(p.device.type != "cpu" for p in parameters):
        raise ValueError("load-only probe must not allocate model parameters on a GPU")
    dtype_elements = Counter()
    for parameter in parameters:
        dtype_elements[str(parameter.dtype)] += parameter.numel()
    print(
        json.dumps(
            {
                "schema": "vlaforge.openpi_policy_load/1",
                "evidence_level": "real-checkpoint-policy-load-only; no action inference, capture, or deployment",
                "pid": os.getpid(),
                "config_name": args.config_name,
                "strict_policy_load": "passed",
                "wall_time_seconds": time.monotonic() - start,
                "max_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "parameter_count": sum(p.numel() for p in parameters),
                "parameter_storage_bytes": sum(
                    p.numel() * p.element_size() for p in parameters
                ),
                "parameter_elements_by_dtype": dict(dtype_elements),
                "parameter_devices": sorted({str(p.device) for p in parameters}),
                "action_dim": loaded.model.config.action_dim,
                "action_horizon": loaded.model.config.action_horizon,
                "source": loaded.provenance["source"],
                "checkpoint": loaded.provenance["checkpoint"],
                "conversion_gate": loaded.provenance["load_gate"],
                "normalization_assets": loaded.provenance["assets"],
                "tokenizer": loaded.provenance["tokenizer"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
