#!/usr/bin/env python3
"""Host-only static logical ONNX signatures; no inference or target timing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import onnx
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from vlaforge.analysis.onnx_inventory import onnx_operator_inventory

    payload = args.input.read_bytes()
    model = onnx.load_model_from_string(payload)
    report = onnx_operator_inventory(model, region_name=args.region)
    report.update(source={"path": str(args.input.resolve()), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)},
                  producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), onnx_version=onnx.__version__)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("region", "static_nodes", "unique_signatures", "shape_inference", "tuning_or_performance_claim")}))


if __name__ == "__main__":
    main()
