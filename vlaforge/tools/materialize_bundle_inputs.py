"""Materialize named binary inputs from a saved NumPy input series."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", type=Path)
    args = parser.parse_args()
    import numpy as np

    report = json.loads((args.source / "report.json").read_text())
    if report.get("status") != "real_series_partition_verified":
        raise ValueError("source series was not independently verified")
    expected_names = None
    if args.schema:
        schema = json.loads(args.schema.read_text())
        expected_names = [item["name"] for item in schema["inputs"]]
    args.output.mkdir(parents=True, exist_ok=False)
    samples = sorted(args.source.glob("sample-*"), key=lambda path: int(path.name.split("-")[1]))
    if len(samples) != report.get("count"):
        raise ValueError("source sample count does not match its report")
    entries = []
    for sample in samples:
        target = args.output / sample.name
        target.mkdir()
        with np.load(sample / "inputs.npz", allow_pickle=False) as pack:
            names = list(pack.files)
            if expected_names is not None and names != expected_names:
                raise ValueError(f"input names differ from bundle schema: {names!r}")
            files = {}
            for name in names:
                path = target / f"{name}.bin"
                np.ascontiguousarray(pack[name]).tofile(path)
                files[name] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
        entries.append({"sample": sample.name, "files": files})
    manifest = {
        "schema": "vlaforge.bundle-input-series/1",
        "source_report_sha256": digest(args.source / "report.json"),
        "count": len(entries),
        "samples": entries,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "materialized", "count": len(entries), "manifest_sha256": digest(args.output / "manifest.json")}))


if __name__ == "__main__":
    main()
