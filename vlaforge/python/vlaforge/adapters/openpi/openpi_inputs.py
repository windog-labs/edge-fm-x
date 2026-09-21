"""Frozen real ALOHA input acquisition and pickle-free OpenPI observation packs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path

from vlaforge.adapters.openpi.openpi_checkpoint import file_digest, import_openpi


ALOHA_DATASET = "physical-intelligence/aloha_pen_uncap_diverse"
ALOHA_REVISION = "e82d8b40b8ac66c0b40273dd80a077dfc40b732e"
ALOHA_EPISODE = "data/chunk-000/episode_000000.parquet"
ALOHA_EPISODE_SHA256 = (
    "f0cc336aadee7ee21225581998f80253b597721a3adbfcff8b1e4e6f25691bc2"
)
ALOHA_EPISODE_BYTES = 576308575


def download_aloha_episode(cache_root: str | Path) -> dict:
    """Acquire only one pinned official episode and its small schema/task files."""
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    names = (ALOHA_EPISODE, "meta/info.json", "meta/tasks.jsonl")
    entries = HfApi().get_paths_info(
        ALOHA_DATASET, list(names), repo_type="dataset", revision=ALOHA_REVISION
    )
    entries = {entry.path: entry for entry in entries}
    if set(entries) != set(names):
        raise ValueError("pinned dataset metadata is incomplete")
    files = {}
    for name in names:
        entry = entries[name]
        lfs_sha = entry.lfs.sha256 if entry.lfs else None
        if name == ALOHA_EPISODE and (
            entry.size != ALOHA_EPISODE_BYTES or lfs_sha != ALOHA_EPISODE_SHA256
        ):
            raise ValueError("canonical real episode metadata changed")
        path = Path(
            hf_hub_download(
                ALOHA_DATASET,
                name,
                repo_type="dataset",
                revision=ALOHA_REVISION,
                cache_dir=str(root),
            )
        )
        if not path.resolve().is_relative_to(root):
            raise ValueError("dataset cache path escapes the isolated cache")
        digest = file_digest(path)
        if digest["size_bytes"] != entry.size:
            raise ValueError(f"dataset size mismatch: {name}")
        if lfs_sha:
            if digest["sha256"] != lfs_sha:
                raise ValueError(f"dataset LFS SHA256 mismatch: {name}")
        else:
            contents = path.read_bytes()
            git_sha = hashlib.sha1(
                f"blob {len(contents)}\0".encode() + contents
            ).hexdigest()
            if git_sha != entry.blob_id:
                raise ValueError(f"dataset Git blob mismatch: {name}")
        files[name] = {
            "path": str(path),
            "digest": digest,
            "git_blob_id": entry.blob_id,
            "lfs_sha256": lfs_sha,
        }
    return {
        "schema": "vlaforge.openpi_aloha_download/1",
        "dataset": ALOHA_DATASET,
        "revision": ALOHA_REVISION,
        "files": files,
    }


def _verified_download_files(report: dict) -> dict[str, Path]:
    if (
        report.get("schema") != "vlaforge.openpi_aloha_download/1"
        or report.get("dataset") != ALOHA_DATASET
        or report.get("revision") != ALOHA_REVISION
        or set(report.get("files", {}))
        != {ALOHA_EPISODE, "meta/info.json", "meta/tasks.jsonl"}
    ):
        raise ValueError("a complete pinned real ALOHA download report is required")
    paths = {}
    for name, entry in report["files"].items():
        path = Path(entry["path"])
        if file_digest(path) != entry["digest"]:
            raise ValueError(f"dataset input file changed: {name}")
        paths[name] = path
    episode = report["files"][ALOHA_EPISODE]["digest"]
    if (
        episode["sha256"] != ALOHA_EPISODE_SHA256
        or episode["size_bytes"] != ALOHA_EPISODE_BYTES
    ):
        raise ValueError("input is not the pinned real ALOHA episode")
    return paths


def pack_aloha_frame(
    *,
    download_report: str | Path,
    output_dir: str | Path,
    frame_index: int = 0,
    noise_seed: int = 20260906,
    action_horizon: int = 50,
    action_dim: int = 32,
) -> dict:
    """Preserve an actual recorded frame; only the saved Gaussian noise is generated."""
    import numpy as np
    from PIL import Image
    import pyarrow.parquet as pq

    if frame_index < 0 or action_horizon < 1 or action_dim < 1:
        raise ValueError("frame and static action dimensions must be valid")
    report_path = Path(download_report)
    report = json.loads(report_path.read_text())
    paths = _verified_download_files(report)
    info = json.loads(paths["meta/info.json"].read_text())
    if info["robot_type"] != "aloha" or info["features"]["observation.state"][
        "shape"
    ] != [14]:
        raise ValueError("the canonical dataset ALOHA state schema changed")
    rows = pq.ParquetFile(paths[ALOHA_EPISODE]).iter_batches(batch_size=1)
    row = None
    for index, batch in enumerate(rows):
        if index == frame_index:
            row = batch.to_pylist()[0]
            break
    if row is None or row["episode_index"] != 0 or row["frame_index"] != frame_index:
        raise ValueError("requested frame is absent or episode/frame identity changed")
    tasks = {
        task["task_index"]: task["task"]
        for line in paths["meta/tasks.jsonl"].read_text().splitlines()
        for task in (json.loads(line),)
    }
    values = {
        "state": np.asarray(row["observation.state"], dtype=np.float32),
        "prompt": np.asarray(tasks[row["task_index"]]),
    }
    image_hashes = {}
    for name, image in row.items():
        if not name.startswith("observation.images."):
            continue
        camera = name.removeprefix("observation.images.")
        if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
            raise ValueError("canonical episode must contain embedded image bytes")
        encoded = image["bytes"]
        decoded = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"))
        values[f"images/{camera}"] = decoded.transpose(2, 0, 1).copy()
        image_hashes[camera] = hashlib.sha256(encoded).hexdigest()
    if "images/cam_high" not in values:
        raise ValueError("real base camera image is missing")
    generator = np.random.Generator(np.random.PCG64(noise_seed))
    values["noise"] = generator.standard_normal(
        (action_horizon, action_dim), dtype=np.float32
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "observation.npz", **values)
    manifest = {
        "schema": "vlaforge.openpi_input_pack/1",
        "evidence_level": "real-recorded-robot-observation; saved synthetic Gaussian action noise",
        "dataset": ALOHA_DATASET,
        "revision": ALOHA_REVISION,
        "episode_index": 0,
        "frame_index": frame_index,
        "timestamp": row["timestamp"],
        "input_npz": "observation.npz",
        "input_digest": file_digest(output / "observation.npz"),
        "source_download_report": file_digest(report_path),
        "source_files": report["files"],
        "encoded_camera_sha256": image_hashes,
        "state_semantics": "unchanged observation.state, as used by pinned LeRobotAlohaDataConfig; no joint reorder",
        "noise": {
            "algorithm": "numpy.PCG64.standard_normal",
            "seed": noise_seed,
            "dtype": "float32",
            "shape": [action_horizon, action_dim],
        },
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
            for name, value in values.items()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_openpi_input_pack(manifest_path: str | Path, *, source_root: str | Path):
    """Restore the official slash-flattened observation tree with no pickle data."""
    import numpy as np

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "vlaforge.openpi_input_pack/1":
        raise ValueError("an OpenPI input-pack manifest is required")
    path = (manifest_path.parent / manifest["input_npz"]).resolve(strict=True)
    if not path.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("input pack escapes its manifest directory")
    if file_digest(path) != manifest["input_digest"]:
        raise ValueError("input pack digest mismatch")
    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != set(manifest["arrays"]) or "noise" not in bundle:
            raise ValueError("input pack array set mismatch")
        arrays = {name: bundle[name] for name in bundle.files}
    for name, array in arrays.items():
        expected = manifest["arrays"][name]
        actual = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
        if actual != expected:
            raise ValueError(f"input array metadata mismatch: {name}")
        if any(not part for part in name.split("/")) or any(
            other.startswith(name + "/") for other in arrays
        ):
            raise ValueError("ambiguous flattened observation path")
    noise = arrays.pop("noise")
    if (
        noise.dtype != np.float32
        or noise.ndim not in (2, 3)
        or not np.isfinite(noise).all()
    ):
        raise ValueError("saved noise must be a finite float32 chunk")
    import_openpi(source_root)
    transforms = importlib.import_module("openpi.transforms")
    observation = transforms.unflatten_dict(
        {
            name: value.item()
            if value.ndim == 0 and value.dtype.kind in "US"
            else value
            for name, value in arrays.items()
        }
    )
    return observation, noise, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--cache-root", type=Path, required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--download-report", type=Path, required=True)
    pack.add_argument("--output-dir", type=Path, required=True)
    pack.add_argument("--frame-index", type=int, default=0)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = (
        download_aloha_episode(**args)
        if command == "download"
        else pack_aloha_frame(**args)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
