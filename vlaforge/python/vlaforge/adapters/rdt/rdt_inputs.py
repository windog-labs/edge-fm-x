"""Safe extraction of complete recorded RDT inputs from a verified first shard."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

from vlaforge.adapters.rdt.rdt_assets import RDT_SOURCE_REVISION, file_identity, verify_file, verify_source


DATASET_REVISION = "a3f4b6624d4fc3a09b538f65081b37401fd8bd84"
FIRST_SHARD = {
    "rfilename": "rdt_data.tar.gz.00", "size": 8589934592,
    "lfs": {"sha256": "660ccdcec29c8e6418915ab2bc3163520e63421797f4d6574721c1b11148b268"},
}
FIRST_EPISODE_MEMBERS = (
    "rdt_data/close_glasses_box/episode_5.hdf5",
    "rdt_data/close_glasses_box/expanded_instruction_gpt-4-turbo.json",
)
# Obtained by complete-member extraction after first-shard LFS verification.
FIRST_EPISODE_SHA256 = (
    "c499dfd948c307cb01ff6e219fbe9bdca3e7827a9babe812bfe046e236539f5a",
    "c8df0fc0f5ec4cc748ce9bd791895ee6aafe99512c4451e0dd99d31e50d66d86",
)
IMAGE_DECODING_PROFILES = ("upstream-opencv-array", "pil-rgb")
CAMERA_ORDER = ("cam_high", "cam_right_wrist", "cam_left_wrist")


def _safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != name.rstrip("/"):
        raise ValueError(f"unsafe tar member: {name}")
    return path


def extract_verified_prefix_members(shard: Path, record: dict, members: tuple[str, ...],
                                    destination: Path) -> dict:
    """Stop after selected complete regular files; never claim archive completion."""
    if not members or len(set(members)) != len(members):
        raise ValueError("select distinct complete archive members")
    for name in members:
        _safe_name(name)
    identity = verify_file(shard, record)
    destination.mkdir(parents=True, exist_ok=False)
    result = {}
    report = {
        "schema": "vlaforge.verified_archive_prefix/1", "source": identity,
        "status": "extracting", "source_shard_verified": True,
        "complete_archive_verified": False, "complete_dataset_verified": False,
        "members": result,
    }
    def save():
        (destination / "extraction.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        with tarfile.open(shard, mode="r|gz") as archive:
            for member in archive:
                path = _safe_name(member.name)
                if member.isdir():
                    continue
                if not member.isreg():
                    raise ValueError(f"links/devices/non-regular tar members rejected: {member.name}")
                if member.name not in members:
                    continue
                if member.name in result or member.size < 1 or member.size > 256 * 1024**2:
                    raise ValueError("duplicate or unexpectedly sized selected tar member")
                output = destination / str(path)
                output.parent.mkdir(parents=True, exist_ok=True)
                if not output.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("selected output escapes extraction directory")
                partial = output.with_name(output.name + ".vf-part")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("selected regular member could not be opened")
                size = 0
                with source, partial.open("xb") as stream:
                    while block := source.read(1024**2):
                        size += len(block)
                        if size > member.size:
                            raise ValueError("selected member exceeds declared size")
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                if size != member.size:
                    raise ValueError("selected member is incomplete in this prefix")
                os.link(partial, output)
                partial.unlink()
                result[member.name] = {"path": str(output.resolve()), **file_identity(output)}
                save()
                if len(result) == len(members):
                    break
        if set(result) != set(members):
            raise ValueError("not every requested member is complete in the verified shard")
        report["status"] = "selected_complete_members_verified"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
    return report


def _recorded_arrays(episode_root: Path, *, step: int, image_decoding: str):
    import h5py
    import numpy as np
    from PIL import Image

    if image_decoding not in IMAGE_DECODING_PROFILES:
        raise ValueError("choose an explicit image channel-decoding profile")
    members = {}
    for name, expected in zip(FIRST_EPISODE_MEMBERS, FIRST_EPISODE_SHA256, strict=True):
        path = episode_root / name
        actual = file_identity(path)
        if actual["sha256"] != expected:
            raise ValueError("recorded episode member differs from the verified shard extraction")
        members[name] = actual
    instruction = json.loads((episode_root / FIRST_EPISODE_MEMBERS[1]).read_text())["instruction"]
    arrays = {}
    image_records = []
    with h5py.File(episode_root / FIRST_EPISODE_MEMBERS[0], "r") as episode:
        if bool(episode.attrs.get("sim", True)) or not bool(episode.attrs.get("compress", False)):
            raise ValueError("selected data must be the real compressed-camera AgileX episode")
        qpos = episode["observations/qpos"]
        if type(step) is not int or not 1 <= step < qpos.shape[0] or qpos.shape[1:] != (14,):
            raise ValueError("select a valid observation with two actual history frames")
        arrays["proprio"] = np.asarray(qpos[step:step + 1]).copy()
        if arrays["proprio"].dtype != np.float32 or not np.isfinite(arrays["proprio"]).all():
            raise ValueError("recorded native proprioception must be finite FP32")
        for history_step in (step - 1, step):
            for camera in CAMERA_ORDER:
                raw = bytes(episode[f"observations/images/{camera}"][history_step])
                if image_decoding == "upstream-opencv-array":
                    import cv2
                    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                else:
                    image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError("recorded JPEG did not decode to an actual three-channel camera frame")
                index = len(image_records)
                arrays[f"image_{index}"] = image.copy()
                arrays[f"jpeg_{index}"] = np.frombuffer(raw, np.uint8).copy()
                image_records.append({"camera": camera, "step": history_step,
                                      "shape": list(image.shape), "jpeg_sha256": hashlib.sha256(raw).hexdigest()})
    return arrays, {"members": members, "instruction": instruction, "images": image_records}


def prepare_recorded_input(*, episode_root: Path, source_root: Path, destination: Path,
                           step: int, seed: int, image_decoding: str) -> dict:
    import numpy as np

    if type(seed) is not int or seed < 0:
        raise ValueError("select an explicit nonnegative reference RNG seed")
    source = verify_source(source_root)
    arrays, provenance = _recorded_arrays(episode_root, step=step, image_decoding=image_decoding)
    frequency = json.loads((source_root / "configs/dataset_control_freq.json").read_text())["agilex"]
    destination.mkdir(parents=True, exist_ok=False)
    np.savez(destination / "inputs.npz", **arrays)
    report = {
        "schema": "vlaforge.rdt_recorded_input/1", "embodiment": "agilex",
        "dataset_repo": "robotics-diffusion-transformer/rdt-ft-data",
        "dataset_revision": DATASET_REVISION, "first_shard": FIRST_SHARD,
        "episode_root": str(episode_root.resolve()), "source_revision": RDT_SOURCE_REVISION,
        "step": step, "seed": seed, "control_frequency": frequency,
        "image_decoding": image_decoding, "camera_slot_order": "history-major then high/right_wrist/left_wrist",
        "camera_slots_are_recorded": [True] * 6,
        "proprio_space": "recorded AgileX native scale before official state formatting",
        "image_channel_semantics": (
            "OpenCV BGR arrays passed unchanged to PIL, matching the published HDF5 decoder"
            if image_decoding == "upstream-opencv-array" else "JPEG decoded by PIL to RGB"
        ),
        "full_archive_verified": False, "full_dataset_verified": False,
        "camera_color_calibration_verified": False, "physical_action_units_verified": False,
        "inputs": file_identity(destination / "inputs.npz"), **provenance,
        "processing_sources": {name: source["files"][name] for name in (
            "scripts/agilex_model.py", "data/hdf5_vla_dataset.py", "configs/dataset_control_freq.json",
        )},
    }
    (destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_recorded_input(pack: Path, *, source_root: Path):
    """Recompute real decoded inputs; manifest booleans alone never admit a pack."""
    import numpy as np

    manifest = json.loads((pack / "manifest.json").read_text())
    if (manifest.get("schema") != "vlaforge.rdt_recorded_input/1"
            or manifest.get("embodiment") != "agilex"
            or manifest.get("dataset_revision") != DATASET_REVISION
            or manifest.get("source_revision") != RDT_SOURCE_REVISION):
        raise ValueError("input source/embodiment profile mismatch")
    for name, expected in manifest["processing_sources"].items():
        if name not in ("scripts/agilex_model.py", "data/hdf5_vla_dataset.py", "configs/dataset_control_freq.json"):
            raise ValueError("unknown input processing source")
        if file_identity(source_root / name) != expected:
            raise ValueError("input processing source content mismatch")
    if len(manifest["processing_sources"]) != 3:
        raise ValueError("input processing source coverage is incomplete")
    arrays, provenance = _recorded_arrays(Path(manifest["episode_root"]), step=manifest["step"],
                                         image_decoding=manifest["image_decoding"])
    frequency = json.loads((source_root / "configs/dataset_control_freq.json").read_text())["agilex"]
    if manifest["instruction"] != provenance["instruction"] or manifest["control_frequency"] != frequency:
        raise ValueError("instruction/control frequency differs from actual recorded source")
    if file_identity(pack / "inputs.npz") != manifest["inputs"]:
        raise ValueError("input tensor package checksum mismatch")
    with np.load(pack / "inputs.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(
            saved[name].dtype != value.dtype or saved[name].shape != value.shape or not np.array_equal(saved[name], value)
            for name, value in arrays.items()
        ):
            raise ValueError("saved input tensors differ from independently decoded real observations")
    return arrays, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("extract", "prepare", "verify"), default="extract")
    parser.add_argument("--first-shard", type=Path)
    parser.add_argument("--episode-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--image-decoding", choices=IMAGE_DECODING_PROFILES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "extract":
        if args.first_shard is None:
            parser.error("extract requires --first-shard")
        report = extract_verified_prefix_members(
            args.first_shard, FIRST_SHARD, FIRST_EPISODE_MEMBERS, args.output,
        )
    elif args.mode == "prepare":
        if any(value is None for value in (args.episode_root, args.source_root, args.step, args.seed, args.image_decoding)):
            parser.error("prepare requires explicit episode, source, step, seed and image decoding")
        report = prepare_recorded_input(
            episode_root=args.episode_root, source_root=args.source_root, destination=args.output,
            step=args.step, seed=args.seed, image_decoding=args.image_decoding,
        )
    else:
        if args.source_root is None:
            parser.error("verify requires --source-root")
        _, report = load_recorded_input(args.output, source_root=args.source_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
