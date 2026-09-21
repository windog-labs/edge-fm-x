"""Extract a pinned Fractal/Google Robot observation for CogACT reference runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    import av
    import numpy as np
    import pyarrow.parquet as pq

    lock = json.loads(args.lock.read_text())
    if lock["status"] != "verified" or lock["repo"] != "IPEC-COMMUNITY/fractal20220817_data_lerobot":
        raise ValueError("requires verified public Fractal dataset lock")
    for item in lock["files"]:
        path = args.source / item["path"]
        if not item["verified"] or path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ValueError(f"source file differs from locked identity: {path}")
    info = json.loads((args.source / "meta/info.json").read_text())
    if info["robot_type"] != "google_robot" or info["fps"] != 3:
        raise ValueError("unexpected robot/fps contract")
    rows = pq.read_table(args.source / "data/chunk-000/episode_000000.parquet").to_pylist()
    selected = [row for row in rows if row["frame_index"] == args.frame]
    if len(selected) != 1 or selected[0]["episode_index"] != 0:
        raise ValueError("requested frame is not unique in episode 0")
    row = selected[0]
    tasks = [json.loads(line) for line in (args.source / "meta/tasks.jsonl").read_text().splitlines()]
    instruction = next(item["task"] for item in tasks if item["task_index"] == row["task_index"])
    video = args.source / "videos/chunk-000/observation.images.image/episode_000000.mp4"
    with av.open(str(video)) as container:
        frames = [frame for index, frame in enumerate(container.decode(video=0)) if index == args.frame]
    if len(frames) != 1:
        raise ValueError("video has no requested observation frame")
    frame = frames[0]
    timestamp = float(frame.pts * frame.time_base)
    if abs(timestamp - row["timestamp"]) > 1e-5:
        raise ValueError("video/parquet timestamp mismatch")
    rgb = frame.to_ndarray(format="rgb24")
    if rgb.shape != (256, 320, 3) or rgb.dtype != np.uint8:
        raise ValueError("unexpected actual image contract")
    args.output.mkdir(parents=True, exist_ok=False)
    from PIL import Image

    Image.fromarray(rgb).save(args.output / "observation.png")
    np.save(args.output / "observation.rgb.npy", rgb)
    record = {
        "schema": "vlaforge.cogact.public-observation/1", "dataset": lock["repo"],
        "revision": lock["revision"], "source_lock_sha256": sha256(args.lock),
        "robot_type": "google_robot", "simulation": False,
        "dataset_lineage": "IPEC public LeRobot conversion of Fractal/RT-1 recorded robot data",
        "image_boundary": "RGB decoded from dataset MP4; not original sensor/JPEG bytes",
        "episode": 0, "frame": args.frame, "timestamp_s": timestamp,
        "instruction": instruction, "unnorm_key": "fractal20220817_data",
        "recorded_row": row, "image_shape": list(rgb.shape),
        "image_sha256": sha256(args.output / "observation.png"),
        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "versions": {"av": av.__version__, "numpy": np.__version__},
        "selection": "episode 0, explicit frame index, selected before model execution",
        "task_success_evaluation": False,
    }
    (args.output / "input.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
