"""Prepare a real CogACT observation series with one verified model load.

The output ``sample-*`` folders use the same explicit feature/RNG contract as
``run_real_cogact_partition.py`` and are intended for a later native Session
benchmark.  This tool does not claim a benchmark or replace the official
reference path; every frame is checked against the instrumented scheduler.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("official-source", "openvla-source", "dlimp-source", "checkpoint", "vision-assets",
                 "public-llm-source", "public-llm-lock", "input-series", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for path in (args.official_source, args.openvla_source, args.dlimp_source):
        sys.path.insert(0, str(path.resolve()))

    import numpy as np
    import tensorflow as tf
    import torch
    from PIL import Image
    from vlaforge.adapters.cogact.cogact_partitioned import (
        build_cogact_partition,
        make_model_inputs,
        produce_rng_tape,
    )
    from vlaforge.adapters.cogact.cogact_real import (
        file_sha256,
        load_verified_candidate,
        prepare_public_llm_candidate,
        record_official_scheduler,
        restore_rng,
        rng_snapshot,
    )

    tf.config.set_visible_devices([], "GPU")
    torch.set_num_threads(8)
    manifest = json.loads((args.input_series / "manifest.json").read_text())
    if manifest.get("schema") != "vlaforge.cogact.real-observation-series/1":
        raise ValueError("unexpected real observation series schema")
    if manifest.get("count", 0) < 2:
        raise ValueError("series must contain at least two real observations")
    report = {
        "schema": "vlaforge.cogact.real-series/1",
        "status": "preparing",
        "reference_identity": "official-code-public-dependency-candidate",
        "series_manifest_sha256": file_sha256(args.input_series / "manifest.json"),
        "seed": args.seed,
        "runs": [],
        "benchmark": False,
        "native_session": False,
        "rng_boundary": "external producer performs 11 upstream CUDA draws per frame; IR consumes explicit tape",
    }
    started = time.monotonic()

    def save() -> None:
        report["elapsed_s"] = time.monotonic() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def exact(left, right, name: str) -> None:
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            error = (left.double() - right.double()).abs().max().item() if left.shape == right.shape else None
            raise ValueError(f"series partition mismatch {name}: max_abs={error}")

    save()
    try:
        llm_root = args.output / "llm-candidate"
        report["dependency"] = prepare_public_llm_candidate(args.public_llm_source, args.public_llm_lock, llm_root)
        model, report["weights"] = load_verified_candidate(args.checkpoint, llm_root, args.vision_assets)
        model = model.to("cuda:0").eval()
        model.action_model.create_ddim(10)
        report["status"] = "running_official_and_partition"
        save()
        for sample_index, item in enumerate(manifest["frames"]):
            input_dir = args.input_series / item["directory"]
            input_record = json.loads((input_dir / "input.json").read_text())
            if input_record["image_sha256"] != item["image_sha256"]:
                raise ValueError("series manifest/input identity mismatch")
            image = Image.open(input_dir / "observation.png").convert("RGB")
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            before = rng_snapshot()
            with record_official_scheduler(model) as official_trace:
                native, normalized = model.predict_action(
                    image, input_record["instruction"], unnorm_key=input_record["unnorm_key"],
                    cfg_scale=1.5, use_ddim=True, num_ddim_steps=10)
            if len(official_trace["steps"]) != 10 or len(official_trace["random_draws"]) != 11:
                raise ValueError("official scheduler did not produce the pinned ten-step/eleven-draw trace")
            official_after = rng_snapshot()
            restore_rng(before)
            tape = produce_rng_tape(model)
            if not torch.equal(tape["rng_states"][-1].cpu(), official_after["torch_cuda"][0]):
                raise ValueError("official and explicit RNG tape final state differ")
            observation = make_model_inputs(model, image, input_record["instruction"])
            partition = build_cogact_partition(model, {**observation, **tape}, unnorm_key=input_record["unnorm_key"])
            with torch.inference_mode(), record_official_scheduler(model) as partition_trace:
                cognition = partition.implementations["prefix"](observation["tokens"], observation["dino"], observation["siglip"])
                carried = partition.implementations["initialize"](tape["initial_noise"], tape["rng_states"], tape["rng_before"])
                for step_index in range(10):
                    carried = partition.implementations["step"](cognition, *carried, tape["step_noise"], tape["rng_states"])
                    exact(carried[0].cpu(), official_trace["steps"][step_index]["output"]["sample"], f"step {step_index} sample")
                final = partition.implementations["finish"](*carried, tape["rng_states"], observation["tokens"])
            if len(partition_trace["cfg_calls"]) != 10 or partition_trace["random_draws"]:
                raise ValueError("partition changed the official ten-call/no-hidden-RNG contract")
            for step_index, (expected, actual) in enumerate(zip(
                    official_trace["cfg_calls"], partition_trace["cfg_calls"], strict=True)):
                for name in ("x", "timestep", "z", "output"):
                    exact(expected[name], actual[name], f"CFG {step_index} {name}")
            exact(final[1].cpu(), torch.from_numpy(normalized), "normalized actions")
            exact(final[2].cpu(), torch.from_numpy(native), "native actions")
            if not bool(final[-1].all()) or final[-2].item() != 11:
                raise ValueError("partition acceptance contract failed")
            sample_dir = args.output / f"sample-{sample_index}"
            sample_dir.mkdir()
            np.savez(sample_dir / "inputs.npz", **{name: value.detach().cpu().numpy() for name, value in partition.inputs.items()})
            np.save(sample_dir / "raw.npy", final[0].detach().cpu().numpy())
            np.save(sample_dir / "normalized.npy", final[1].detach().cpu().numpy())
            np.save(sample_dir / "native.npy", final[2].detach().cpu().numpy())
            (sample_dir / "input.json").write_text(json.dumps(input_record, indent=2) + "\n")
            report["runs"].append({
                "index": sample_index, "frame": item["frame"], "input_directory": item["directory"],
                "full_action_exact": True, "all_finite": True, "explicit_rng_draws": 11,
                "input_sha256": file_sha256(sample_dir / "inputs.npz"),
                "output_sha256": file_sha256(sample_dir / "native.npy"),
            })
            save()
        report["status"] = "real_series_partition_verified"
        report["count"] = len(report["runs"])
        save()
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
