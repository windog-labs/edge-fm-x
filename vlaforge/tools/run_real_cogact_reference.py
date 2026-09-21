"""Run full official CogACT actions with a labelled public dependency candidate."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--openvla-source", type=Path, required=True)
    parser.add_argument("--dlimp-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vision-assets", type=Path, required=True)
    parser.add_argument("--public-llm-source", type=Path, required=True)
    parser.add_argument("--public-llm-lock", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--diagnostic-max-position-embeddings", type=int, choices=(4096,))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for path in (args.official_source, args.openvla_source, args.dlimp_source):
        sys.path.insert(0, str(path.resolve()))
    import numpy as np
    import torch
    from vlaforge.adapters.cogact.cogact_real import (
        file_sha256,
        load_verified_candidate,
        prepare_public_llm_candidate,
        record_official_scheduler,
        restore_rng,
        rng_snapshot,
    )

    report = {"schema": "vlaforge.cogact.official-reference-candidate/1", "status": "preparing",
              "strict_original_meta_config": False, "evidence_kind": "real-weight-official-code-public-dependency-candidate",
              "paper_fidelity_gate": "not_verified", "no_python_deployment": False,
              "task_success_evaluated": False, "pid": os.getpid(), "command": sys.argv,
              "python": sys.version, "runs": [], "versions": {name: importlib.metadata.version(name) for name in
                  ("torch", "torchvision", "transformers", "timm", "tokenizers", "sentencepiece", "numpy")}}
    started = time.monotonic()

    def save() -> None:
        report["elapsed_s"] = time.monotonic() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        llm_root = args.output / "llm-candidate"
        report["dependency"] = prepare_public_llm_candidate(
            args.public_llm_source, args.public_llm_lock, llm_root,
            diagnostic_max_position_embeddings=args.diagnostic_max_position_embeddings,
        )
        if args.prepare_only:
            report["status"] = "dependency_prepared_not_executed"
            return
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        torch.set_num_threads(8)
        report["status"] = "loading_verified_weights"
        save()
        model, report["weights"] = load_verified_candidate(args.checkpoint, llm_root, args.vision_assets)
        model = model.to("cuda:0").eval()
        model.action_model.create_ddim(ddim_step=10)
        scheduler = model.action_model.ddim_diffusion
        report["scheduler"] = {"name": type(scheduler).__name__, "steps": scheduler.num_timesteps,
                               "original_num_steps": scheduler.original_num_steps, "timestep_map": scheduler.timestep_map,
                               "cfg_scale": 1.5, "eta": 0.0, "clip_denoised": False,
                               "rng_draw_semantics": "one randn initial sample plus 10 randn_like DDIM draws, including eta=0",
                               "model_mean_type": str(scheduler.model_mean_type), "model_var_type": str(scheduler.model_var_type)}
        report["hardware"] = {"device": str(torch.cuda.get_device_name(0)), "cuda": torch.version.cuda,
                               "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                               "nvidia_smi": subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=True).stdout}
        report["precision"] = {"weight_dtypes": sorted({str(value.dtype) for value in model.parameters()}),
                                "vlm_autocast_enabled": model.vlm.enable_mixed_precision_training,
                                "vlm_autocast_dtype": str(model.vlm.llm_backbone.half_precision_dtype),
                                "action_net_dtype": str(model.action_model.net.z_embedder.uncondition.dtype),
                                "float32_matmul_precision": torch.get_float32_matmul_precision()}
        inputs = json.loads((args.input / "input.json").read_text())
        if inputs["schema"] != "vlaforge.cogact.public-observation/1" or inputs["simulation"]:
            raise ValueError("actual locked robot observation required")
        if file_sha256(args.input / "observation.png") != inputs["image_sha256"]:
            raise ValueError("robot observation bytes changed")
        from PIL import Image

        image = Image.open(args.input / "observation.png").convert("RGB")
        report["input"] = inputs
        kwargs = {"unnorm_key": inputs["unnorm_key"], "cfg_scale": 1.5, "use_ddim": True, "num_ddim_steps": 10}
        report["status"] = "running_official_candidate"
        save()
        for seed in args.seed or [42, 43]:
            run_dir = args.output / f"seed-{seed}"
            run_dir.mkdir()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            before = rng_snapshot()
            plain_native, plain_normalized = model.predict_action(image, inputs["instruction"], **kwargs)
            plain_after = rng_snapshot()
            restore_rng(before)
            with record_official_scheduler(model) as trace:
                native, normalized = model.predict_action(image, inputs["instruction"], **kwargs)
            after = rng_snapshot()
            if not (np.array_equal(native, plain_native) and np.array_equal(normalized, plain_normalized)):
                raise ValueError("observation hooks changed actual official output")
            if not torch.equal(plain_after["torch_cpu"], after["torch_cpu"]) or not all(
                    torch.equal(first, second) for first, second in zip(plain_after["torch_cuda"], after["torch_cuda"], strict=True)):
                raise ValueError("observation hooks changed Torch RNG consumption")
            if (plain_after["python"] != after["python"]
                    or not np.array_equal(plain_after["numpy"][1], after["numpy"][1])
                    or plain_after["numpy"][0] != after["numpy"][0]
                    or plain_after["numpy"][2:] != after["numpy"][2:]):
                raise ValueError("observation hooks changed host RNG consumption")
            if native.shape != (16, 7) or normalized.shape != (16, 7) or not (np.isfinite(native).all() and np.isfinite(normalized).all()):
                raise ValueError("official complete action output is invalid")
            if len(trace["steps"]) != 10 or len(trace["cfg_calls"]) != 10 or len(trace["random_draws"]) != 11:
                raise ValueError("observed scheduler did not match exact expected RNG/step contract")
            raw = trace["steps"][-1]["output"]["sample"][0].numpy()
            if not np.isfinite(raw).all():
                raise ValueError("postprocessing must not hide a non-finite raw diffusion result")
            np.save(run_dir / "native-actions.npy", native)
            np.save(run_dir / "normalized-actions.npy", normalized)
            np.save(run_dir / "plain-native-actions.npy", plain_native)
            np.save(run_dir / "plain-normalized-actions.npy", plain_normalized)
            np.save(run_dir / "raw-diffusion-actions.npy", raw)
            trace.update(rng_before=before, rng_after=after, plain_rng_after=plain_after)
            torch.save(trace, run_dir / "scheduler-trace.pt")
            np.savez(run_dir / "scheduler-arrays.npz", **{key: value for key, value in vars(scheduler).items() if isinstance(value, np.ndarray)})
            report["runs"].append({"seed": seed, "shape": [16, 7], "all_finite": True,
                                   "raw_diffusion_all_finite": True,
                                   "instrumented_vs_plain_bitwise_equal": True, "rng_consumption_equal": True,
                                   "observed_original_timesteps": [item["timestep"].tolist() for item in trace["cfg_calls"]],
                                   "actual_rotary_position_ranges": [[int(value.min()), int(value.max())] for value in trace["rotary_positions"]],
                                   "native_actions": native.tolist(), "normalized_actions": normalized.tolist(),
                                   "files": {path.name: file_sha256(path) for path in run_dir.iterdir()}})
            save()
        report["status"] = "reference_candidate_verified"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
