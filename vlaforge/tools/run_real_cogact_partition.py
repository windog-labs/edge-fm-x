"""Verify real CogACT producer/partition semantics before strict Tensor capture."""

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
                 "public-llm-source", "public-llm-lock", "input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
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
        capture_cogact_partition,
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
    from vlaforge.frontend import load_exported_region
    from vlaforge.interpreter import Interpreter
    from vlaforge.ir.serializer import canonical_json

    tf.config.set_visible_devices([], "GPU")
    torch.set_num_threads(8)
    report = {"schema": "vlaforge.cogact.explicit-rng-partition-candidate/1", "status": "preparing",
              "reference_identity": "official-code-public-dependency-candidate", "original_meta_config_verified": False,
              "autonomous_cpp_rng": False, "no_python_deployment_verified": False, "runs": [],
              "rng_boundary": "external producer performs 11 upstream CUDA global-PRNG draws; IR consumes tape and returns a state receipt"}
    started = time.monotonic()

    def save():
        report["elapsed_s"] = time.monotonic() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def exact(left, right, name):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            error = (left.double() - right.double()).abs().max().item() if left.shape == right.shape else None
            raise ValueError(f"actual partition mismatch {name}: max_abs={error}, shapes={left.shape}/{right.shape}")

    def same_rng(left, right):
        exact(left["torch_cpu"], right["torch_cpu"], "CPU RNG")
        for a, b in zip(left["torch_cuda"], right["torch_cuda"], strict=True):
            exact(a, b, "CUDA RNG")
        if (left["python"] != right["python"] or left["numpy"][0] != right["numpy"][0]
                or left["numpy"][2:] != right["numpy"][2:] or not np.array_equal(left["numpy"][1], right["numpy"][1])):
            raise ValueError("host RNG state differs")

    save()
    try:
        llm_root = args.output / "llm-candidate"
        report["dependency"] = prepare_public_llm_candidate(args.public_llm_source, args.public_llm_lock, llm_root)
        model, report["weights"] = load_verified_candidate(args.checkpoint, llm_root, args.vision_assets)
        model = model.to("cuda:0").eval()
        model.action_model.create_ddim(10)
        input_record = json.loads((args.input / "input.json").read_text())
        if file_sha256(args.input / "observation.png") != input_record["image_sha256"] or input_record["simulation"]:
            raise ValueError("verified actual robot image required")
        image = Image.open(args.input / "observation.png").convert("RGB")
        observation = make_model_inputs(model, image, input_record["instruction"])
        report["input"] = input_record
        report["status"] = "verifying_official_vs_partition"
        save()
        capture_target = None
        for seed in (42, 43):
            folder = args.output / f"seed-{seed}"
            folder.mkdir()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            before = rng_snapshot()
            with record_official_scheduler(model) as official_trace:
                native, normalized = model.predict_action(image, input_record["instruction"],
                    unnorm_key=input_record["unnorm_key"], cfg_scale=1.5, use_ddim=True, num_ddim_steps=10)
            official_after = rng_snapshot()
            restore_rng(before)
            tape = produce_rng_tape(model)
            producer_after = rng_snapshot()
            same_rng(producer_after, official_after)
            exact(tape["initial_noise"].cpu(), official_trace["random_draws"][0]["value"], "initial draw")
            for index in range(10):
                exact(tape["step_noise"][index].cpu(), official_trace["random_draws"][index + 1]["value"], f"draw {index}")
                exact(tape["rng_states"][index + 1].cpu(), official_trace["steps"][index]["rng_before"]["torch_cuda"][0], f"step {index} RNG before")
                exact(tape["rng_states"][index + 2].cpu(), official_trace["steps"][index]["rng_after"]["torch_cuda"][0], f"step {index} RNG after")
            partition = build_cogact_partition(model, {**observation, **tape}, unnorm_key=input_record["unnorm_key"])
            same_rng(producer_after, rng_snapshot())
            steps = []
            with torch.inference_mode(), record_official_scheduler(model) as partition_trace:
                cognition = partition.implementations["prefix"](observation["tokens"], observation["dino"], observation["siglip"])
                exact(cognition.cpu(), official_trace["cfg_calls"][0]["z"][:1], "cognition")
                carried = partition.implementations["initialize"](tape["initial_noise"], tape["rng_states"], tape["rng_before"])
                for index in range(10):
                    carried = partition.implementations["step"](cognition, *carried, tape["step_noise"], tape["rng_states"])
                    exact(carried[0].cpu(), official_trace["steps"][index]["output"]["sample"], f"step {index} sample")
                    steps.append(tuple(value.detach().cpu().clone() for value in carried))
                final = partition.implementations["finish"](*carried, tape["rng_states"], observation["tokens"])
            exact(final[1].cpu(), torch.from_numpy(normalized), "complete normalized actions")
            exact(final[2].cpu(), torch.from_numpy(native), "complete native actions")
            exact(final[3].cpu(), official_after["torch_cuda"][0], "final RNG receipt")
            if not bool(final[-1].all()) or final[-2].item() != 11:
                raise ValueError("partition rejected its full tape/finite contract")
            if partition_trace["random_draws"] or len(partition_trace["cfg_calls"]) != 10:
                raise ValueError("partition has hidden RNG work or missing CFG calls")
            for index, (a, b) in enumerate(zip(official_trace["cfg_calls"], partition_trace["cfg_calls"], strict=True)):
                for name in ("x", "timestep", "z", "output"):
                    exact(a[name], b[name], f"CFG {index} {name}")
            same_rng(producer_after, rng_snapshot())
            executor = Interpreter(partition.program.module, regions=partition.program.regions,
                                   validators=partition.program.validators)
            with torch.inference_mode():
                result = executor.run(inputs=partition.bind_inputs(revision=1))
            for name, value in zip(("raw_action_chunk", "normalized_action_chunk", "native_action_chunk", "rng_after", "draws_consumed"), final[:-1], strict=True):
                actual = result.committed_outputs.output(name)
                exact(actual, value, f"Invocation output {name}")
            rejected = []
            valid_inputs = dict(partition.inputs)
            for revision, bad_case in enumerate(("rng_receipt", "padded_tokens", "nonfinite_noise"), start=2):
                partition.inputs = dict(valid_inputs)
                if bad_case == "rng_receipt":
                    partition.inputs["rng_before"] = valid_inputs["rng_before"].clone()
                    partition.inputs["rng_before"][0] ^= 1
                elif bad_case == "padded_tokens":
                    partition.inputs["tokens"] = valid_inputs["tokens"].clone()
                    partition.inputs["tokens"][0, 1] = model.vlm.llm_backbone.tokenizer.pad_token_id
                else:
                    partition.inputs["initial_noise"] = valid_inputs["initial_noise"].clone()
                    partition.inputs["initial_noise"][0, 0, 0] = float("nan")
                try:
                    with torch.inference_mode():
                        executor.run(inputs=partition.bind_inputs(revision=revision))
                except RuntimeError as error:
                    if "validation" not in str(error):
                        raise
                    rejected.append(bad_case)
                else:
                    raise ValueError(f"invalid input {bad_case} published an output")
            partition.inputs = valid_inputs
            same_rng(producer_after, rng_snapshot())
            torch.save({"official": official_trace, "partition": partition_trace, "partition_steps": steps,
                        "rng_before": before, "official_rng_after": official_after, "producer_rng_after": producer_after}, folder / "traces.pt")
            np.savez(folder / "inputs.npz", **{name: value.detach().cpu().numpy() for name, value in partition.inputs.items()})
            for name, value in zip(("raw", "normalized", "native", "rng_after", "draws_consumed"), final[:-1], strict=True):
                np.save(folder / f"{name}.npy", value.detach().cpu().numpy())
            report["runs"].append({"seed": seed, "all_steps_cfg_tensors_exact": True, "full_actions_exact": True,
                                    "producer_global_rng_after_exact": True, "partition_hidden_rng_draws": 0,
                                    "tape_draws_consumed": 11, "invocation_all_outputs_exact": True,
                                    "invalid_input_transactions_rejected": rejected,
                                    "files": {path.name: file_sha256(path) for path in folder.iterdir()}})
            save()
            if capture_target is None:
                capture_target = partition
        (args.output / "module.json").write_text(canonical_json(capture_target.program.module, indent=2) + "\n")
        report["status"] = "partition_candidate_verified"
        save()
        if args.capture:
            report["status"] = "capturing_verified_partition"
            save()
            captures = capture_cogact_partition(capture_target, args.output / "exports")
            report["captures"] = [item.evidence.to_dict() for item in captures]
            report["export_files"] = {path.name: {"sha256": file_sha256(path), "size": path.stat().st_size}
                                      for path in (args.output / "exports").iterdir()}
            save()
            loaded = {region.name: load_exported_region(args.output / "exports" / f"{region.name}.pt2e").module()
                      for region in capture_target.program.module.regions}
            exported_executor = Interpreter(capture_target.program.module, regions=loaded,
                                            validators=capture_target.program.validators)
            captured_rng = rng_snapshot()
            report["saved_export_full_loop"] = []
            for seed in (42, 43):
                folder = args.output / f"seed-{seed}"
                with np.load(folder / "inputs.npz", allow_pickle=False) as pack:
                    capture_target.inputs = {name: torch.from_numpy(pack[name].copy()).to("cuda:0") for name in pack.files}
                with torch.inference_mode():
                    result = exported_executor.run(inputs=capture_target.bind_inputs(revision=seed))
                for name, filename in (("raw_action_chunk", "raw"), ("normalized_action_chunk", "normalized"),
                                       ("native_action_chunk", "native"), ("rng_after", "rng_after"),
                                       ("draws_consumed", "draws_consumed")):
                    value = result.committed_outputs.output(name).detach().cpu()
                    exact(value, torch.from_numpy(np.load(folder / f"{filename}.npy", allow_pickle=False)), f"saved export seed {seed} {name}")
                    np.save(folder / f"exported-{filename}.npy", value.numpy())
                same_rng(captured_rng, rng_snapshot())
                report["saved_export_full_loop"].append({"seed": seed, "all_five_outputs_bitwise_exact": True,
                                                       "global_rng_unchanged": True})
                save()
            report["status"] = "partition_capture_candidate_verified"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
