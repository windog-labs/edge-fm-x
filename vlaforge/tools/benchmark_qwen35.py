"""Model-independent Transformers multimodal baseline, not VLAForge deployment.

The historical filename is retained; checkpoints are selected through Auto APIs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from cogact_gpu_monitor import child_handshake, run_monitored


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def device_snapshot(gpu):
    raw = subprocess.check_output([
        "nvidia-smi", "-i", gpu,
        "--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = list(csv.reader(raw.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or rows[0][0] != gpu or not gpu.startswith("GPU-"):
        raise ValueError("requested GPU UUID is not present on this host")
    return dict(zip(("uuid", "name", "driver", "memory_mib", "utilization"), rows[0]))


def decode_rate(count, first_ns, end_ns):
    if count < 1 or first_ns >= end_ns:
        raise ValueError("invalid token timing interval")
    return (count - 1) * 1e9 / (end_ns - first_ns) if count > 1 else None


class TokenStream:
    """Timestamp CPU-available generated tokens, excluding the prompt callback."""

    def __init__(self, clock=time.perf_counter_ns):
        self.clock = clock
        self.prompt = True
        self.first_ns = None
        self.tokens = []

    def put(self, value):
        if self.prompt:
            self.prompt = False
            return
        tokens = value.detach().cpu().reshape(-1).tolist()
        if self.first_ns is None:
            self.first_ns = self.clock()
        self.tokens.extend(tokens)

    def end(self):
        pass


def validate_settings(args):
    if min(args.workers, args.measured, args.new_tokens) < 1 or args.warmup < 0:
        raise ValueError("invalid measurement counts")
    if args.mode == "formal" and (args.workers < 5 or args.warmup < 128 or args.measured < 1024):
        raise ValueError("formal protocol requires at least 5 workers and 128+1024 calls")


def check_tokens(actual, reference):
    import numpy as np

    if actual.dtype != np.int64 or actual.shape != reference.shape or not np.array_equal(actual, reference):
        raise ValueError("generated token identity or length differs from the untimed reference")


def asset_hashes(model):
    return {str(p.relative_to(model)): sha(p) for p in sorted(model.rglob("*"))
            if p.is_file() and ".cache" not in p.parts}


def verify_protocol(protocol):
    if sha(protocol["image"]) != protocol["image_sha256"]:
        raise ValueError("image identity changed")
    if asset_hashes(Path(protocol["model"])) != protocol["model_assets"]:
        raise ValueError("checkpoint identity changed")
    for path, expected in protocol["source_files"].items():
        if sha(path) != expected:
            raise ValueError("benchmark source changed")


def loading_metadata(info):
    return {key: sorted(value) if isinstance(value, set) else value for key, value in info.items()}


def profile_stages(model, inputs, generation, vision_path, reference):
    import torch

    events = {"forward": [], "vision": []}
    handles = []

    def pre(name):
        def hook(module, values):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events[name].append([event])
        return hook

    def post(name):
        def hook(module, values, result):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events[name][-1].append(event)
        return hook

    try:
        for name, module in (("forward", model), ("vision", model.get_submodule(vision_path))):
            handles.extend([module.register_forward_pre_hook(pre(name)), module.register_forward_hook(post(name))])
        with torch.inference_mode():
            diagnostic = model.generate(**inputs, **generation)[0, inputs["input_ids"].shape[-1]:].cpu().numpy()
        torch.cuda.synchronize()
        check_tokens(diagnostic, reference)
        if len(events["vision"]) != 1 or len(events["forward"]) != len(reference):
            raise ValueError("diagnostic did not cover vision, prefill and cached decode")
        return {k: [a.elapsed_time(b) for a, b in v] for k, v in events.items()}
    finally:
        for handle in handles:
            handle.remove()


def run_worker(args):
    child_handshake()  # NVML/container PID handshake precedes Torch and model work.
    import importlib.metadata
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    protocol_path = args.output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol)
    output = args.output / f"worker-{args.worker:02d}"
    output.mkdir(exist_ok=False)
    record = {"schema": "vlaforge.multimodal_baseline_worker/1", "status": "running",
              "pid": os.getpid(), "worker": args.worker, "protocol_sha256": sha(protocol_path),
              "mode": protocol["mode"], "backend": protocol["backend"],
              "gpu": device_snapshot(protocol["gpu_uuid"]), "no_python_deployment": False}
    write(output / "execution.json", record)
    try:
        torch.manual_seed(0)
        torch.set_num_threads(2)
        record["environment"] = {"python": sys.version, "torch": torch.__version__,
            "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                         if d.metadata.get("Name")}, "threads": torch.get_num_threads()}
        image_bytes = Path(protocol["image"]).read_bytes()
        start = time.perf_counter_ns()
        processor = AutoProcessor.from_pretrained(protocol["model"], local_files_only=True)
        model, loading_info = AutoModelForMultimodalLM.from_pretrained(
            protocol["model"], local_files_only=True, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, output_loading_info=True,
        )
        model.eval()
        if loading_info.get("missing_keys") or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
            raise ValueError(f"incomplete model loading: {loading_info}")
        torch.cuda.synchronize()
        record.update(initialization_ns=time.perf_counter_ns() - start, loading_info=loading_metadata(loading_info),
                      generation_config=json.loads(model.generation_config.to_json_string()))
        write(output / "execution.json", record)

        def prepare():
            with Image.open(io.BytesIO(image_bytes)) as source_image:
                image = source_image.convert("RGB")
            text = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"},
                  {"type": "text", "text": protocol["prompt"]}]}],
                add_generation_prompt=True, tokenize=False,
            )
            cpu = processor(text=[text], images=[image], return_tensors="pt")
            if "pixel_values" not in cpu:
                raise ValueError("processor did not produce visual inputs")
            return cpu, {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in cpu.items()}

        generation = dict(max_new_tokens=protocol["new_tokens"], min_new_tokens=protocol["new_tokens"],
                          do_sample=False, use_cache=True)
        cpu, inputs = prepare()
        record["input_tensors"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                   for k, v in cpu.items() if hasattr(v, "shape")}
        np.savez(output / "prepared_inputs.npz", **{k: v.numpy() for k, v in cpu.items()})
        with torch.inference_mode():
            reference = model.generate(**inputs, **generation)[0, inputs["input_ids"].shape[-1]:].cpu().numpy()
        if reference.shape != (protocol["new_tokens"],):
            raise ValueError("fixed-length reference generation failed")
        np.save(output / "reference_tokens.npy", reference, allow_pickle=False)
        record["reference_text"] = processor.decode(reference, skip_special_tokens=True)
        count = protocol["warmup"] + protocol["measured"]
        raw = np.lib.format.open_memmap(output / "generated_tokens.npy", mode="w+", dtype=np.int64,
                                        shape=(count, protocol["new_tokens"]))
        torch.cuda.reset_peak_memory_stats()
        columns = ["run", "phase", "prepare_h2d_ns", "ttft_ns", "resident_generate_ns",
                   "input_to_tokens_ns", "decode_tokens_per_s", "new_tokens"]
        with (output / "samples.csv").open("x", newline="") as stream, torch.inference_mode():
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for index in range(count):
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                _, inputs = prepare()
                torch.cuda.synchronize()
                prepared = time.perf_counter_ns()
                sink = TokenStream()
                result = model.generate(**inputs, **generation, streamer=sink)
                tokens = result[0, inputs["input_ids"].shape[-1]:].cpu().numpy()
                torch.cuda.synchronize()
                end = time.perf_counter_ns()
                check_tokens(tokens, reference)
                if sink.first_ns is None or sink.tokens != tokens.tolist():
                    raise ValueError("streamer token evidence differs from the generated output")
                raw[index] = tokens
                raw.flush()
                writer.writerow({"run": index, "phase": "warmup" if index < protocol["warmup"] else "measured",
                    "prepare_h2d_ns": prepared - start, "ttft_ns": sink.first_ns - start,
                    "resident_generate_ns": end - prepared, "input_to_tokens_ns": end - start,
                    "decode_tokens_per_s": decode_rate(len(tokens), sink.first_ns, end), "new_tokens": len(tokens)})
                stream.flush()
        record.update(all_calls_equal_untimed_reference=True, completed_calls=count,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if protocol.get("vision_module"):
            # Diagnostic hooks are never installed in the measured loop.
            record["diagnostic_stage_cuda_ms"] = profile_stages(
                model, inputs, generation, protocol["vision_module"], reference)
        record["imported_source_files"] = {str(Path(m.__file__).resolve()): sha(m.__file__)
            for m in tuple(sys.modules.values()) if getattr(m, "__file__", None)
            and str(m.__file__).endswith(".py") and Path(m.__file__).is_file()}
        record["numerical_flags"] = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark}
        verify_protocol(protocol)
        record["files"] = {p.name: sha(p) for p in output.iterdir()
                           if p.is_file() and p.name != "execution.json"}
        record.update(status="passed", finished_ns=time.time_ns(),
                      fidelity_scope="same official backend repeatability; no optimized-backend parity")
        write(output / "execution.json", record)
        return 0
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        write(output / "execution.json", record)
        raise


def run_controller(args):
    args.output.mkdir(parents=True, exist_ok=False)
    validate_settings(args)
    here = Path(__file__).resolve()
    protocol = {"schema": "vlaforge.multimodal_baseline_protocol/1", "mode": args.mode,
        "backend": "transformers-python-bf16", "no_python_deployment": False,
        "model": str(args.model.resolve(strict=True)), "model_assets": asset_hashes(args.model),
        "image": str(args.image.resolve(strict=True)), "image_sha256": sha(args.image), "prompt": args.prompt,
        "gpu_uuid": args.gpu, "workers": args.workers, "warmup": args.warmup, "measured": args.measured,
        "new_tokens": args.new_tokens, "vision_module": args.vision_module,
        "generation_policy": "greedy with min_new_tokens=max_new_tokens; fixed-length EOS suppression",
        "source_files": {str(p): sha(p) for p in (here, here.with_name("cogact_gpu_monitor.py"))},
        "boundary": "encoded image bytes in host memory -> preprocessing/H2D/vision/prefill/decode/CPU tokens",
        "excluded": ["checkpoint/processor initialization", "image file IO", "validation/log IO", "text detokenization"],
        "timing_includes": "synchronous CPU token streaming; first token excluded from decode numerator"}
    write(args.output / "protocol.json", protocol)
    controller = {"schema": "vlaforge.multimodal_baseline_campaign/1", "status": "running",
                  "pid": os.getpid(), "mode": args.mode, "no_python_deployment": False,
                  "protocol_sha256": sha(args.output / "protocol.json"), "runs": []}

    def save():
        write(args.output / "campaign.json", controller)

    save()
    try:
        for worker in range(args.workers):
            before = device_snapshot(args.gpu)
            command = [sys.executable, str(here), "--worker", str(worker), "--output", str(args.output)]
            entry = {"worker": worker, "command": command, "device": before, "start_ns": time.time_ns(),
                     "disk_free_bytes": shutil.disk_usage(args.output).free}
            controller["runs"].append(entry)
            save()
            result = run_monitored(command, args.output / f"monitor-{worker:02d}", gpu=args.gpu,
                environment={"CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONDONTWRITEBYTECODE": "1",
                             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            entry.update(exit_code=result.returncode, end_ns=time.time_ns())
            path = args.output / f"worker-{worker:02d}/execution.json"
            data = json.loads(path.read_text())
            entry.update(report_sha256=sha(path), status=data["status"],
                         monitor_sha256=sha(args.output / f"monitor-{worker:02d}/monitor.json"))
            if result.returncode or data["status"] != "passed" or data["protocol_sha256"] != controller["protocol_sha256"]:
                raise RuntimeError("worker failed or protocol binding differs")
            save()
        verify_protocol(protocol)
        controller.update(status="passed", finished_ns=time.time_ns(), independent_audit_complete=False)
        save()
        return 0
    except BaseException as error:
        controller.update(status="failed", error=repr(error))
        save()
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="Describe this image in one short sentence.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--measured", type=int, default=1024)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--vision-module", help="Vision module path for a separate untimed stage diagnostic")
    parser.add_argument("--mode", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.worker is not None:
        if "COGACT_GPU_OWNER_FOLDER" not in os.environ:
            parser.error("worker requires the monitored controller handshake")
        return run_worker(args)
    if args.model is None or args.image is None or args.gpu is None:
        parser.error("controller requires --model, --image, --gpu")
    validate_settings(args)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
