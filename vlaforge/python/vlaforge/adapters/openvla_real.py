"""Real-checkpoint OpenVLA adapter for the VLA-focused reference IR.

OpenVLA's reference ``predict_action`` path is stateless across control ticks:
it deterministically generates a bounded set of action tokens, converts them to
continuous values, validates the action, and publishes it.  The Hugging Face
generation implementation remains inside one pure ``TensorRegion``.  This is
intentional: KV-cache details are local implementation state, not VLA session
state, and therefore do not belong in the business IR.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vlaforge.frontend.builder import ModuleBuilder
from vlaforge.interpreter import Epoch, InputSample, Interpreter
from vlaforge.ir import ops
from vlaforge.ir.attrs import FreshnessConstraint
from vlaforge.ir.program import (
    Block,
    ClockDomain,
    InputStream,
    Policy,
    TensorRegion,
    Value,
)
from vlaforge.ir.types import EpochType, ScalarType, TensorType


OPAQUE = ScalarType("opaque")


@dataclass(frozen=True, slots=True)
class RealOpenVLAConfig:
    checkpoint_path: Path
    revision: str
    device: str = "cuda:0"
    unnorm_key: str = "bridge_orig"
    instruction: str = "pick up the block"
    tolerance: float = 1e-6
    load_in_4bit: bool = True


@dataclass(frozen=True, slots=True)
class RealOpenVLAEvidence:
    schema: str
    evidence_kind: str
    checkpoint_path: str
    checkpoint_revision: str
    checkpoint_shards: tuple[dict[str, Any], ...]
    torch_version: str
    transformers_version: str
    tokenizers_version: str
    timm_version: str
    bitsandbytes_version: str | None
    device: str
    gpu_name: str | None
    quantization: str
    unnorm_key: str
    action_shape: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    eager_action: tuple[float, ...]
    ir_action: tuple[float, ...]
    eager_seconds: float
    ir_seconds: float
    peak_memory_mb: float | None
    token_ids_equal: bool
    action_max_abs_error: float
    trace_events: int
    passed: bool

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def build_real_openvla_action_program(*, action_dim: int) -> Any:
    """Build the minimal stateless OpenVLA policy program."""

    tokens = TensorType((1, action_dim), "i64")
    action = TensorType((action_dim,), "f64")
    builder = ModuleBuilder("openvla_real_action")
    builder.add_clock(ClockDomain("observation", period_ns=50_000_000))
    builder.add_clock(ClockDomain("control", period_ns=50_000_000))
    builder.add_input(
        InputStream(
            "model_inputs",
            OPAQUE,
            "observation",
            FreshnessConstraint(max_age_ns=60_000_000),
        )
    )
    builder.add_region(
        TensorRegion(
            "generate_action_tokens",
            (Value("inputs_arg", OPAQUE),),
            (tokens,),
        )
    )
    builder.add_region(
        TensorRegion(
            "detokenize_action",
            (Value("tokens_arg", tokens),),
            (action,),
        )
    )
    body = Block.of(
        (
            ops.sample_input(
                "inputs_value",
                "inputs_epoch",
                OPAQUE,
                "model_inputs",
                "observation",
                max_age_ns=60_000_000,
            ),
            ops.transaction_begin("txn", "tick"),
            ops.invoke(
                ("action_tokens",),
                (tokens,),
                "generate_action_tokens",
                ("inputs_value",),
            ),
            ops.invoke(
                ("decoded_action",),
                (action,),
                "detokenize_action",
                ("action_tokens",),
            ),
            ops.validate("action_valid", "decoded_action", "finite_action"),
            ops.action_create(
                "pending_action",
                "decoded_action",
                action,
                "tick",
            ),
            ops.transaction_commit(
                "committed_action",
                action,
                "txn",
                "pending_action",
                "action_valid",
            ),
            ops.action_publish("committed_action"),
            ops.return_values("committed_action"),
        )
    )
    builder.add_policy(
        Policy(
            "act",
            "control",
            body,
            inputs=(Value("tick", EpochType("control")),),
            metadata={
                "action_generation": "real_autoregressive_discrete",
                "persistent_state": "none",
            },
        )
    )
    return builder.build()


def run_real_openvla(
    config: RealOpenVLAConfig,
    *,
    trace_path: str | Path | None = None,
) -> RealOpenVLAEvidence:
    """Run eager and IR paths through the same pinned OpenVLA checkpoint."""

    import numpy as np
    import timm
    import tokenizers
    import torch
    import transformers
    from PIL import Image
    from transformers import (
        AutoModelForVision2Seq,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    checkpoint = config.checkpoint_path.resolve()
    _validate_checkpoint(checkpoint)
    if not torch.cuda.is_available() and config.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
    )
    load_arguments: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    bitsandbytes_version: str | None = None
    if config.load_in_4bit:
        import bitsandbytes

        bitsandbytes_version = bitsandbytes.__version__
        load_arguments.update(
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
                "device_map": "auto",
            }
        )

    if config.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = AutoModelForVision2Seq.from_pretrained(checkpoint, **load_arguments)
    if not config.load_in_4bit:
        model = model.to(config.device)
    model.eval()

    action_dim = int(model.get_action_dim(config.unnorm_key))
    image = _deterministic_image(np, Image)
    prompt = (
        "In: What action should the robot take to "
        f"{config.instruction.strip().lower()}?\nOut:"
    )
    model_inputs = processor(prompt, image)
    model_inputs = model_inputs.to(config.device, dtype=torch.bfloat16)

    captured: list[Any] = []
    original_generate = model.generate

    def capture_generate(*args: Any, **kwargs: Any) -> Any:
        result = original_generate(*args, **kwargs)
        captured.append(result.detach().clone())
        return result

    model.generate = capture_generate
    started = time.perf_counter()
    with torch.inference_mode():
        eager_action = model.predict_action(
            **model_inputs,
            unnorm_key=config.unnorm_key,
            do_sample=False,
            use_cache=True,
        )
    _synchronize(torch, config.device)
    eager_seconds = time.perf_counter() - started
    model.generate = original_generate
    if len(captured) != 1:
        raise RuntimeError(
            f"OpenVLA predict_action called generate {len(captured)} times"
        )
    eager_tokens = captured[0][:, -action_dim:]

    ir_token_runs: list[Any] = []

    def generate_action_tokens(inputs: Any) -> Any:
        input_ids = inputs["input_ids"]
        if not torch.all(input_ids[:, -1] == 29871):
            empty_token = torch.tensor(
                [[29871]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat((input_ids, empty_token), dim=1)
        generation_inputs = dict(inputs)
        generation_inputs["input_ids"] = input_ids
        generated = original_generate(
            **generation_inputs,
            max_new_tokens=action_dim,
            do_sample=False,
            use_cache=True,
        )
        action_tokens = generated[:, -action_dim:]
        ir_token_runs.append(action_tokens.detach().clone())
        return action_tokens

    def detokenize_action(action_tokens: Any) -> Any:
        token_ids = action_tokens[0].detach().cpu().numpy()
        discretized = model.vocab_size - token_ids
        discretized = np.clip(
            discretized - 1,
            a_min=0,
            a_max=model.bin_centers.shape[0] - 1,
        )
        normalized = model.bin_centers[discretized]
        stats = model.get_action_stats(config.unnorm_key)
        mask = stats.get(
            "mask",
            np.ones_like(stats["q01"], dtype=bool),
        )
        high = np.asarray(stats["q99"])
        low = np.asarray(stats["q01"])
        return np.where(
            mask,
            0.5 * (normalized + 1) * (high - low) + low,
            normalized,
        )

    module = build_real_openvla_action_program(action_dim=action_dim)
    runtime = Interpreter(
        module,
        regions={
            "generate_action_tokens": generate_action_tokens,
            "detokenize_action": detokenize_action,
        },
        validators={"finite_action": lambda action: bool(np.isfinite(action).all())},
    )
    tick = Epoch("control", 0, 0, 0)
    observation_epoch = Epoch("observation", 0, 0, 0)
    started = time.perf_counter()
    with torch.inference_mode():
        result = runtime.run_tick(
            "act",
            tick,
            {"model_inputs": InputSample(model_inputs, observation_epoch)},
        )
    _synchronize(torch, config.device)
    ir_seconds = time.perf_counter() - started
    ir_action = result.returns[0].value

    if len(ir_token_runs) != 1:
        raise RuntimeError(
            f"OpenVLA IR called generate_action_tokens {len(ir_token_runs)} times"
        )
    token_ids_equal = bool(torch.equal(eager_tokens, ir_token_runs[0]))
    action_error = float(np.max(np.abs(eager_action - ir_action)))
    if trace_path is not None:
        runtime.trace.write(trace_path)

    peak_memory = (
        float(torch.cuda.max_memory_allocated() / 2**20)
        if config.device.startswith("cuda")
        else None
    )
    shard_evidence = tuple(_checkpoint_shards(checkpoint))
    passed = token_ids_equal and action_error <= config.tolerance
    eager_values = tuple(float(item) for item in np.asarray(eager_action).tolist())
    ir_values = tuple(float(item) for item in np.asarray(ir_action).tolist())
    return RealOpenVLAEvidence(
        schema="vlaforge.real_model_evidence/0.1",
        evidence_kind="real_checkpoint",
        checkpoint_path=str(checkpoint),
        checkpoint_revision=config.revision,
        checkpoint_shards=shard_evidence,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        tokenizers_version=tokenizers.__version__,
        timm_version=timm.__version__,
        bitsandbytes_version=bitsandbytes_version,
        device=config.device,
        gpu_name=(
            torch.cuda.get_device_name()
            if config.device.startswith("cuda")
            else None
        ),
        quantization="bitsandbytes-nf4" if config.load_in_4bit else "bfloat16",
        unnorm_key=config.unnorm_key,
        action_shape=tuple(int(dim) for dim in np.asarray(eager_action).shape),
        generated_token_ids=tuple(int(item) for item in eager_tokens[0].cpu()),
        eager_action=eager_values,
        ir_action=ir_values,
        eager_seconds=eager_seconds,
        ir_seconds=ir_seconds,
        peak_memory_mb=peak_memory,
        token_ids_equal=token_ids_equal,
        action_max_abs_error=action_error,
        trace_events=len(runtime.trace.events),
        passed=passed,
    )


def _validate_checkpoint(path: Path) -> None:
    required = (
        "config.json",
        "model.safetensors.index.json",
        "modeling_prismatic.py",
        "processing_prismatic.py",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing OpenVLA checkpoint files: {missing}")
    shards = sorted(path.glob("model-*.safetensors"))
    if len(shards) != 3 or any(item.stat().st_size < 1_000_000 for item in shards):
        raise FileNotFoundError(
            "OpenVLA checkpoint requires three materialized safetensor shards"
        )


def _checkpoint_shards(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "file": shard.name,
            "bytes": shard.stat().st_size,
            "sha256": _sha256(shard),
        }
        for shard in sorted(path.glob("model-*.safetensors"))
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deterministic_image(np: Any, Image: Any) -> Any:
    grid_y, grid_x = np.indices((224, 224), dtype=np.uint16)
    pixels = np.stack(
        (
            grid_x % 256,
            grid_y % 256,
            (grid_x + 2 * grid_y) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
