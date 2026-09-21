"""Build a native Qwen3.5 Session from captured explicit-state Regions."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from vlaforge.adapters.qwen3_5.qwen3_5_state import (
    QwenAdvanceDecode,
    QwenAppendToken,
    QwenMakeDecodeState,
    QwenSeedTokens,
    QwenSelectToken,
)
from vlaforge.codegen.session_runner import render_resident_tensor_runner
from vlaforge.deployment import (
    ArtifactIdentity,
    ArtifactKind,
    EffectAudit,
    RegionArtifactContract,
    ValueContract,
    WorkspaceContract,
    build_artifact_compile_bundle,
)
from vlaforge.deployment.capabilities import torchscript_backend_capability
from vlaforge.deployment.torchscript_export import export_torchscript_region
from vlaforge.frontend import InvocationBuilder, capture_region
from vlaforge.frontend.annotations import RegionSpec
from vlaforge.frontend.tensor_types import tensor_type_from_torch
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.serializer import io_schema_digest
from vlaforge.ir.types import type_from_dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def region_from_evidence(name: str, evidence: dict) -> tuple:
    inputs = tuple(
        Value(row["name"], type_from_dict(row["type"]))
        for row in evidence["inputs"]
    )
    outputs = tuple(type_from_dict(row["type"]) for row in evidence["outputs"])
    return inputs, outputs


def declaration(name: str, inputs: tuple[Value, ...], outputs: tuple):
    def call(*_args):
        raise AssertionError(f"declaration-only Region {name}")

    call.__vlaforge_region__ = RegionSpec(
        name, inputs, outputs, (__import__("vlaforge.ir.attrs", fromlist=["Effect"]).Effect.PURE,)
    )
    return call


def capture_tiny(name: str, module, examples: tuple, output: Path):
    region = __import__(
        "vlaforge.ir.program", fromlist=["TensorRegion"]
    ).TensorRegion(
        name,
        tuple(
            Value(f"arg_{index}", tensor_type_from_torch(value))
            for index, value in enumerate(examples)
        ),
        tuple(tensor_type_from_torch(value) for value in (
            module(*examples) if isinstance(module(*examples), tuple) else (module(*examples),)
        )),
    )
    outcome = capture_region(region, module, examples, strict=False).require_supported()
    archive = output / f"{name}.pt"
    audit = export_torchscript_region(
        outcome.exported_program, archive, validation_cases=(examples,)
    )
    return outcome, archive, audit, region


def artifact_contract(
    module,
    region,
    evidence: dict,
    artifact: Path,
    *,
    model_name: str,
    checkpoint_identity: str,
):
    def values(which):
        return tuple(ValueContract.from_dict(row) for row in evidence[which])

    dtypes = {
        value.type.dtype
        for value in (*values("inputs"), *values("outputs"))
        if hasattr(value.type, "dtype")
    }
    return RegionArtifactContract(
        region_id=next(
            index for index, item in enumerate(module.regions)
            if item.name == region.name
        ),
        region_name=region.name,
        inputs=values("inputs"),
        outputs=values("outputs"),
        io_schema_digest=io_schema_digest(module),
        identity=ArtifactIdentity(
            model_name=model_name,
            upstream_revision="qwen3.5-native-20260910",
            checkpoint_identity=checkpoint_identity,
            graph_sha256=evidence["graph_digest"],
        ),
        artifact_kind=ArtifactKind.TORCHSCRIPT_ARCHIVE,
        artifact_path=f"artifacts/{region.name}.pt",
        artifact_sha256=sha256(artifact),
        artifact_size_bytes=artifact.stat().st_size,
        workspace=WorkspaceContract(device="cuda:0"),
        capability=torchscript_backend_capability(
            "sm_90", tuple(sorted(dtypes))
        ),
        effect_audit=EffectAudit.from_dict(evidence["effect_audit"]),
        backend_variant="torchscript-aten/1",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-report", type=Path, required=True)
    parser.add_argument("--prefill-artifact", type=Path, required=True)
    parser.add_argument("--decode-artifact", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-identity", required=True)
    parser.add_argument("--prompt-length", type=int, default=327)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads(args.region_report.read_text())
    if report.get("status") != "passed":
        raise ValueError("captured Region report is not accepted")
    if sha256(args.prefill_artifact) != report["prefill_archive_sha256"]:
        raise ValueError("prefill artifact hash differs from captured report")
    if sha256(args.decode_artifact) != report["decode_archive_sha256"]:
        raise ValueError("decode artifact hash differs from captured report")
    prefill_inputs, prefill_outputs = region_from_evidence(
        "qwen_prefill", report["prefill"]
    )
    decode_inputs, decode_outputs = region_from_evidence(
        "qwen_decode", report["decode"]
    )
    if len(prefill_inputs) != 3 or len(decode_inputs) != 51:
        raise ValueError("unexpected captured Qwen Region ABI")
    state_types = tuple(decode_outputs[1:])
    state_count = len(state_types)
    logits_type = prefill_outputs[0]
    rope_type = prefill_outputs[1]
    token_type = logits_type if False else __import__(
        "vlaforge.ir.types", fromlist=["TensorType"]
    ).TensorType((1, 1), "i64")
    tokens_type = __import__(
        "vlaforge.ir.types", fromlist=["TensorType"]
    ).TensorType((1, args.new_tokens), "i64")
    step_type = __import__(
        "vlaforge.ir.types", fromlist=["TensorType"]
    ).TensorType((1,), "i64")
    valid_type = __import__(
        "vlaforge.ir.types", fromlist=["TensorType"]
    ).TensorType((1,), "bool")

    tiny_dir = args.output / "artifacts"
    tiny_dir.mkdir()
    device = "cuda:0"
    select = QwenSelectToken().eval()
    seed = QwenSeedTokens(args.new_tokens).eval()
    make_state = QwenMakeDecodeState(args.prompt_length).eval()
    append = QwenAppendToken().eval()
    advance = QwenAdvanceDecode().eval()
    examples = {
        "qwen_select_token": (torch.zeros(tuple(logits_type.shape), dtype=torch.bfloat16, device=device),),
        "qwen_seed_tokens": (torch.zeros((1, 1), dtype=torch.int64, device=device),),
        "qwen_make_decode_state": (torch.zeros((1, 1), dtype=torch.int64, device=device),),
        "qwen_append_token": (
            torch.zeros((1, args.new_tokens), dtype=torch.int64, device=device),
            torch.zeros((1, 1), dtype=torch.int64, device=device),
            torch.ones((1,), dtype=torch.int64, device=device),
        ),
        "qwen_advance_decode": (
            torch.zeros((3, 1, 1), dtype=torch.int64, device=device),
            torch.full((1,), args.prompt_length, dtype=torch.int64, device=device),
            torch.ones((1,), dtype=torch.int64, device=device),
        ),
    }
    tiny = {}
    for name, module in (
        ("qwen_select_token", select),
        ("qwen_seed_tokens", seed),
        ("qwen_make_decode_state", make_state),
        ("qwen_append_token", append),
        ("qwen_advance_decode", advance),
    ):
        outcome, artifact, audit, region = capture_tiny(
            name, module, examples[name], tiny_dir
        )
        tiny[name] = {
            "outcome": outcome,
            "artifact": artifact,
            "audit": audit,
            "region": region,
            "module": module,
        }
    prefill_region = __import__(
        "vlaforge.ir.program", fromlist=["TensorRegion"]
    ).TensorRegion("qwen_prefill", prefill_inputs, prefill_outputs)
    decode_region = __import__(
        "vlaforge.ir.program", fromlist=["TensorRegion"]
    ).TensorRegion("qwen_decode", decode_inputs, decode_outputs)

    declarations = {
        "qwen_prefill": declaration("qwen_prefill", prefill_inputs, prefill_outputs),
        "qwen_decode": declaration("qwen_decode", decode_inputs, decode_outputs),
    }
    for name, item in tiny.items():
        declarations[name] = declaration(name, item["region"].inputs, item["region"].outputs)

    builder = InvocationBuilder(
        "qwen35_native_generate",
        inputs=tuple(
            InputPort(
                name,
                value.type,
                device=report["prefill"]["inputs"][index]["device"],
            )
            for index, (name, value) in enumerate(zip(
                ("input_ids", "attention_mask", "pixel_values"),
                prefill_inputs,
                strict=True,
            ))
        ),
        outputs=(
            OutputPort("logits", logits_type, device=device, group="generation"),
            OutputPort("tokens", tokens_type, device=device, group="generation"),
        ),
    )
    input_ids = builder.input("input_ids")
    attention_mask = builder.input("attention_mask")
    pixel_values = builder.input("pixel_values")
    prefill_result = builder.call(
        declarations["qwen_prefill"], input_ids, attention_mask, pixel_values
    )
    logits, rope_deltas, *states = prefill_result
    (first_token,) = builder.call(declarations["qwen_select_token"], logits)
    tokens, step, valid = builder.call(declarations["qwen_seed_tokens"], first_token)
    position_ids, cache_position = builder.call(
        declarations["qwen_make_decode_state"], rope_deltas
    )
    carried = (
        *states, tokens, first_token, position_ids, cache_position, step, valid, logits
    )

    def body(_index, *values):
        current_states = values[:state_count]
        current_tokens = values[state_count]
        last_token = values[state_count + 1]
        current_position = values[state_count + 2]
        current_cache = values[state_count + 3]
        current_step = values[state_count + 4]
        current_valid = values[state_count + 5]
        current_logits = values[state_count + 6]
        decoded = builder.call(
            declarations["qwen_decode"],
            last_token,
            current_position,
            current_cache,
            *current_states,
        )
        next_logits, *next_states = decoded
        (next_token,) = builder.call(
            declarations["qwen_select_token"], next_logits
        )
        (next_tokens,) = builder.call(
            declarations["qwen_append_token"],
            current_tokens,
            next_token,
            current_step,
        )
        next_position, next_cache, next_step = builder.call(
            declarations["qwen_advance_decode"],
            current_position,
            current_cache,
            current_step,
        )
        return (
            *next_states,
            next_tokens,
            next_token,
            next_position,
            next_cache,
            next_step,
            current_valid,
            next_logits,
        )

    final = builder.iterate(carried, body, steps=args.new_tokens - 1)
    final_tokens = final[state_count]
    final_valid = final[state_count + 5]
    final_logits = final[state_count + 6]
    program = builder.finish(
        {"logits": final_logits, "tokens": final_tokens},
        accepted=final_valid,
    )
    module = program.module
    template = (args.source / "tools/session_benchmark_runner.cpp.in").read_text()
    runner, runner_contract = render_resident_tensor_runner(
        module,
        template,
        outputs=[
            {"name": "logits", "role": "primary-action"},
            {"name": "tokens", "role": "exact"},
        ],
        samples=args.new_tokens,
    )
    artifacts = {
        "qwen_prefill": args.prefill_artifact,
        "qwen_decode": args.decode_artifact,
        **{name: item["artifact"] for name, item in tiny.items()},
    }
    evidence = {
        "qwen_prefill": report["prefill"],
        "qwen_decode": report["decode"],
        **{name: item["outcome"].evidence.to_dict() for name, item in tiny.items()},
    }
    regions = {
        "qwen_prefill": prefill_region,
        "qwen_decode": decode_region,
        **{name: item["region"] for name, item in tiny.items()},
    }
    contracts = {
        name: artifact_contract(
            module,
            regions[name],
            evidence[name],
            artifacts[name],
            model_name=args.model_id,
            checkpoint_identity=args.checkpoint_identity,
        )
        for name in artifacts
    }
    bundle = args.output / "bundle"
    build_artifact_compile_bundle(
        module,
        bundle,
        region_artifacts=contracts,
        artifact_sources=artifacts,
        validators=program.cpp_validators(),
        runner_source=runner,
        runtime_root=args.source,
        cmake_prefix_path=torch.utils.cmake_prefix_path,
        backend_versions={
            "torchscript": torch.__version__,
            "cuda": str(torch.version.cuda),
        },
        profile="verified",
        loop_execution="off",
        source_revision="qwen3.5-native-20260910",
        source_dirty=True,
        default_device=device,
        state_device=device,
        environment={
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "CMAKE_BUILD_PARALLEL_LEVEL": "2",
        },
        auxiliary_files={
            "evidence/region-capture.json": args.region_report,
        },
    )
    output = {
        "schema": "vlaforge.qwen35.native_build/1",
        "status": "native_bundle_built",
        "model": args.model_id,
        "checkpoint_identity": args.checkpoint_identity,
        "region_report_sha256": sha256(args.region_report),
        "regions": {
            name: {
                "artifact": str(artifacts[name]),
                "sha256": sha256(artifacts[name]),
                "graph_digest": evidence[name]["graph_digest"],
            }
            for name in artifacts
        },
        "bundle": str(bundle),
        "bundle_manifest_sha256": sha256(bundle / "bundle.json"),
        "runner_contract": runner_contract,
        "loop_steps": args.new_tokens - 1,
        "state_tensors": state_count,
        "python_deployment": False,
    }
    write(args.output / "report.json", output)
    print(json.dumps({
        "status": output["status"],
        "bundle": str(bundle),
        "bundle_manifest_sha256": output["bundle_manifest_sha256"],
        "state_tensors": state_count,
    }, indent=2))


if __name__ == "__main__":
    main()
