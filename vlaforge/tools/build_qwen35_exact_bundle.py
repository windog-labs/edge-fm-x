"""Package a validated exact Qwen3.5 TorchScript artifact as a VLAForge bundle."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch
from vlaforge.codegen.session_runner import render_resident_tensor_runner
from vlaforge.deployment import (
    ArtifactDiagnostic,
    ArtifactIdentity,
    ArtifactKind,
    DiagnosticSeverity,
    EffectAudit,
    RegionArtifactContract,
    ValueContract,
    WorkspaceContract,
    build_artifact_compile_bundle,
)
from vlaforge.deployment.capabilities import torchscript_backend_capability
from vlaforge.frontend import InvocationProgram
from vlaforge.ir import ops
from vlaforge.ir.program import (
    Block,
    InputPort,
    Invocation,
    Module,
    OutputPort,
    TensorRegion,
    Value,
)
from vlaforge.ir.serializer import io_schema_digest
from vlaforge.ir.types import TensorType


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


def build_module(
    *,
    prompt_length: int,
    new_tokens: int,
    pixel_shape: tuple[int, int],
    vocab_size: int,
    device: str,
) -> tuple[Module, TensorRegion]:
    input_ids_type = TensorType((1, prompt_length), "i64")
    attention_type = TensorType((1, prompt_length), "i64")
    pixels_type = TensorType(pixel_shape, "f32")
    logits_type = TensorType((1, 1, vocab_size), "bf16")
    tokens_type = TensorType((1, new_tokens), "i64")
    accepted_type = TensorType((1,), "bool")
    region = TensorRegion(
        "qwen_generate_exact",
        (
            Value("arg_0", input_ids_type),
            Value("arg_1", attention_type),
            Value("arg_2", pixels_type),
        ),
        (tokens_type, logits_type),
    )
    inputs = tuple(
        InputPort(name, type_, input_id=index, device=device)
        for index, (name, type_) in enumerate((
            ("input_ids", input_ids_type),
            ("attention_mask", attention_type),
            ("pixel_values", pixels_type),
            ("accepted", accepted_type),
        ))
    )
    outputs = tuple(
        OutputPort(name, type_, output_id=index, device=device, group="generation")
        for index, (name, type_) in enumerate((
            ("tokens", tokens_type),
            ("logits", logits_type),
        ))
    )
    body = Block.of(
        (
            ops.input_read("input_ids_value", "input_ids_revision", input_ids_type, "input_ids"),
            ops.input_read("attention_value", "attention_revision", attention_type, "attention_mask"),
            ops.input_read("pixels_value", "pixels_revision", pixels_type, "pixel_values"),
            ops.input_read("accepted_value", "accepted_revision", accepted_type, "accepted"),
            ops.transaction_begin("txn"),
            ops.invoke(
                ("tokens_value", "logits_value"),
                (tokens_type, logits_type),
                region.name,
                ("input_ids_value", "attention_value", "pixels_value"),
            ),
            ops.validate("tokens_valid", "accepted_value", "vlaforge_predicate_true"),
            ops.output_create("pending_tokens", "tokens_value", tokens_type, "tokens"),
            ops.output_create("pending_logits", "logits_value", logits_type, "logits"),
            ops.output_group(
                "pending_outputs",
                "generation",
                (
                    ("pending_tokens", __import__(
                        "vlaforge.ir.types", fromlist=["PendingOutputType"]
                    ).PendingOutputType("tokens", tokens_type)),
                    ("pending_logits", __import__(
                        "vlaforge.ir.types", fromlist=["PendingOutputType"]
                    ).PendingOutputType("logits", logits_type)),
                ),
            ),
            ops.transaction_commit(
                "committed_outputs",
                (
                    __import__("vlaforge.ir.types", fromlist=["PendingOutputType"]).PendingOutputType(
                        "tokens", tokens_type
                    ),
                    __import__("vlaforge.ir.types", fromlist=["PendingOutputType"]).PendingOutputType(
                        "logits", logits_type
                    ),
                ),
                "generation",
                "txn",
                "pending_outputs",
                "tokens_valid",
            ),
            ops.return_values("committed_outputs"),
        )
    )
    module = Module(
        "qwen35_exact_generate",
        inputs,
        outputs,
        (),
        (region,),
        (Invocation("act", body),),
    )
    return module, region


def build_region_output_contracts(
    *,
    new_tokens: int,
    vocab_size: int,
    device: str,
) -> tuple[ValueContract, ValueContract]:
    """Declare the artifact outputs in the saved TorchScript tuple order."""
    return (
        ValueContract.from_ir(
            "output_0", TensorType((1, new_tokens), "i64"), device=device
        ),
        ValueContract.from_ir(
            "output_1", TensorType((1, 1, vocab_size), "bf16"), device=device
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, default=327)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--pixel-height", type=int, default=1200)
    parser.add_argument("--pixel-width", type=int, default=1536)
    parser.add_argument("--vocab-size", type=int, default=248320)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    validation = json.loads(args.validation_report.read_text())
    if validation.get("status") != "passed" or validation.get("all_tokens_equal_official") is not True:
        raise ValueError("trace artifact token validation is not accepted")
    if sha256(args.artifact) != validation["artifact_sha256"]:
        raise ValueError("artifact hash differs from validation report")
    expected_artifact_outputs = [
        {"index": 0, "name": "tokens", "dtype": "i64", "shape": [1, args.new_tokens]},
        {
            "index": 1,
            "name": "logits",
            "dtype": "bf16",
            "shape": [1, 1, args.vocab_size],
        },
    ]
    if validation.get("artifact_outputs") != expected_artifact_outputs:
        raise ValueError("trace artifact output contract differs from bundle contract")
    graph_audit = validation.get("graph_audit")
    if (
        not isinstance(graph_audit, dict)
        or graph_audit.get("python_op") is not False
        or graph_audit.get("random_ops") != []
        or graph_audit.get("external_io_ops") != []
        or validation.get("inputs_unchanged") is not True
    ):
        raise ValueError("trace artifact effect audit is not accepted")
    device = "cuda:0"
    module, region = build_module(
        prompt_length=args.prompt_length,
        new_tokens=args.new_tokens,
        pixel_shape=(args.pixel_height, args.pixel_width),
        vocab_size=args.vocab_size,
        device=device,
    )
    graph = json.loads(json.dumps(graph_audit))
    audit = EffectAudit(
        diagnostics=(
            ArtifactDiagnostic(
                "frontend.torchscript_graph_audit",
                "saved TorchScript graph has no PythonOp, random operator, external I/O, "
                "and leaves caller inputs unchanged",
                severity=DiagnosticSeverity.INFO,
            ),
        )
    )
    region_inputs = (
        ValueContract.from_ir("arg_0", TensorType((1, args.prompt_length), "i64"), device=device),
        ValueContract.from_ir("arg_1", TensorType((1, args.prompt_length), "i64"), device=device),
        ValueContract.from_ir(
            "arg_2", TensorType((args.pixel_height, args.pixel_width), "f32"), device=device
        ),
    )
    region_outputs = build_region_output_contracts(
        new_tokens=args.new_tokens,
        vocab_size=args.vocab_size,
        device=device,
    )
    contract = RegionArtifactContract(
        region_id=0,
        region_name=region.name,
        inputs=region_inputs,
        outputs=region_outputs,
        io_schema_digest=io_schema_digest(module),
        identity=ArtifactIdentity(
            model_name=args.model_id,
            upstream_revision="qwen3.5-native-20260910",
            checkpoint_identity=args.checkpoint_identity,
            graph_sha256=sha256(args.artifact),
        ),
        artifact_kind=ArtifactKind.TORCHSCRIPT_ARCHIVE,
        artifact_path=f"artifacts/{region.name}.pt",
        artifact_sha256=sha256(args.artifact),
        artifact_size_bytes=args.artifact.stat().st_size,
        workspace=WorkspaceContract(device=device),
        capability=torchscript_backend_capability(
            "sm_90", ("bf16", "f32", "i64")
        ),
        effect_audit=audit,
        backend_variant="torchscript-aten/1",
    )
    template = (args.source / "tools/session_benchmark_runner.cpp.in").read_text()
    runner, runner_contract = render_resident_tensor_runner(
        module,
        template,
        outputs=[
            {"name": "tokens", "role": "exact"},
            {"name": "logits", "role": "primary-action"},
        ],
        samples=args.new_tokens,
    )
    bundle = args.output / "bundle"
    build_artifact_compile_bundle(
        module,
        bundle,
        region_artifacts={region.name: contract},
        artifact_sources={region.name: args.artifact},
        validators=InvocationProgram(module, {}).cpp_validators(),
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
        auxiliary_files={"evidence/validation.json": args.validation_report},
    )
    report = {
        "schema": "edgefm.qwen35.native_bundle/1",
        "status": "native_bundle_built",
        "model": args.model_id,
        "checkpoint_identity": args.checkpoint_identity,
        "artifact_sha256": sha256(args.artifact),
        "validation_report_sha256": sha256(args.validation_report),
        "bundle": str(bundle),
        "bundle_manifest_sha256": sha256(bundle / "bundle.json"),
        "graph_audit": graph,
        "runner_contract": runner_contract,
        "python_deployment": False,
    }
    write(args.output / "report.json", report)
    print(json.dumps({
        "status": report["status"],
        "bundle": str(bundle),
        "bundle_manifest_sha256": report["bundle_manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
