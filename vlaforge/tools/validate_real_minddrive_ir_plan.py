#!/usr/bin/env python3
"""Run real MindDrive Regions through Semantic IR and Plan executors."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

from vlaforge.adapters.minddrive_real import (
    MINDDRIVE_INPUT_TYPES,
    MINDDRIVE_OUTPUT_TYPES,
    MINDDRIVE_STATE_TYPES,
    build_real_minddrive_program,
    load_real_minddrive_model,
    make_minddrive_flash_vision_encoder,
    make_minddrive_torch_initial_state,
)
from vlaforge.compiler import compile_module
from vlaforge.frontend import load_exported_region
from vlaforge.interpreter import (
    InputBinding,
    InputStamp,
    Interpreter,
    InterpreterError,
    TensorView,
)
from vlaforge.plan import PlanExecutor
from vlaforge.validation import normalize_plan_trace_for_runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "position_encoder": (
            root / "perception" / "position_encoder.pt2e"
        ),
        "map_encoder": root / "perception" / "map_encoder.pt2e",
        "detection_encoder": (
            root / "perception" / "detection_encoder.pt2e"
        ),
        "detection_decoder": (
            root / "perception" / "detection_decoder.pt2e"
        ),
        "decision_expert": root / "language" / "decision_expert.pt2e",
        "action_expert": root / "language" / "action_expert.pt2e",
        "trajectory_decoder": (
            root / "trajectory" / "trajectory_decoder.pt2e"
        ),
    }


class _LazyExportedRegion:
    """Load one CUDA export for a call, then release all owned storage."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.loads = 0
        self.calls = 0

    def __call__(self, *arguments: Any) -> Any:
        import torch

        exported = load_exported_region(self.path)
        implementation = exported.module()
        self.loads += 1
        with torch.no_grad():
            result = implementation(*arguments)
        self.calls += 1
        del implementation
        del exported
        gc.collect()
        torch.cuda.empty_cache()
        return result


class _NoGradRegion:
    """Make a source Region executable obey the inference-only ABI."""

    def __init__(self, implementation: Any) -> None:
        self.implementation = implementation
        self.calls = 0

    def __call__(self, *arguments: Any) -> Any:
        import torch

        with torch.no_grad():
            result = self.implementation(*arguments)
        self.calls += 1
        return result


def _load_regions(
    source_root: Path,
    release_root: Path,
    artifact_root: Path,
    *,
    device: str,
) -> tuple[dict[str, Any], list[Any], dict[str, object]]:
    import torch

    model = load_real_minddrive_model(
        source_root,
        release_root,
        device=device,
    )
    vision = _NoGradRegion(make_minddrive_flash_vision_encoder(model))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    regions: dict[str, Any] = {"vision_encoder": vision}
    owners: list[Any] = [vision]
    evidence: dict[str, object] = {
        "vision_encoder": {
            "provider": "flash-attn-static-tensor-region-plugin",
            "source_root": str(source_root),
        }
    }
    for name, path in _artifact_paths(artifact_root).items():
        provider = _LazyExportedRegion(path)
        owners.append(provider)
        regions[name] = provider
        evidence[name] = {
            "provider": "torch-export",
            "loading_policy": "per-call-load-release",
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
    return regions, owners, evidence


def _bindings(
    tensors: dict[str, Any],
    *,
    revision: int | None,
    device: str,
) -> dict[str, InputBinding]:
    result = {}
    for name, payload in MINDDRIVE_INPUT_TYPES:
        tensor = tensors[name].to(device)
        result[name] = InputBinding(
            TensorView(
                tensor,
                tuple(int(item) for item in payload.shape),
                payload.dtype,
                payload.layout,
                device,
                64,
            ),
            InputStamp(revision=revision),
        )
    return result


def _clone_outputs(group: Any) -> dict[str, Any]:
    result = {}
    for name, _ in MINDDRIVE_OUTPUT_TYPES:
        value = group.output(name)
        if hasattr(value, "detach"):
            value = value.detach().cpu().clone()
        result[name] = value
    return result


def _outputs_exact(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, bool]:
    import torch

    result = {}
    for name, _ in MINDDRIVE_OUTPUT_TYPES:
        left_value = left[name]
        right_value = right[name]
        if isinstance(left_value, torch.Tensor):
            result[name] = bool(torch.equal(left_value, right_value))
        else:
            result[name] = left_value == right_value
    return result


def _versions(runtime: Any) -> dict[str, int]:
    return {
        name: runtime.state_store.versions(name)[-1].version
        for name, _ in MINDDRIVE_STATE_TYPES
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--invocation", type=Path, action="append", required=True
    )
    parser.add_argument("--pipeline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    source_root = args.source_root.resolve()
    release_root = args.release_root.resolve()
    artifact_root = args.artifact_root.resolve()
    module = build_real_minddrive_program(device=args.device)
    compilation = compile_module(
        module,
        default_device=args.device,
        state_device=args.device,
    )
    module = compilation.module
    regions, owners, region_evidence = _load_regions(
        source_root,
        release_root,
        artifact_root,
        device=args.device,
    )
    initial_state = make_minddrive_torch_initial_state(
        torch, device=args.device
    )

    def valid(trajectory: Any) -> bool:
        return bool(torch.isfinite(trajectory).all().item())

    semantic = Interpreter(
        module,
        regions=regions,
        validators={"minddrive_output_contract": valid},
        initial_state=initial_state,
    )
    plan = PlanExecutor(
        compilation.plan,
        module,
        regions=regions,
        validators={"minddrive_output_contract": valid},
        initial_state=initial_state,
    )
    invocation_tensors = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in args.invocation
    ]
    semantic_outputs = []
    plan_outputs = []
    for revision, tensors in enumerate(invocation_tensors):
        bindings = _bindings(
            tensors, revision=revision, device=args.device
        )
        semantic_outputs.append(
            _clone_outputs(
                semantic.run("run", bindings).committed_outputs
            )
        )
    for revision, tensors in enumerate(invocation_tensors):
        bindings = _bindings(
            tensors, revision=revision, device=args.device
        )
        plan_outputs.append(
            _clone_outputs(
                plan.run("run", bindings).committed_outputs
            )
        )
    output_parity = [
        _outputs_exact(semantic_output, plan_output)
        for semantic_output, plan_output in zip(
            semantic_outputs, plan_outputs, strict=True
        )
    ]
    trace_parity = tuple(
        event.as_tuple()
        for event in normalize_plan_trace_for_runtime(
            semantic.trace, compilation.plan, module
        )
    ) == tuple(
        event.as_tuple()
        for event in normalize_plan_trace_for_runtime(
            plan.trace, compilation.plan, module
        )
    )
    expected_version = len(invocation_tensors)
    semantic_versions = _versions(semantic)
    plan_versions = _versions(plan)

    # A failed validation must not publish outputs or advance state.  Retry
    # uses the same exact input revision and may reuse the derived vision
    # feature while recomputing every state-dependent Region.
    failure_revision = len(invocation_tensors)
    failure_bindings = _bindings(
        invocation_tensors[-1],
        revision=failure_revision,
        device=args.device,
    )
    before_failure_versions = _versions(semantic)
    before_failure_output = _clone_outputs(
        semantic.run(
            "run",
            _bindings(
                invocation_tensors[-1],
                revision=failure_revision - 1,
                device=args.device,
            ),
        ).committed_outputs
    )
    # The preceding same-revision Run is a cache hit and one successful commit.
    versions_after_same_revision = _versions(semantic)
    semantic.validators["minddrive_output_contract"] = (
        lambda _trajectory: False
    )
    failed = False
    try:
        semantic.run("run", failure_bindings)
    except InterpreterError:
        failed = True
    after_failure_versions = _versions(semantic)
    after_failure_output = {
        name: (
            semantic.read_output(name).detach().cpu().clone()
            if hasattr(semantic.read_output(name), "detach")
            else semantic.read_output(name)
        )
        for name, _ in MINDDRIVE_OUTPUT_TYPES
    }
    semantic.validators["minddrive_output_contract"] = valid
    retry = _clone_outputs(
        semantic.run("run", failure_bindings).committed_outputs
    )
    versions_after_retry = _versions(semantic)

    semantic.reset_episode(1)
    reset_versions = _versions(semantic)
    reset_output_absent = False
    try:
        semantic.read_output("trajectory")
    except Exception:
        reset_output_absent = True
    misses_before_missing_revision = semantic.cache.misses
    hits_before_missing_revision = semantic.cache.hits
    semantic.run(
        "run",
        _bindings(
            invocation_tensors[0], revision=None, device=args.device
        ),
    )
    semantic.run(
        "run",
        _bindings(
            invocation_tensors[0], revision=None, device=args.device
        ),
    )
    missing_revision_misses = (
        semantic.cache.misses - misses_before_missing_revision
    )
    missing_revision_hits = (
        semantic.cache.hits - hits_before_missing_revision
    )

    checks = {
        "semantic_plan_named_outputs_exact": all(
            all(frame.values()) for frame in output_parity
        ),
        "semantic_plan_trace_exact": trace_parity,
        "semantic_versions_after_sequence": all(
            value == expected_version
            for value in semantic_versions.values()
        ),
        "plan_versions_after_sequence": all(
            value == expected_version
            for value in plan_versions.values()
        ),
        "validation_failure_raised": failed,
        "validation_failure_versions_unchanged": (
            after_failure_versions == versions_after_same_revision
        ),
        "validation_failure_outputs_unchanged": all(
            _outputs_exact(
                before_failure_output, after_failure_output
            ).values()
        ),
        "retry_committed_once": all(
            versions_after_retry[name]
            == versions_after_same_revision[name] + 1
            for name, _ in MINDDRIVE_STATE_TYPES
        ),
        "retry_outputs_finite": all(
            bool(torch.isfinite(value).all().item())
            if isinstance(value, torch.Tensor)
            and value.is_floating_point()
            else True
            for value in retry.values()
        ),
        "reset_versions_zero": all(
            value == 0 for value in reset_versions.values()
        ),
        "reset_output_absent": reset_output_absent,
        "missing_revision_two_misses": missing_revision_misses == 2,
        "missing_revision_zero_hits": missing_revision_hits == 0,
    }
    report = {
        "schema": "vlaforge.minddrive_real_ir_plan_validation/1",
        "passed": all(checks.values()),
        "evidence_level": "real-L2-semantic-ir-plan",
        "checks": checks,
        "regions": region_evidence,
        "sequence": {
            "invocations": [
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
                for path in args.invocation
            ],
            "semantic_state_versions": semantic_versions,
            "plan_state_versions": plan_versions,
            "output_parity": output_parity,
        },
        "cache": {
            "semantic_hits": semantic.cache.hits,
            "semantic_misses": semantic.cache.misses,
            "plan_hits": plan.cache.hits,
            "plan_misses": plan.cache.misses,
            "missing_revision_delta": {
                "hits": missing_revision_hits,
                "misses": missing_revision_misses,
            },
        },
        "transaction": {
            "before_failure_versions": before_failure_versions,
            "versions_after_same_revision": (
                versions_after_same_revision
            ),
            "after_failure_versions": after_failure_versions,
            "versions_after_retry": versions_after_retry,
        },
        "pipeline_report": {
            "path": str(args.pipeline_report.resolve()),
            "sha256": _sha256(args.pipeline_report),
        },
        "core_op_delta": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del owners
    if not report["passed"]:
        raise ValueError(f"MindDrive IR/Plan validation failed: {report}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
