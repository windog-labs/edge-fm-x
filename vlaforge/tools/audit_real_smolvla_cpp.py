#!/usr/bin/env python3
"""Audit the real generated SmolVLA C++ AOTI deployment end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from vlaforge.adapters import build_real_smolvla_action_program
from vlaforge.interpreter import Epoch, InputSample, Interpreter
from vlaforge.plan import PlanExecutor, lower_to_plan, physicalize_plan
from vlaforge.validation import (
    compare_traces,
    normalize_plan_trace_for_runtime,
)


@dataclass(frozen=True, slots=True)
class NumericContract:
    absolute_tolerance: float
    relative_tolerance: float


@dataclass(frozen=True, slots=True)
class NumericEvidence:
    name: str
    dtype: str
    shape: tuple[int, ...]
    maximum_absolute_error: float
    maximum_relative_error: float
    contract: NumericContract
    passed: bool


@dataclass(frozen=True, slots=True)
class AuditReport:
    schema: str
    model: str
    frontend_report: str
    checkpoint_revision: str
    checkpoint_digests: tuple[dict[str, str], ...]
    torch_version: str
    cuda_version: str | None
    gpu_name: str
    export_digests: dict[str, str]
    package_digests: dict[str, str]
    runner_digest: str
    no_python_environment: bool
    no_libpython_dependency: bool
    cpp_run_seconds: float
    cpp_trace_events: int
    semantic_trace_events: int
    semantic_plan_trace_equal: bool
    plan_cpp_trace_equal: bool
    reset_trace_valid: bool
    transaction_ids: tuple[int, ...]
    committed_state_versions: dict[str, tuple[int, ...]]
    numerical_evidence: tuple[NumericEvidence, ...]
    action_max_abs_errors: tuple[float, ...]
    evidence_size_bytes: int
    expected_evidence_size_bytes: int
    passed: bool


_REGIONS = ("prepare_prefix", "solver_step", "trim_action_chunk")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--frontend-report", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    exports = {
        name: torch.export.load(args.export_dir / f"{name}.pt2e")
        for name in _REGIONS
    }
    prefix_inputs = tuple(exports["prepare_prefix"].example_inputs[0])
    if not prefix_inputs or not prefix_inputs[0].is_cuda:
        raise RuntimeError("real SmolVLA audit requires CUDA exports")

    started = time.perf_counter()
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHOME": "/definitely/not/a/python/home",
            "PYTHONPATH": "/definitely/not/a/python/path",
        }
    )
    completed = subprocess.run(
        [
            str(args.runner),
            *(str(args.package_dir / f"{name}.pt2") for name in _REGIONS),
            str(args.evidence),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    torch.cuda.synchronize(prefix_inputs[0].device)
    cpp_seconds = time.perf_counter() - started
    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    args.stdout.write_text(completed.stdout, encoding="utf-8")

    linked = subprocess.run(
        ["ldd", str(args.runner)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.lower()
    no_libpython = "libpython" not in linked

    with torch.inference_mode():
        prefix_reference = _as_tuple(
            exports["prepare_prefix"].module()(*prefix_inputs)
        )
        sample = torch.linspace(
            -1.0,
            1.0,
            50 * 32,
            device=prefix_inputs[0].device,
            dtype=torch.float32,
        ).reshape(1, 50, 32)
        solver_reference = []
        for step in range(10):
            timestep = torch.full(
                (1,),
                1.0 - step / 10.0,
                device=sample.device,
                dtype=torch.float32,
            )
            sample = exports["solver_step"].module()(
                prefix_reference[0],
                sample,
                timestep,
                *prefix_reference[1:],
            )
            solver_reference.append(sample.detach().clone())
        action_reference = exports["trim_action_chunk"].module()(sample)

    binary = args.evidence.read_bytes()
    offset = 0
    prefix_actual = []
    for reference in prefix_reference:
        actual, offset = _read_tensor(binary, offset, reference)
        prefix_actual.append(actual)
    solver_actual = []
    for reference in solver_reference:
        actual, offset = _read_tensor(binary, offset, reference)
        solver_actual.append(actual)
    action_actual, offset = _read_tensor(binary, offset, action_reference)
    action_lines, runtime_trace, reset_seen = _parse_stdout(completed.stdout)
    action_tensors = []
    for action in action_lines:
        action_tensors.append(
            torch.tensor(
                action,
                device=action_reference.device,
                dtype=action_reference.dtype,
            ).reshape(1, -1)
        )
    expected_size = offset + sum(tensor.nbytes for tensor in action_tensors)
    binary_actions = []
    for reference in action_tensors:
        actual, offset = _read_tensor(binary, offset, reference)
        binary_actions.append(actual)
    if offset != len(binary):
        raise RuntimeError(
            f"evidence has trailing bytes: parsed={offset} actual={len(binary)}"
        )

    numerical: list[NumericEvidence] = []
    exact = NumericContract(0.0, 0.0)
    prefix_bf16 = NumericContract(0.5, 0.02)
    iterative = NumericContract(0.08, 0.05)
    published = NumericContract(0.012, 0.05)
    for index, (reference, actual) in enumerate(
        zip(prefix_reference, prefix_actual, strict=True)
    ):
        numerical.append(
            _compare(
                f"prepare_prefix.output.{index}",
                reference,
                actual,
                exact if reference.dtype is torch.bool else prefix_bf16,
            )
        )
    for index, (reference, actual) in enumerate(
        zip(solver_reference, solver_actual, strict=True)
    ):
        numerical.append(
            _compare(
                f"solver_step.iteration.{index}",
                reference,
                actual,
                iterative,
            )
        )
    numerical.append(
        _compare(
            "trim_action_chunk",
            action_reference,
            action_actual,
            iterative,
        )
    )
    action_errors = []
    for index, (text_action, binary_action) in enumerate(
        zip(action_tensors, binary_actions, strict=True)
    ):
        reference = action_reference[:, index, :]
        text_result = _compare(
            f"published_action.{index}.stdout",
            reference,
            text_action,
            published,
        )
        binary_result = _compare(
            f"published_action.{index}.binary",
            reference,
            binary_action,
            published,
        )
        numerical.extend((text_result, binary_result))
        action_errors.append(text_result.maximum_absolute_error)

    module = build_real_smolvla_action_program(
        chunk_size=50,
        max_action_dim=32,
        output_action_dim=6,
        num_steps=10,
    )
    plan = physicalize_plan(lower_to_plan(module))
    region_functions = _region_functions(exports, prefix_inputs)
    validators = {
        "finite_action": lambda value: bool(torch.isfinite(value).all())
    }
    initial_state = {
        "action_queue": torch.zeros_like(action_reference),
        "queue_cursor": 50,
    }
    semantic = Interpreter(
        module,
        regions=region_functions,
        validators=validators,
        initial_state=initial_state,
    )
    scheduled = PlanExecutor(
        plan,
        module,
        regions=region_functions,
        validators=validators,
        initial_state=initial_state,
    )
    for sequence in range(3):
        tick = Epoch("control", sequence, sequence * 20_000_000, 0)
        inputs = {
            "batch": InputSample(
                prefix_inputs,
                Epoch("observation", sequence, sequence * 20_000_000, 0),
            ),
            "noise": InputSample(
                torch.linspace(
                    -1.0,
                    1.0,
                    50 * 32,
                    device=prefix_inputs[0].device,
                    dtype=torch.float32,
                ).reshape(1, 50, 32),
                Epoch("observation", sequence, sequence * 20_000_000, 0),
            ),
        }
        semantic_result = semantic.run_tick("act", tick, inputs)
        scheduled_result = scheduled.run_tick("act", tick, inputs)
        if not torch.equal(
            semantic_result.published_actions[0].value,
            scheduled_result.published_actions[0].value,
        ):
            raise RuntimeError("semantic and Plan action values diverged")

    semantic_plan = compare_traces(semantic.trace, scheduled.trace)
    refill_sequences = {
        int(event.tick["sequence"])
        for event in semantic.trace.events
        if event.kind == "region"
        and event.data["region"] == "prepare_prefix"
    }

    def select_conditional_task(
        event: Any, candidates: tuple[Any, ...]
    ) -> Any:
        branch = (
            "/region:0/"
            if int(event.tick["sequence"]) in refill_sequences
            else "/region:1/"
        )
        matching = [
            task for task in candidates if branch in task.source_location
        ]
        if len(matching) != 1:
            raise ValueError(
                f"cannot resolve conditional trace task: {candidates}"
            )
        return matching[0]

    semantic_runtime_trace = tuple(
        event.as_tuple()
        for event in normalize_plan_trace_for_runtime(
            semantic.trace,
            plan,
            module,
            task_selector=select_conditional_task,
        )
    )
    plan_runtime_trace = tuple(
        event.as_tuple()
        for event in normalize_plan_trace_for_runtime(
            scheduled.trace,
            plan,
            module,
            task_selector=select_conditional_task,
        )
    )
    cpp_without_reset = tuple(runtime_trace[:-1]) if reset_seen else ()
    plan_cpp_equal = (
        semantic_runtime_trace == plan_runtime_trace == cpp_without_reset
    )
    reset_valid = bool(
        reset_seen
        and runtime_trace[-1] == (9, 0, 0, 0, 0, 0, 0, 0, 1)
    )

    transaction_ids = tuple(
        event[4] for event in runtime_trace if event[0] == 0
    )
    state_versions = {
        "action_queue": tuple(
            event[3]
            for event in runtime_trace
            if event[0] == 3 and event[2] == 0
        ),
        "queue_cursor": tuple(
            event[3]
            for event in runtime_trace
            if event[0] == 3 and event[2] == 1
        ),
    }
    frontend = json.loads(args.frontend_report.read_text(encoding="utf-8"))
    report = AuditReport(
        schema="vlaforge.real_smolvla_cpp_audit/1",
        model="SmolVLA",
        frontend_report=str(args.frontend_report.resolve()),
        checkpoint_revision=str(frontend["checkpoint_revision"]),
        checkpoint_digests=tuple(frontend["checkpoint_digests"]),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(prefix_inputs[0].device),
        export_digests={
            name: _sha256(args.export_dir / f"{name}.pt2e")
            for name in _REGIONS
        },
        package_digests={
            name: _sha256(args.package_dir / f"{name}.pt2")
            for name in _REGIONS
        },
        runner_digest=_sha256(args.runner),
        no_python_environment=True,
        no_libpython_dependency=no_libpython,
        cpp_run_seconds=cpp_seconds,
        cpp_trace_events=len(runtime_trace),
        semantic_trace_events=len(semantic_runtime_trace),
        semantic_plan_trace_equal=semantic_plan.equal,
        plan_cpp_trace_equal=plan_cpp_equal,
        reset_trace_valid=reset_valid,
        transaction_ids=transaction_ids,
        committed_state_versions=state_versions,
        numerical_evidence=tuple(numerical),
        action_max_abs_errors=tuple(action_errors),
        evidence_size_bytes=len(binary),
        expected_evidence_size_bytes=expected_size,
        passed=bool(
            no_libpython
            and semantic_plan.equal
            and plan_cpp_equal
            and reset_valid
            and transaction_ids == (0, 1, 2)
            and all(values == (1, 2, 3) for values in state_versions.values())
            and all(item.passed for item in numerical)
            and len(binary) == expected_size
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"model={report.model} passed={report.passed} "
        f"trace={report.cpp_trace_events} "
        f"action_max_abs={max(report.action_max_abs_errors):.9g} "
        f"report={args.report}"
    )
    return 0 if report.passed else 1


def _region_functions(
    exports: dict[str, Any],
    prefix_inputs: tuple[torch.Tensor, ...],
) -> dict[str, Callable[..., object]]:
    def prepare_prefix(_: object) -> tuple[torch.Tensor, ...]:
        return _as_tuple(exports["prepare_prefix"].module()(*prefix_inputs))

    def solver_step(
        prefix: tuple[torch.Tensor, ...],
        sample: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        timestep = torch.full(
            (1,),
            1.0 - int(step) / 10.0,
            device=sample.device,
            dtype=torch.float32,
        )
        return exports["solver_step"].module()(
            prefix[0], sample, timestep, *prefix[1:]
        )

    def trim_action_chunk(sample: torch.Tensor) -> torch.Tensor:
        return (
            exports["trim_action_chunk"]
            .module()(sample)
            .detach()
            .clone()
        )

    def queue_select(
        queue: torch.Tensor, cursor: int
    ) -> torch.Tensor:
        return queue[:, int(cursor)].detach().clone()

    return {
        "prepare_prefix": prepare_prefix,
        "solver_step": solver_step,
        "trim_action_chunk": trim_action_chunk,
        "queue_is_empty": lambda cursor: int(cursor) >= 50,
        "queue_select": queue_select,
        "queue_advance": lambda cursor: int(cursor) + 1,
        "queue_zero": lambda: 0,
    }


def _read_tensor(
    data: bytes,
    offset: int,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    end = offset + reference.nbytes
    if end > len(data):
        raise RuntimeError(
            f"truncated evidence at {offset}: need={reference.nbytes}"
        )
    value = torch.frombuffer(
        bytearray(data[offset:end]), dtype=reference.dtype
    ).reshape(reference.shape)
    return value.to(reference.device), end


def _compare(
    name: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    contract: NumericContract,
) -> NumericEvidence:
    if reference.dtype is torch.bool:
        maximum_absolute = float(
            torch.ne(reference, actual).to(torch.float32).max().item()
        )
        maximum_relative = maximum_absolute
    else:
        reference_f32 = reference.to(torch.float32)
        actual_f32 = actual.to(torch.float32)
        difference = torch.abs(reference_f32 - actual_f32)
        maximum_absolute = float(difference.max().item())
        denominator = torch.clamp(torch.abs(reference_f32), min=1e-6)
        maximum_relative = float((difference / denominator).max().item())
    passed = bool(
        torch.allclose(
            reference,
            actual,
            atol=contract.absolute_tolerance,
            rtol=contract.relative_tolerance,
        )
    )
    return NumericEvidence(
        name=name,
        dtype=str(reference.dtype).removeprefix("torch."),
        shape=tuple(reference.shape),
        maximum_absolute_error=maximum_absolute,
        maximum_relative_error=maximum_relative,
        contract=contract,
        passed=passed,
    )


def _parse_stdout(
    output: str,
) -> tuple[list[tuple[float, ...]], list[tuple[int, ...]], bool]:
    actions: list[tuple[float, ...]] = []
    trace: list[tuple[int, ...]] = []
    reset_seen = False
    for line in output.splitlines():
        fields = line.split(",")
        if fields[0] == "ACTION":
            actions.append(tuple(float(value) for value in fields[2:]))
        elif fields[0] == "TRACE":
            trace.append(tuple(int(value) for value in fields[1:]))
        elif fields == ["RESET", "1"]:
            reset_seen = True
    if len(actions) != 3 or not trace:
        raise RuntimeError("runner output is missing actions or trace")
    return actions, trace, reset_seen


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
