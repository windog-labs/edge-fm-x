#!/usr/bin/env python3
"""Audit the generated real OpenVLA TorchScript C++ deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from vlaforge.adapters import build_real_openvla_action_program
from vlaforge.codegen import openvla_spec_from_capture_reports
from vlaforge.interpreter import Epoch, InputSample, Interpreter
from vlaforge.plan import PlanExecutor, lower_to_plan, physicalize_plan
from vlaforge.validation import (
    compare_traces,
    normalize_plan_trace_for_runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--frontend-report", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    reports = {
        name: json.loads(
            (args.capture_dir / f"{name}.capture.json").read_text(
                encoding="utf-8"
            )
        )
        for name in (
            "generate_action_tokens_prefill",
            "generate_action_tokens_decode_step",
            "detokenize_action",
        )
    }
    spec = openvla_spec_from_capture_reports(
        reports["generate_action_tokens_prefill"],
        reports["generate_action_tokens_decode_step"],
        reports["detokenize_action"],
    )
    cpp_seconds = None
    if not args.reuse_existing:
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONHOME": "/definitely/not/a/python/home",
                "PYTHONPATH": "/definitely/not/a/python/path",
            }
        )
        started = time.perf_counter()
        completed = subprocess.run(
            [
                str(args.runner),
                str(args.archive),
                str(args.input_dir),
                str(args.evidence),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        cpp_seconds = time.perf_counter() - started
        args.stdout.write_text(completed.stdout, encoding="utf-8")
    output = args.stdout.read_text(encoding="utf-8")
    token_rows, action_rows, runtime_trace = _parse_stdout(output)
    frontend = json.loads(args.frontend_report.read_text(encoding="utf-8"))
    expected_tokens = tuple(
        int(value)
        for value in _validation_check(
            frontend, "official_token_ids"
        ).split(",")
    )

    inputs = tuple(
        _read_raw_input(
            args.input_dir / f"input_{index}.bin", tensor_spec
        )
        for index, tensor_spec in enumerate(spec.prefill_inputs)
    )
    module = torch.jit.load(str(args.archive), map_location="cpu").eval()
    numerical = []
    token_reference: list[int] = []
    offset = 0
    expected_evidence_size = 0
    with args.evidence.open("rb") as evidence_file:
        mapped = mmap.mmap(
            evidence_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        try:
            with torch.inference_mode():
                prefix = tuple(module.prefill(*inputs))
                offset = _compare_tensor_sequence(
                    mapped,
                    offset,
                    prefix,
                    "prefill",
                    numerical,
                )
                token = torch.argmax(prefix[0], dim=-1, keepdim=True)
                token_reference.append(int(token.item()))
                cache = prefix[1:]
                for step in range(spec.decode_steps):
                    decoded = tuple(module.decode(token, *cache))
                    offset = _compare_tensor_sequence(
                        mapped,
                        offset,
                        decoded,
                        f"decode.{step}",
                        numerical,
                    )
                    token = torch.argmax(
                        decoded[0], dim=-1, keepdim=True
                    )
                    token_reference.append(int(token.item()))
                    cache = decoded[1:]
                action_tokens = torch.tensor(
                    token_reference, dtype=torch.int64
                ).reshape(1, -1)
                action = module.detokenize(action_tokens)
                offset = _compare_tensor_sequence(
                    mapped,
                    offset,
                    (action_tokens, action),
                    "final",
                    numerical,
                )
                for index in range(3):
                    offset = _compare_tensor_sequence(
                        mapped,
                        offset,
                        (action,),
                        f"published.{index}",
                        numerical,
                    )
            expected_evidence_size = offset
        finally:
            mapped.close()
    if offset != args.evidence.stat().st_size:
        raise RuntimeError(
            f"evidence size mismatch: parsed={offset} "
            f"actual={args.evidence.stat().st_size}"
        )

    action_values = tuple(float(value) for value in action.tolist())
    module_ir = build_real_openvla_action_program(
        action_dim=spec.action_dim
    )
    plan = physicalize_plan(lower_to_plan(module_ir))

    def prefill_region(*_: object) -> tuple[tuple[int, ...], int]:
        return (expected_tokens[:1], 1)

    def decode_region(
        carry: tuple[tuple[int, ...], int], step: int
    ) -> tuple[tuple[int, ...], int]:
        values, _ = carry
        return (values + (expected_tokens[int(step) + 1],), int(step) + 2)

    regions = {
        "generate_action_tokens_prefill": prefill_region,
        "generate_action_tokens_decode_step": decode_region,
        "extract_action_tokens": lambda carry: carry[0],
        "detokenize_action": lambda _: action_values,
    }
    validators = {
        "finite_action": lambda values: all(
            value == value and abs(value) != float("inf")
            for value in values
        )
    }
    semantic = Interpreter(
        module_ir, regions=regions, validators=validators
    )
    scheduled = PlanExecutor(
        plan,
        module_ir,
        regions=regions,
        validators=validators,
    )
    for sequence in range(3):
        tick = Epoch("control", sequence, sequence * 50_000_000, 0)
        observation = Epoch(
            "observation", sequence, sequence * 50_000_000, 0
        )
        samples = {
            "image": InputSample(None, observation),
            "instruction_tokens": InputSample(None, observation),
            "instruction_mask": InputSample(None, observation),
        }
        semantic.run_tick("act", tick, samples)
        scheduled.run_tick("act", tick, samples)
    semantic_plan = compare_traces(semantic.trace, scheduled.trace)
    semantic_runtime = tuple(
        item.as_tuple()
        for item in normalize_plan_trace_for_runtime(
            semantic.trace, plan, module_ir
        )
    )
    plan_runtime = tuple(
        item.as_tuple()
        for item in normalize_plan_trace_for_runtime(
            scheduled.trace, plan, module_ir
        )
    )
    linked = subprocess.run(
        ["ldd", str(args.runner)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.lower()
    no_libpython = "libpython" not in linked
    tokens_equal = all(
        tuple(row) == expected_tokens for row in token_rows
    ) and tuple(token_reference) == expected_tokens
    actions_equal = all(
        tuple(row) == action_values for row in action_rows
    )
    transaction_ids = tuple(
        event[4] for event in runtime_trace if event[0] == 0
    )
    passed = bool(
        no_libpython
        and tokens_equal
        and actions_equal
        and all(item["passed"] for item in numerical)
        and semantic_plan.equal
        and semantic_runtime == plan_runtime == tuple(runtime_trace)
        and transaction_ids == (0, 1, 2)
        and not any(event[0] in {1, 2, 3} for event in runtime_trace)
    )
    report = {
        "schema": "vlaforge.real_openvla_cpp_audit/1",
        "model": "OpenVLA",
        "passed": passed,
        "checkpoint_revision": frontend["checkpoint_revision"],
        "checkpoint_digests": frontend["checkpoint_digests"],
        "torch_version": torch.__version__,
        "archive": {
            "path": str(args.archive.resolve()),
            "sha256": _sha256(args.archive),
            "size_bytes": args.archive.stat().st_size,
            "entrypoints": ["prefill", "decode", "detokenize"],
        },
        "runner_sha256": _sha256(args.runner),
        "no_python_environment": True,
        "no_libpython_dependency": no_libpython,
        "cpp_run_seconds": cpp_seconds,
        "token_ids": list(expected_tokens),
        "token_ids_equal": tokens_equal,
        "actions_equal": actions_equal,
        "action": list(action_values),
        "semantic_plan_trace_equal": semantic_plan.equal,
        "plan_cpp_trace_equal": (
            semantic_runtime == plan_runtime == tuple(runtime_trace)
        ),
        "cpp_trace_events": len(runtime_trace),
        "transaction_ids": list(transaction_ids),
        "persistent_state_slots": [],
        "state_commit_events": sum(
            event[0] == 3 for event in runtime_trace
        ),
        "numeric_contract": {
            "absolute_tolerance": 0.0,
            "relative_tolerance": 0.0,
            "reason": (
                "Python and C++ invoke the same CPU TorchScript archive "
                "through LibTorch."
            ),
        },
        "numeric_evidence": numerical,
        "evidence_size_bytes": args.evidence.stat().st_size,
        "expected_evidence_size_bytes": expected_evidence_size,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"model=OpenVLA passed={passed} tokens={tokens_equal} "
        f"trace={len(runtime_trace)} numeric={len(numerical)} "
        f"report={args.report}"
    )
    return 0 if passed else 1


def _compare_tensor_sequence(
    mapped: mmap.mmap,
    offset: int,
    references: tuple[torch.Tensor, ...],
    name: str,
    output: list[dict[str, Any]],
) -> int:
    for index, reference in enumerate(references):
        end = offset + reference.nbytes
        if end > len(mapped):
            raise RuntimeError(f"truncated evidence at offset {offset}")
        view = memoryview(mapped)[offset:end]
        actual = torch.frombuffer(view, dtype=reference.dtype).reshape(
            reference.shape
        )
        if reference.dtype is torch.bool:
            maximum = float(
                torch.ne(reference, actual).to(torch.float32).max().item()
            )
        else:
            maximum = float(
                torch.max(
                    torch.abs(
                        reference.to(torch.float32)
                        - actual.to(torch.float32)
                    )
                ).item()
            )
        passed = bool(torch.equal(reference, actual))
        output.append(
            {
                "name": f"{name}.output.{index}",
                "shape": list(reference.shape),
                "dtype": str(reference.dtype).removeprefix("torch."),
                "maximum_absolute_error": maximum,
                "maximum_relative_error": 0.0 if maximum == 0.0 else None,
                "passed": passed,
            }
        )
        del actual
        view.release()
        offset = end
    return offset


def _read_raw_input(path: Path, spec: Any) -> torch.Tensor:
    raw = bytearray(path.read_bytes())
    dtype = {
        "bool": torch.bool,
        "i32": torch.int32,
        "i64": torch.int64,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "f64": torch.float64,
    }[spec.dtype]
    return torch.frombuffer(raw, dtype=dtype).clone().reshape(spec.shape)


def _parse_stdout(
    text: str,
) -> tuple[list[list[int]], list[list[float]], list[tuple[int, ...]]]:
    tokens: dict[int, list[int]] = {}
    actions: list[list[float]] = []
    trace: list[tuple[int, ...]] = []
    for line in text.splitlines():
        fields = line.split(",")
        if fields[0] == "TOKEN":
            tokens.setdefault(int(fields[1]), []).append(int(fields[3]))
        elif fields[0] == "ACTION":
            actions.append([float(value) for value in fields[2:]])
        elif fields[0] == "TRACE":
            trace.append(tuple(int(value) for value in fields[1:]))
    if sorted(tokens) != [0, 1, 2] or len(actions) != 3 or not trace:
        raise RuntimeError("runner stdout is incomplete")
    return [tokens[index] for index in range(3)], actions, trace


def _validation_check(report: dict[str, Any], name: str) -> str:
    for item in report["validation_checks"]:
        if item["name"] == name:
            return str(item["value"])
    raise KeyError(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
