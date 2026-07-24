"""Command-line inspection, verification, execution, and trace diff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlaforge.adapters import build_openvla_fixture, build_smolvla_fixture
from vlaforge.analysis import verify
from vlaforge.codegen import (
    generate_cpp_session,
    openvla_fixture_regions,
    openvla_fixture_runner_source,
    openvla_fixture_validators,
)
from vlaforge.interpreter import Epoch, InputSample, Interpreter, Trace
from vlaforge.ir.parser import parse_module
from vlaforge.ir.serializer import module_digest
from vlaforge.plan import lower_to_plan, physicalize_plan
from vlaforge.validation import NumericContract, compare_traces


def _load_module(path: str):
    return parse_module(Path(path).read_text(encoding="utf-8"))


def _fixture(name: str):
    if name == "smolvla-fixture":
        return build_smolvla_fixture()
    if name == "openvla-fixture":
        return build_openvla_fixture()
    raise ValueError(f"unknown adapter fixture: {name}")


def _load_ticks(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ticks = []
    for item in data["ticks"]:
        tick = Epoch(**item["tick"])
        inputs = {
            name: InputSample(
                sample["value"],
                Epoch(**sample["epoch"]),
            )
            for name, sample in item["inputs"].items()
        }
        ticks.append((tick, inputs))
    return ticks


def _inspect(args: argparse.Namespace) -> int:
    module = _load_module(args.program)
    data = {
        "name": module.name,
        "schema": module.schema_version,
        "digest": module_digest(module),
        "clocks": [clock.name for clock in module.clocks],
        "inputs": [stream.name for stream in module.inputs],
        "states": [
            {
                "name": state.name,
                "clock": state.version_clock,
                "retention": state.retention,
                "scope": state.scope.value,
                "authoritative": state.authoritative,
            }
            for state in module.states
        ],
        "regions": [region.name for region in module.regions],
        "policies": [policy.name for policy in module.policies],
    }
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    module = _load_module(args.program)
    diagnostics = verify(module, raise_on_error=False)
    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic)
        return 1
    print(
        f"verification passed: module={module.name} "
        f"digest={module_digest(module)}"
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    module = _load_module(args.program)
    fixture = _fixture(args.adapter)
    runtime = Interpreter(
        module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    ticks = (
        _load_ticks(args.inputs)
        if args.inputs
        else [(item.tick, item.inputs) for item in fixture.ticks]
    )
    for tick, inputs in ticks:
        runtime.run_tick(args.policy, tick, inputs)
    runtime.trace.write(args.trace)
    print(
        f"execution passed: ticks={len(ticks)} events={len(runtime.trace.events)} "
        f"trace={args.trace}"
    )
    return 0


def _diff(args: argparse.Namespace) -> int:
    report = compare_traces(
        Trace.read(args.expected),
        Trace.read(args.actual),
        NumericContract(args.atol, args.rtol),
    )
    print(report.format())
    return 0 if report.equal else 1


def _codegen(args: argparse.Namespace) -> int:
    fixture = _fixture(args.adapter)
    if args.adapter != "openvla-fixture":
        raise ValueError(
            "static C++ fixture codegen currently supports openvla-fixture"
        )
    plan = physicalize_plan(lower_to_plan(fixture.module))
    sources = generate_cpp_session(
        plan,
        fixture.module,
        regions=openvla_fixture_regions(),
        validators=openvla_fixture_validators(),
        runner_source=openvla_fixture_runner_source(),
    )
    output = Path(args.output)
    expected_names = {name for name, _ in sources.files}
    if output.exists():
        unexpected = sorted(
            path.name for path in output.iterdir()
            if path.name not in expected_names
        )
        if unexpected:
            raise ValueError(
                "refusing to write generated sources into a directory "
                f"with unrelated entries: {unexpected}"
            )
    sources.write(output)
    print(
        f"C++ generation passed: adapter={args.adapter} "
        f"digest={sources.digest()} output={output}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlaforge")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("program")
    inspect_parser.set_defaults(handler=_inspect)

    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("program")
    verify_parser.set_defaults(handler=_verify)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("program")
    run_parser.add_argument(
        "--adapter",
        required=True,
        choices=("smolvla-fixture", "openvla-fixture"),
    )
    run_parser.add_argument("--inputs")
    run_parser.add_argument("--policy", default="act")
    run_parser.add_argument("--trace", required=True)
    run_parser.set_defaults(handler=_run)

    diff_parser = commands.add_parser("diff")
    diff_parser.add_argument("expected")
    diff_parser.add_argument("actual")
    diff_parser.add_argument("--atol", type=float, default=0.0)
    diff_parser.add_argument("--rtol", type=float, default=0.0)
    diff_parser.set_defaults(handler=_diff)

    codegen_parser = commands.add_parser("codegen")
    codegen_parser.add_argument(
        "--adapter",
        required=True,
        choices=("openvla-fixture",),
    )
    codegen_parser.add_argument("--output", required=True)
    codegen_parser.set_defaults(handler=_codegen)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
