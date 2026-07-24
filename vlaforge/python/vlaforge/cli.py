"""Command-line inspection, verification, execution, and trace diff."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

from vlaforge.adapters import build_openvla_fixture, build_smolvla_fixture
from vlaforge.analysis import verify
from vlaforge.codegen import (
    generate_compiled_cpp_session,
    openvla_fixture_regions,
    openvla_fixture_runner_source,
    openvla_fixture_validators,
)
from vlaforge.compiler import CompilerProfile, compile_module
from vlaforge.deployment import build_compile_bundle, load_bundle_manifest
from vlaforge.interpreter import Epoch, InputSample, Interpreter, Trace
from vlaforge.ir.parser import parse_module
from vlaforge.ir.serializer import module_digest
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
    compilation = compile_module(
        fixture.module,
        profile=args.profile,
        allow_test_profile=args.allow_test_profile,
    )
    sources = generate_compiled_cpp_session(
        compilation,
        regions=openvla_fixture_regions(),
        validators=openvla_fixture_validators(),
        runner_source=openvla_fixture_runner_source(),
    )
    output = Path(args.output)
    expected_names = {name for name, _ in sources.files}
    expected_names.add("compilation_certificate.json")
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
    (output / "compilation_certificate.json").write_text(
        compilation.certificate.canonical_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"C++ generation passed: adapter={args.adapter} "
        f"profile={compilation.certificate.profile.value} "
        f"certificate={compilation.certificate.digest()} "
        f"digest={sources.digest()} output={output}"
    )
    return 0


def _compile(args: argparse.Namespace) -> int:
    fixture = _fixture(args.adapter)
    if args.adapter != "openvla-fixture":
        raise ValueError(
            "standalone fixture bundle currently supports openvla-fixture"
        )
    repository = Path(__file__).resolve().parents[3]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    environment = {
        "host": platform.platform(),
        "machine": platform.machine(),
    }
    for name in ("CXX", "CUDA_VISIBLE_DEVICES"):
        if name in os.environ:
            environment[name] = os.environ[name]
    manifest = build_compile_bundle(
        fixture.module,
        args.output,
        regions=openvla_fixture_regions(),
        validators=openvla_fixture_validators(),
        runner_source=openvla_fixture_runner_source(),
        runtime_root=Path(__file__).resolve().parents[2],
        profile=args.profile,
        allow_test_profile=args.allow_test_profile,
        source_revision=revision,
        source_dirty=dirty,
        environment=environment,
    )
    print(
        f"Compile Bundle passed: adapter={args.adapter} "
        f"profile={manifest.compilation_certificate.profile.value} "
        f"digest={manifest.digest()} output={args.output}"
    )
    return 0


def _bundle_verify(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = load_bundle_manifest(manifest_path)
    manifest.verify_files(manifest_path.parent)
    print(
        f"Compile Bundle verification passed: "
        f"digest={manifest.digest()} root={manifest_path.parent}"
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
    codegen_parser.add_argument(
        "--profile",
        default=CompilerProfile.VERIFIED.value,
        choices=(
            CompilerProfile.OFF.value,
            "conservative",
            CompilerProfile.VERIFIED.value,
            "auto",
            CompilerProfile.FORCE_ON.value,
        ),
    )
    codegen_parser.add_argument(
        "--allow-test-profile",
        action="store_true",
        help="allow the force-on profile, which is never for production",
    )
    codegen_parser.set_defaults(handler=_codegen)

    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument(
        "--adapter",
        required=True,
        choices=("openvla-fixture",),
    )
    compile_parser.add_argument("--output", required=True)
    compile_parser.add_argument(
        "--profile",
        default=CompilerProfile.VERIFIED.value,
        choices=(
            CompilerProfile.OFF.value,
            "conservative",
            CompilerProfile.VERIFIED.value,
            "auto",
            CompilerProfile.FORCE_ON.value,
        ),
    )
    compile_parser.add_argument("--allow-test-profile", action="store_true")
    compile_parser.set_defaults(handler=_compile)

    bundle_verify_parser = commands.add_parser("bundle-verify")
    bundle_verify_parser.add_argument("manifest")
    bundle_verify_parser.set_defaults(handler=_bundle_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
