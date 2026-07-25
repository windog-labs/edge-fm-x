from pathlib import Path
import json
import os
import subprocess
import sys

from vlaforge.adapters import build_smolvla_fixture
from vlaforge.cli import main
from vlaforge.deployment import load_bundle_manifest
from vlaforge.ir.printer import print_module


def test_inspect_verify_run_and_diff(tmp_path, capsys):
    program = tmp_path / "program.vla"
    program.write_text(
        print_module(build_smolvla_fixture().module),
        encoding="utf-8",
    )
    assert main(["inspect", str(program)]) == 0
    inspected = capsys.readouterr().out
    assert '"action_queue"' in inspected
    assert '"io_schema_digest"' in inspected
    assert "clock" not in inspected

    assert main(["verify", str(program)]) == 0
    assert "verification passed" in capsys.readouterr().out

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for trace in (first, second):
        assert (
            main(
                [
                    "run",
                    str(program),
                    "--adapter",
                    "smolvla-fixture",
                    "--trace",
                    str(trace),
                ]
            )
            == 0
        )
        assert trace.exists()
        capsys.readouterr()

    assert main(["diff", str(first), str(second)]) == 0
    assert "comparison passed" in capsys.readouterr().out


def test_codegen_cli_is_reproducible(tmp_path, capsys):
    output = tmp_path / "generated"
    command = [
        "codegen",
        "--adapter",
        "openvla-fixture",
        "--output",
        str(output),
    ]
    assert main(command) == 0
    first = {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
    }
    message = capsys.readouterr().out
    assert "C++ generation passed" in message
    assert "session_generated.cpp" in first
    assert "compilation_certificate.json" in first

    assert main(command) == 0
    second = {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
    }
    assert second == first
    capsys.readouterr()


def test_compile_bundle_cli_builds_and_verifies_no_python(
    tmp_path,
    capsys,
) -> None:
    output = tmp_path / "bundle"
    assert (
        main(
            [
                "compile",
                "--adapter",
                "openvla-fixture",
                "--profile",
                "verified",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert "Compile Bundle passed" in capsys.readouterr().out
    assert main(["bundle-verify", str(output / "bundle.json")]) == 0
    assert "verification passed" in capsys.readouterr().out
    manifest = load_bundle_manifest(output / "bundle.json")
    assert any(
        record.role == "compilation_certificate"
        for record in manifest.generated_sources
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHOME": "/definitely/not/a/python/home",
            "PYTHONPATH": "/definitely/not/a/python/path",
        }
    )
    completed = subprocess.run(
        [str(output / "bin" / "vlaforge_generated_runner")],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "OUTPUT,0" in completed.stdout
    input_schema = json.loads(
        (output / "metadata" / "input_schema.json").read_text()
    )
    output_schema = json.loads(
        (output / "metadata" / "output_schema.json").read_text()
    )
    assert input_schema["schema"] == "vlaforge.input_schema/2"
    assert input_schema["io_schema_digest"] == manifest.io_schema_digest
    assert output_schema["schema"] == "vlaforge.output_schema/2"
    assert output_schema["runtime_output"] == "CommittedOutputGroup"
    assert "clock" not in json.dumps(input_schema) + json.dumps(output_schema)
    if sys.platform.startswith("linux"):
        linked = subprocess.run(
            ["ldd", str(output / "bin" / "vlaforge_generated_runner")],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.lower()
        assert "libpython" not in linked


def test_stateful_smolvla_codegen_cli(tmp_path, capsys) -> None:
    output = tmp_path / "smolvla-generated"
    assert (
        main(
            [
                "codegen",
                "--adapter",
                "smolvla-fixture",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    certificate = json.loads(
        (output / "compilation_certificate.json").read_text()
    )
    assert certificate["schema"] == "vlaforge.compilation_certificate/2"
    assert certificate["caches"][0]["identity_fields"] == [
        "episode",
        "model",
        "artifact",
    ]
    assert "C++ generation passed" in capsys.readouterr().out
