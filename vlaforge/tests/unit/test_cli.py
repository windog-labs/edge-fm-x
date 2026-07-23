from pathlib import Path

from vlaforge.adapters import build_smolvla_fixture
from vlaforge.cli import main
from vlaforge.ir.printer import print_module


def test_inspect_verify_run_and_diff(tmp_path, capsys):
    program = tmp_path / "program.vla"
    program.write_text(
        print_module(build_smolvla_fixture().module),
        encoding="utf-8",
    )
    assert main(["inspect", str(program)]) == 0
    assert '"prefix_cache"' in capsys.readouterr().out

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

