from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from vlaforge import cli


def test_source_provenance_falls_back_to_installed_version(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLAFORGE_SOURCE_REVISION", raising=False)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=128,
            stdout="",
        ),
    )
    monkeypatch.setattr(
        cli.importlib.metadata,
        "version",
        lambda name: "9.8.7",
    )

    assert cli._source_provenance() == (
        "package:vlaforge-9.8.7",
        False,
    )


def test_explicit_source_provenance_is_validated(monkeypatch) -> None:
    monkeypatch.setenv("VLAFORGE_SOURCE_REVISION", "release-123")
    monkeypatch.setenv("VLAFORGE_SOURCE_DIRTY", "1")
    assert cli._source_provenance() == ("release-123", True)

    monkeypatch.setenv("VLAFORGE_SOURCE_DIRTY", "maybe")
    try:
        cli._source_provenance()
    except ValueError as error:
        assert "must be 0 or 1" in str(error)
    else:
        raise AssertionError("invalid dirty provenance was accepted")


def test_runtime_root_accepts_complete_installed_source(
    tmp_path: Path,
) -> None:
    (tmp_path / "runtime").mkdir()
    (tmp_path / "include" / "vlaforge" / "runtime").mkdir(
        parents=True
    )
    (tmp_path / "CMakeLists.txt").write_text("", encoding="utf-8")
    (tmp_path / "runtime" / "state_store.cpp").write_text(
        "",
        encoding="utf-8",
    )
    (tmp_path / "include" / "vlaforge" / "runtime" / "session.h").write_text(
        "",
        encoding="utf-8",
    )

    assert cli._runtime_root(str(tmp_path)) == tmp_path.resolve()
