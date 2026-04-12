from __future__ import annotations

import os
import sys
from pathlib import Path


def _normalize_build_dir(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def candidate_build_dirs(project_root: Path) -> list[Path]:
    project_root = project_root.resolve()
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: str | Path) -> None:
        build_dir = _normalize_build_dir(project_root, path)
        if build_dir.is_dir() and build_dir not in seen:
            seen.add(build_dir)
            candidates.append(build_dir)

    env_build_dir = os.environ.get("EDGE_FM_BUILD_DIR", "").strip()
    if env_build_dir:
        add(env_build_dir)

    add(project_root / "build")
    for build_dir in sorted(project_root.glob("build*")):
        add(build_dir)

    return candidates


def resolve_build_dir(project_root: Path) -> Path | None:
    for build_dir in candidate_build_dirs(project_root):
        if (
            (build_dir / "python").is_dir()
            or (build_dir / "install" / "python").is_dir()
            or (build_dir / "lib").is_dir()
            or (build_dir / "install" / "lib").is_dir()
        ):
            return build_dir
    return None


def candidate_python_paths(project_root: Path) -> list[Path]:
    paths: list[Path] = []
    for build_dir in candidate_build_dirs(project_root):
        for python_dir in (build_dir / "python", build_dir / "install" / "python"):
            if python_dir.is_dir():
                paths.append(python_dir)
    return paths


def prepend_built_python_paths(project_root: Path) -> list[Path]:
    inserted: list[Path] = []
    for python_dir in candidate_python_paths(project_root):
        python_dir_str = str(python_dir)
        if python_dir_str not in sys.path:
            sys.path.insert(0, python_dir_str)
        inserted.append(python_dir)
    return inserted
