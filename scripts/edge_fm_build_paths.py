from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SUPPORTED_PLATFORMS = ("3060", "a800", "orin", "j6m")
DEFAULT_PLATFORM = "a800"
PLATFORM_BUILD_DIR_NAMES = {
    "3060": "build-3060",
    "a800": "build-a800",
    "orin": "build-orin",
    "j6m": "build-j6m",
}
DEFAULT_PLATFORM_PROBE_ORDER = (
    DEFAULT_PLATFORM,
    *(platform for platform in SUPPORTED_PLATFORMS if platform != DEFAULT_PLATFORM),
)


def _normalize_build_dir(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _normalize_platform(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in SUPPORTED_PLATFORMS:
        return normalized
    return None


def build_dir_name_for_platform(platform: str) -> str:
    normalized = _normalize_platform(platform)
    if normalized is None:
        raise ValueError(f"Unsupported platform: {platform}")
    return PLATFORM_BUILD_DIR_NAMES[normalized]


def _read_cmake_cache(cache_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not cache_path.exists():
        return values

    for line in cache_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line or ":" not in line:
            continue
        key_type, value = line.split("=", 1)
        key, _sep, _type = key_type.partition(":")
        values[key] = value
    return values


def platform_from_build_dir(build_dir: Path) -> str | None:
    cache_path = build_dir / "CMakeCache.txt"
    return _normalize_platform(_read_cmake_cache(cache_path).get("PLATFORM"))


def candidate_build_dirs(project_root: Path, explicit_platform: str | None = None) -> list[Path]:
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

    preferred_platforms: list[str] = []
    explicit = _normalize_platform(explicit_platform)
    if explicit is not None:
        preferred_platforms.append(explicit)

    for env_key in ("EDGE_FM_PLATFORM", "PLATFORM"):
        env_platform = _normalize_platform(os.environ.get(env_key))
        if env_platform is not None and env_platform not in preferred_platforms:
            preferred_platforms.append(env_platform)

    for platform in preferred_platforms:
        add(project_root / build_dir_name_for_platform(platform))

    for platform in DEFAULT_PLATFORM_PROBE_ORDER:
        if platform not in preferred_platforms:
            add(project_root / build_dir_name_for_platform(platform))

    return candidates


def resolve_build_dir(project_root: Path, explicit_platform: str | None = None) -> Path | None:
    for build_dir in candidate_build_dirs(project_root, explicit_platform=explicit_platform):
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
    python_paths = candidate_python_paths(project_root)
    for python_dir in reversed(python_paths):
        python_dir_str = str(python_dir)
        while python_dir_str in sys.path:
            sys.path.remove(python_dir_str)
        sys.path.insert(0, python_dir_str)
    return python_paths


def resolve_platform(project_root: Path, explicit_platform: str | None = None) -> str:
    explicit = _normalize_platform(explicit_platform)
    if explicit is not None:
        return explicit

    for env_key in ("EDGE_FM_PLATFORM", "PLATFORM"):
        env_value = _normalize_platform(os.environ.get(env_key))
        if env_value is not None:
            return env_value

    env_build_dir = os.environ.get("EDGE_FM_BUILD_DIR", "").strip()
    if env_build_dir:
        platform = platform_from_build_dir(_normalize_build_dir(project_root, env_build_dir))
        if platform is not None:
            return platform

    build_dir = resolve_build_dir(project_root)
    if build_dir is not None:
        platform = platform_from_build_dir(build_dir)
        if platform is not None:
            return platform

    return DEFAULT_PLATFORM


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve edge-fm build directories and platform names.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--print-build-dir", action="store_true", help="Print the resolved build directory.")
    group.add_argument("--print-platform", action="store_true", help="Print the resolved platform name.")
    group.add_argument("--print-python-paths", action="store_true", help="Print candidate built Python paths.")
    parser.add_argument("--project-root", type=Path, default=_default_project_root(), help="Project root directory.")
    parser.add_argument(
        "--explicit-platform",
        default=None,
        help="Prefer this platform's build directory before probing the rest.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when the requested value cannot be resolved.",
    )
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()

    if args.print_build_dir:
        build_dir = resolve_build_dir(project_root, explicit_platform=args.explicit_platform)
        if build_dir is None:
            return 1 if args.strict else 0
        print(build_dir)
        return 0

    if args.print_platform:
        print(resolve_platform(project_root, explicit_platform=args.explicit_platform))
        return 0

    python_paths = candidate_python_paths(project_root)
    if not python_paths:
        return 1 if args.strict else 0
    for path in python_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
