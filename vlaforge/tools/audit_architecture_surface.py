#!/usr/bin/env python3
"""Prove that VLAForge exposes only the passive invocation architecture.

The audit is intentionally static and conservative. It checks the tracked
production surface, the declared Semantic IR opcode set, and every VLAForge
CMake source/subdirectory edge. Negative tests and paper-analysis tools may
mention rejected legacy symbols; those files are reported but are not part of
the production surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.analysis.verifier import ALLOWED_OPS  # noqa: E402


REPORT_SCHEMA = "vlaforge.architecture_surface_audit/1"
PRODUCTION_ROOTS = (
    "vlaforge/python/vlaforge/ir",
    "vlaforge/python/vlaforge/compiler.py",
    "vlaforge/python/vlaforge/plan",
    "vlaforge/python/vlaforge/codegen",
    "vlaforge/python/vlaforge/deployment",
    "vlaforge/runtime",
    "vlaforge/include/vlaforge",
    "vlaforge/backends",
    "vlaforge/CMakeLists.txt",
    "vlaforge/cmake",
)
EXPECTED_OPS = frozenset(
    {
        "vla.input.read",
        "vla.txn.begin",
        "vla.state.read_latest",
        "vla.snapshot.value",
        "vla.invoke",
        "vla.for",
        "vla.if",
        "vla.yield",
        "vla.return",
        "vla.state.stage_write",
        "vla.validate",
        "vla.output.create",
        "vla.output.group",
        "vla.txn.commit",
        "vla.txn.abort",
    }
)
_TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".py",
    ".txt",
}
_FORBIDDEN_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "clock_domain": re.compile(r"\bClockDomain\b|\bclock_domain\b"),
    "run_tick": re.compile(r"\bRunTick\b|\brun_tick\b|vla\.tick"),
    "epoch_expression": re.compile(r"\bEpochExpr\b|\bepoch_expr\b"),
    "state_scope": re.compile(r"\bStateScope\b|\bstate_scope\b"),
    "physical_deadline": re.compile(r"\bdeadline\b", re.IGNORECASE),
    "physical_period": re.compile(r"\bperiod\b", re.IGNORECASE),
    "physical_jitter": re.compile(r"\bjitter\b", re.IGNORECASE),
    "publish_operation": re.compile(
        r"vla\.(?:action\.)?publish|action\.publish"
    ),
    "core_action_queue": re.compile(r"\baction_queue\b|\bqueue_cursor\b"),
    "internal_sleep": re.compile(
        r"\bsleep_for\b|\bnanosleep\b|\busleep\b|\btimerfd\b|"
        r"\bclock_nanosleep\b"
    ),
    "sensor_middleware": re.compile(
        r"\bsensor_?sync\b|\brclcpp\b|\bros::|\bcyber::",
        re.IGNORECASE,
    ),
    "python_runtime": re.compile(
        r"\bPython\.h\b|\bPy_Initialize\b|\blibpython\b"
    ),
}
_MIGRATION_MAP = (
    {
        "old": "ClockDomain + period/deadline/jitter + RunTick",
        "new": "caller-owned scheduling + passive Session::Run",
        "production_status": "removed",
    },
    {
        "old": "EpochExpr.current/next",
        "new": "StateStore allocated commit version",
        "production_status": "removed",
    },
    {
        "old": "action.publish / host I/O",
        "new": "transactional named outputs + ReadOutput",
        "production_status": "removed",
    },
    {
        "old": "core ActionQueue",
        "new": "ChunkedAction Adapter authoritative state",
        "production_status": "adapter-only",
    },
    {
        "old": "sensor synchronization / middleware",
        "new": "caller-prepared TensorView/ScalarValue inputs",
        "production_status": "outside-framework",
    },
    {
        "old": "EdgeFM custom CUDA operators",
        "new": "verified external AOTI/RegionExecutable artifacts",
        "production_status": "not-a-VLAForge-build-dependency",
    },
)


def _git(arguments: Iterable[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_files(paths: Iterable[str]) -> tuple[Path, ...]:
    output = _git(("ls-files", "--", *paths))
    return tuple(
        _REPOSITORY_ROOT / item
        for item in output.splitlines()
        if item
    )


def _text_files(paths: Iterable[str]) -> tuple[Path, ...]:
    return tuple(
        path
        for path in _tracked_files(paths)
        if path.name == "CMakeLists.txt" or path.suffix in _TEXT_SUFFIXES
    )


def _relative(path: Path) -> str:
    return path.relative_to(_REPOSITORY_ROOT).as_posix()


def scan_forbidden_text(
    files: Iterable[Path],
    patterns: Mapping[str, re.Pattern[str]] = _FORBIDDEN_PATTERNS,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for category, pattern in patterns.items():
                match = pattern.search(line)
                if match is not None:
                    findings.append(
                        {
                            "category": category,
                            "path": _relative(path),
                            "line": line_number,
                            "match": match.group(0),
                        }
                    )
    return findings


def _cmake_invocations(
    text: str,
    commands: Iterable[str],
) -> Iterable[tuple[str, tuple[str, ...]]]:
    command_pattern = "|".join(re.escape(command) for command in commands)
    for match in re.finditer(
        rf"(?is)\b({command_pattern})\s*\((.*?)\)",
        text,
    ):
        body = re.sub(r"#.*", "", match.group(2))
        tokens = tuple(
            token.strip('"')
            for token in re.split(r"\s+", body.strip())
            if token
        )
        yield match.group(1).lower(), tokens


def _audit_cmake() -> dict[str, Any]:
    cmake_files = tuple(
        path
        for path in _tracked_files(("vlaforge",))
        if path.name == "CMakeLists.txt" or path.suffix == ".cmake"
    )
    declared_sources: list[dict[str, str]] = []
    subdirectories: list[dict[str, str]] = []
    invalid_edges: list[dict[str, str]] = []
    for cmake in cmake_files:
        text = cmake.read_text(encoding="utf-8")
        for command, tokens in _cmake_invocations(
            text,
            ("add_library", "add_executable", "target_sources"),
        ):
            target = tokens[0] if tokens else ""
            for token in tokens[1:]:
                suffix = Path(token).suffix.lower()
                if suffix not in {
                    ".c",
                    ".cc",
                    ".cpp",
                    ".cxx",
                    ".cu",
                    ".cuh",
                }:
                    continue
                source = (cmake.parent / token).resolve()
                record = {
                    "cmake": _relative(cmake),
                    "target": target,
                    "source": (
                        _relative(source)
                        if source.is_relative_to(_REPOSITORY_ROOT)
                        else str(source)
                    ),
                }
                declared_sources.append(record)
                if (
                    not source.is_relative_to(_SOURCE_ROOT)
                    or not source.is_file()
                    or suffix in {".cu", ".cuh"}
                ):
                    invalid_edges.append(record)
        for _, tokens in _cmake_invocations(text, ("add_subdirectory",)):
            if not tokens:
                continue
            source = (cmake.parent / tokens[0]).resolve()
            record = {
                "cmake": _relative(cmake),
                "source": (
                    _relative(source)
                    if source.is_relative_to(_REPOSITORY_ROOT)
                    else str(source)
                ),
            }
            subdirectories.append(record)
            if not source.is_relative_to(_SOURCE_ROOT):
                invalid_edges.append(record)

    root_cmake = (_REPOSITORY_ROOT / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    root_drives_vlaforge = bool(
        re.search(
            r"(?is)add_subdirectory\s*\(\s*vlaforge(?:\s|\))",
            root_cmake,
        )
    )
    tracked_cuda_sources = [
        _relative(path)
        for path in _tracked_files(("vlaforge",))
        if path.suffix.lower() in {".cu", ".cuh", ".ptx"}
    ]
    vlaforge_cmake = (_SOURCE_ROOT / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    forbidden_build_references = [
        literal
        for literal in (
            "../src",
            "${CMAKE_SOURCE_DIR}/src",
            "src/operators",
            "edge_fm",
            "third_party/coda-kernels",
        )
        if literal in vlaforge_cmake
    ]
    if (
        invalid_edges
        or tracked_cuda_sources
        or forbidden_build_references
        or root_drives_vlaforge
    ):
        raise ValueError(
            "VLAForge build graph is not isolated from old EdgeFM CUDA code"
        )
    return {
        "cmake_files": [_relative(path) for path in cmake_files],
        "declared_sources": declared_sources,
        "subdirectories": subdirectories,
        "invalid_edges": [],
        "tracked_cuda_sources": [],
        "forbidden_build_references": [],
        "root_edgefm_build_drives_vlaforge": False,
        "optional_cuda_contract": (
            "C++ AOTI backend links CUDA::cudart and executes external "
            "compiled artifacts; VLAForge declares no CUDA kernel source"
        ),
        "passed": True,
    }


def _non_production_mentions() -> list[dict[str, Any]]:
    files = _text_files(
        (
            "vlaforge/tests",
            "vlaforge/tools",
        )
    )
    selected = {
        key: value
        for key, value in _FORBIDDEN_PATTERNS.items()
        if key
        in {
            "clock_domain",
            "run_tick",
            "publish_operation",
            "physical_deadline",
            "physical_period",
        }
    }
    return scan_forbidden_text(files, selected)


def audit_repository() -> dict[str, Any]:
    production_files = _text_files(PRODUCTION_ROOTS)
    findings = scan_forbidden_text(production_files)
    if findings:
        raise ValueError(
            "legacy or out-of-scope semantics remain in production surface: "
            f"{findings}"
        )
    if frozenset(ALLOWED_OPS) != EXPECTED_OPS:
        raise ValueError(
            "Semantic IR opcode set changed without architecture audit"
        )
    revision = _git(("rev-parse", "HEAD"))
    tracked_status = _git(
        ("status", "--short", "--untracked-files=no")
    )
    file_manifest = [
        {
            "path": _relative(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in production_files
    ]
    return {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "passed": True,
        "repository": {
            "revision": revision,
            "source_dirty": bool(tracked_status),
            "tracked_status": (
                tracked_status.splitlines() if tracked_status else []
            ),
        },
        "production_surface": {
            "roots": list(PRODUCTION_ROOTS),
            "file_count": len(file_manifest),
            "files": file_manifest,
            "forbidden_findings": [],
        },
        "semantic_ir": {
            "schema": "0.2",
            "allowed_opcodes": sorted(ALLOWED_OPS),
            "old_control_or_publish_opcodes": [],
            "core_action_queue": False,
        },
        "build_graph": _audit_cmake(),
        "migration_map": list(_MIGRATION_MAP),
        "non_production_negative_mentions": _non_production_mentions(),
        "summary": {
            "passive_invocation_only": True,
            "physical_scheduling_absent": True,
            "middleware_and_publish_absent": True,
            "core_action_queue_absent": True,
            "python_runtime_dependency_absent": True,
            "old_edgefm_cuda_build_dependency_absent": True,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    build = report["build_graph"]
    lines = [
        "# VLAForge architecture and build-surface audit",
        "",
        f"Status: **{report['status']}**.",
        "",
        "The production surface contains only passive, caller-driven model "
        "invocation semantics. Mentions in negative tests are reported "
        "separately and are not runtime implementations.",
        "",
        "## Results",
        "",
        "| Invariant | Result |",
        "|---|---|",
        "| No tick/clock/deadline/period/jitter | pass |",
        "| No middleware, sensor sync, publish, or internal sleep | pass |",
        "| No core action queue/cursor | pass |",
        "| No Python runtime dependency in production surface | pass |",
        "| Semantic IR opcode set equals frozen v0.2 set | pass |",
        "| VLAForge build has no `.cu`, `.cuh`, or `.ptx` source | pass |",
        "| No source/subdirectory edge escapes `vlaforge/` | pass |",
        "| Root EdgeFM build does not implicitly build VLAForge | pass |",
        "",
        "## Build graph",
        "",
        f"- Audited CMake files: {len(build['cmake_files'])}",
        f"- Declared C/C++ sources: {len(build['declared_sources'])}",
        "- CUDA source files: 0",
        f"- Contract: {build['optional_cuda_contract']}",
        "",
        "## Old to new migration",
        "",
        "| Old surface | New surface | Status |",
        "|---|---|---|",
    ]
    for item in report["migration_map"]:
        lines.append(
            f"| {item['old']} | {item['new']} | "
            f"{item['production_status']} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This audit proves source/build isolation. Runtime correctness, "
            "no-libpython linkage, and CUDA execution remain separate clean "
            "build and generated-Session gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    report = audit_repository()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
