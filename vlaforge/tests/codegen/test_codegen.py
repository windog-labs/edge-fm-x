from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from vlaforge.adapters import build_openvla_fixture, build_smolvla_fixture
from vlaforge.codegen import (
    CodegenUnsupportedError,
    generate_cpp_session,
    openvla_fixture_regions,
    openvla_fixture_runner_source,
    openvla_fixture_validators,
)
from vlaforge.interpreter import Interpreter
from vlaforge.plan import PlanExecutor, lower_to_plan, physicalize_plan
from vlaforge.validation import normalize_plan_trace_for_runtime


SOURCE_GOLDEN_DIGEST = (
    "d05684708daa9e96c15d26319bdfdb8fefcca3eb3a57920abfc815e53764ef9d"
)


def _fixture_sources():
    fixture = build_openvla_fixture()
    plan = physicalize_plan(lower_to_plan(fixture.module))
    sources = generate_cpp_session(
        plan,
        fixture.module,
        regions=openvla_fixture_regions(),
        validators=openvla_fixture_validators(),
        runner_source=openvla_fixture_runner_source(),
    )
    return fixture, plan, sources


def test_codegen_is_deterministic_and_matches_golden() -> None:
    _, _, first = _fixture_sources()
    _, _, second = _fixture_sources()
    assert first.files == second.files
    assert first.digest() == second.digest() == SOURCE_GOLDEN_DIGEST
    assert [name for name, _ in first.files] == [
        "CMakeLists.txt",
        "memory_constants.h",
        "runner.cpp",
        "session_generated.cpp",
        "session_generated.h",
    ]
    generated_text = "\n".join(content for _, content in first.files).lower()
    assert "python.h" not in generated_text
    assert "pybind" not in generated_text
    assert "nlohmann" not in generated_text
    assert "smolvla" not in generated_text
    assert "openvla" not in generated_text


def test_codegen_reports_unsupported_control_without_fallback() -> None:
    fixture = build_smolvla_fixture()
    plan = physicalize_plan(lower_to_plan(fixture.module))
    with pytest.raises(CodegenUnsupportedError, match="definitions mismatch"):
        generate_cpp_session(
            plan,
            fixture.module,
            regions={},
            validators={},
        )


def test_generated_runner_is_no_python_and_trace_equivalent(
    tmp_path: Path,
) -> None:
    fixture, plan, sources = _fixture_sources()
    source_dir = tmp_path / "source"
    build_dir = tmp_path / "build"
    sources.write(source_dir)
    runtime_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DVLAFORGE_RUNTIME_ROOT={runtime_root}",
            "-DBUILD_TESTING=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel"],
        check=True,
        capture_output=True,
        text=True,
    )
    install_dir = tmp_path / "install"
    subprocess.run(
        [
            "cmake",
            "--install",
            str(build_dir),
            "--prefix",
            str(install_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    export_file = (
        install_dir
        / "lib"
        / "cmake"
        / "VLAForgeRuntime"
        / "VLAForgeRuntimeTargets.cmake"
    )
    assert export_file.exists()
    assert "vlaforge_generated_session" in export_file.read_text(
        encoding="utf-8"
    )
    consumer_source = tmp_path / "consumer"
    consumer_source.mkdir()
    (consumer_source / "main.cpp").write_text(
        '#include "session_generated.h"\n'
        "int main() {\n"
        "  vlaforge_generated::GeneratedSession session;\n"
        "  return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    (consumer_source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(vlaforge_generated_consumer LANGUAGES CXX)\n"
        f'include("{export_file}")\n'
        "add_executable(consumer main.cpp)\n"
        "target_link_libraries(consumer PRIVATE "
        "VLAForge::vlaforge_generated_session)\n",
        encoding="utf-8",
    )
    consumer_build = tmp_path / "consumer-build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(consumer_source),
            "-B",
            str(consumer_build),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(consumer_build), "--parallel"],
        check=True,
        capture_output=True,
        text=True,
    )
    runner = build_dir / "vlaforge_generated_runner"
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHOME": "/definitely/not/a/python/home",
            "PYTHONPATH": "/definitely/not/a/python/path",
        }
    )
    completed = subprocess.run(
        [str(runner)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    linked = subprocess.run(
        ["ldd", str(runner)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.lower()
    assert "python" not in linked

    actions = []
    runtime_trace = []
    for line in completed.stdout.splitlines():
        fields = line.split(",")
        if fields[0] == "ACTION":
            actions.append((float(fields[2]), float(fields[3])))
        elif fields[0] == "TRACE":
            runtime_trace.append(tuple(int(item) for item in fields[1:]))

    executor = PlanExecutor(
        plan,
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    expected_actions = []
    for item in fixture.ticks:
        result = executor.run_tick("act", item.tick, item.inputs)
        expected_actions.append(result.published_actions[0].value)
    expected_trace = [
        event.as_tuple()
        for event in normalize_plan_trace_for_runtime(
            executor.trace, plan, fixture.module
        )
    ]
    semantic = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    semantic_actions = []
    for item in fixture.ticks:
        result = semantic.run_tick("act", item.tick, item.inputs)
        semantic_actions.append(result.published_actions[0].value)
    semantic_trace = [
        event.as_tuple()
        for event in normalize_plan_trace_for_runtime(
            semantic.trace, plan, fixture.module
        )
    ]

    assert len(actions) == len(expected_actions) == 3
    assert semantic_actions == expected_actions
    for actual, expected in zip(actions, expected_actions, strict=True):
        assert actual == pytest.approx(expected, abs=1e-6, rel=0)
    assert semantic_trace == expected_trace
    assert runtime_trace == expected_trace
