from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from vlaforge.adapters import build_openvla_fixture, build_smolvla_fixture
from vlaforge.codegen import (
    AotiTensorSpec,
    CodegenUnsupportedError,
    OpenVLATorchScriptSpec,
    SmolVLAAotiSpec,
    generate_cpp_session,
    generate_compiled_cpp_session,
    generate_real_smolvla_aoti_runner,
    generate_real_openvla_torchscript_runner,
    openvla_fixture_regions,
    openvla_fixture_runner_source,
    openvla_fixture_validators,
)
from vlaforge.compiler import compile_module
from vlaforge.codegen.real_aoti import temporal_cache_dependencies
from vlaforge.interpreter import Interpreter
from vlaforge.plan import PlanExecutor, lower_to_plan, physicalize_plan
from vlaforge.validation import normalize_plan_trace_for_runtime


SOURCE_GOLDEN_DIGEST = (
    "024fcaace55af835665232bce50e013cd968474629b351af0107ba44240b1315"
)
REAL_SMOLVLA_SOURCE_GOLDEN_DIGEST = (
    "aecc8a598256d59e63b62c7b6ef6e414df80996eb1a67ab495e62b22ee25d057"
)
REAL_OPENVLA_SOURCE_GOLDEN_DIGEST = (
    "7ea0c7ce6ea16263132b24df50f4eb52ee90199b12c9792cae654fc9dbfe1558"
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


def test_normal_codegen_consumes_verified_certificate() -> None:
    fixture = build_openvla_fixture()
    compilation = compile_module(fixture.module, profile="verified")
    sources = generate_compiled_cpp_session(
        compilation,
        regions=openvla_fixture_regions(),
        validators=openvla_fixture_validators(),
        runner_source=openvla_fixture_runner_source(),
    )
    files = sources.as_dict()
    assert "optimization_certificate.h" in files
    assert compilation.certificate.digest() in files[
        "optimization_certificate.h"
    ]
    assert "cache_guards_[0u].Lookup" in files["session_generated.cpp"]
    assert "cache.Invalidate()" in files["session_generated.cpp"]


def test_codegen_rejects_certificate_plan_mismatch() -> None:
    fixture = build_openvla_fixture()
    compilation = compile_module(fixture.module, profile="verified")
    baseline = compile_module(fixture.module, profile="off")
    with pytest.raises(ValueError, match="plan digest mismatch"):
        generate_cpp_session(
            baseline.plan,
            baseline.module,
            regions=openvla_fixture_regions(),
            validators=openvla_fixture_validators(),
            compilation_certificate=compilation.certificate,
        )


def test_real_smolvla_aoti_codegen_is_deterministic_and_no_python() -> None:
    spec = SmolVLAAotiSpec(
        image=AotiTensorSpec((1, 3, 256, 256), "f32"),
        state=AotiTensorSpec((1, 6), "f32"),
        token_ids=(1, 2, 3),
        token_mask=(True, True, False),
        prefix_outputs=(
            AotiTensorSpec((1, 113), "bool"),
            AotiTensorSpec((1, 113, 5, 64), "bf16"),
            AotiTensorSpec((1, 113, 5, 64), "bf16"),
        ),
        solver_output=AotiTensorSpec((1, 50, 32), "f32"),
        action_chunk=AotiTensorSpec((1, 50, 6), "f32"),
    )
    first = generate_real_smolvla_aoti_runner(spec)
    second = generate_real_smolvla_aoti_runner(spec)
    assert first == second
    assert first.digest() == REAL_SMOLVLA_SOURCE_GOLDEN_DIGEST
    text = "\n".join(content for _, content in first.files).lower()
    assert "python.h" not in text
    assert "pybind" not in text
    assert "nlohmann" not in text
    assert "std::unordered_map" not in text
    assert "ktasksolver" in text
    assert "for (std::int64_t step = 0; step < 10; ++step)" in text
    assert "epochversioncacheguard prefix_cache" in text
    assert "compilation_certificate.json" in first.as_dict()
    benchmark = generate_real_smolvla_aoti_runner(
        spec,
        optimization_benchmark=True,
    )
    benchmark_text = "\n".join(
        content for _, content in benchmark.files
    )
    assert "EpochVersionCacheGuard prefix_cache" in benchmark_text
    assert "benchmark_licm_disabled" in benchmark_text
    assert "BENCH_TICK_US" in benchmark_text
    assert benchmark.digest() != first.digest()


def test_real_openvla_codegen_is_deterministic_and_no_python() -> None:
    prefill_cache = AotiTensorSpec((1, 32, 275, 128), "bf16")
    decode_cache = AotiTensorSpec((1, 32, 276, 128), "bf16")
    logits = AotiTensorSpec((1, 32064), "f32")
    spec = OpenVLATorchScriptSpec(
        prefill_inputs=(
            AotiTensorSpec((1, 6, 224, 224), "bf16"),
            AotiTensorSpec((1, 19), "i64"),
            AotiTensorSpec((1, 19), "i64"),
        ),
        prefill_outputs=(logits,) + (prefill_cache,) * 64,
        decode_inputs=(AotiTensorSpec((1, 1), "i64"),)
        + (prefill_cache,) * 64,
        decode_outputs=(logits,) + (decode_cache,) * 64,
        action_tokens=AotiTensorSpec((1, 7), "i64"),
        action=AotiTensorSpec((7,), "f64"),
    )
    first = generate_real_openvla_torchscript_runner(spec)
    second = generate_real_openvla_torchscript_runner(spec)
    assert first == second
    assert first.digest() == REAL_OPENVLA_SOURCE_GOLDEN_DIGEST
    text = "\n".join(content for _, content in first.files).lower()
    assert "python.h" not in text
    assert "pybind" not in text
    assert "nlohmann" not in text
    assert "std::unordered_map" not in text
    assert "ktaskdecode" in text
    assert "step < kdecodesteps" in text
    assert "epochversioncacheguard prefill_cache" in text
    assert "compilation_certificate.json" in first.as_dict()
    benchmark = generate_real_openvla_torchscript_runner(
        spec,
        optimization_benchmark=True,
    )
    benchmark_text = "\n".join(
        content for _, content in benchmark.files
    )
    assert "EpochVersionCacheGuard prefill_cache" in benchmark_text
    assert "BENCH_TICK_US" in benchmark_text
    assert benchmark.digest() != first.digest()


def test_real_codegen_cache_dependencies_come_from_semantic_epochs() -> None:
    from vlaforge.adapters import (
        build_real_openvla_action_program,
        build_real_smolvla_action_program,
    )

    smol = temporal_cache_dependencies(
        build_real_smolvla_action_program(
            chunk_size=50,
            max_action_dim=32,
            output_action_dim=6,
            num_steps=10,
        ),
        "prepare_prefix",
    )
    assert [
        (item.kind, item.subject_id, item.max_age_ns)
        for item in smol
    ] == [("epoch", 0, 50_000_000)]

    openvla = temporal_cache_dependencies(
        build_real_openvla_action_program(action_dim=7),
        "generate_action_tokens_prefill",
    )
    assert [
        (item.kind, item.subject_id, item.max_age_ns)
        for item in openvla
    ] == [
        ("epoch", 0, 60_000_000),
        ("epoch", 1, 60_000_000),
        ("epoch", 2, 60_000_000),
    ]


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
