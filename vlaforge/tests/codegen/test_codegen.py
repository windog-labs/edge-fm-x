import os
import subprocess
from pathlib import Path

import pytest

from vlaforge.adapters import (
    build_driving_diffusion_fixture,
    build_driving_trajectory_fixture,
    build_hybrid_external_feature_fixture,
    build_openvla_fixture,
    build_smolvla_fixture,
)
from vlaforge.codegen import (
    CodegenUnsupportedError,
    CppArtifactRegionDefinition,
    CppValidatorDefinition,
    driving_diffusion_regions,
    driving_diffusion_runner_source,
    driving_diffusion_validators,
    generate_cpp_session,
    hybrid_external_feature_regions,
    hybrid_external_feature_runner_source,
    hybrid_external_feature_validators,
    openvla_fixture_regions,
    openvla_fixture_runner_source,
    openvla_fixture_validators,
    smolvla_fixture_regions,
    smolvla_fixture_runner_source,
    smolvla_fixture_validators,
)
from vlaforge.compiler import CompilerProfile, compile_module
from vlaforge.interpreter import (
    InputBinding,
    InputStamp,
    Interpreter,
    TensorView,
)
from vlaforge.plan import PlanExecutor, lower_to_plan, physicalize_plan
from vlaforge.validation import normalize_plan_trace_for_runtime


SOURCE_GOLDEN_DIGEST = (
    "67c5ffad7c823b571779a9951c59dabb6925c37e01c3ad16c01394a6814e2bc7"
)


def _sources():
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


def test_codegen_is_deterministic_and_has_v02_apis():
    _, _, first = _sources()
    _, _, second = _sources()
    assert first.files == second.files
    assert first.digest() == second.digest() == SOURCE_GOLDEN_DIGEST
    files = first.as_dict()
    header = files["session_generated.h"]
    source = files["session_generated.cpp"]
    assert "ModelInputs" in header
    assert "ModelOutputs" in header
    assert "enum class InputId" in header
    assert "vlaforge_model_session_api" in header
    assert "BindTensor" in header
    assert "BindScalar" in header
    assert "ReadOutputTensor" in header
    assert "RunTick" not in header + source
    assert "ClockDomain" not in header + source
    assert "python.h" not in (header + source).lower()
    assert "pybind" not in (header + source).lower()


def test_codegen_emits_bundle_loaded_aoti_regions() -> None:
    fixture = build_driving_trajectory_fixture()
    compilation = compile_module(
        fixture.module,
        profile=CompilerProfile.OFF,
        default_device="cuda:0",
    )
    definitions = {
        region.name: CppArtifactRegionDefinition(
            region_name=region.name,
            backend="aoti",
            artifact_path=f"artifacts/{region.name}.pt2",
            artifact_sha256=f"{index + 1:064x}",
            artifact_size_bytes=1024 + index,
            io_schema_digest=compilation.plan.io_schema_digest,
            target="sm_86",
            device="cuda:0",
            backend_variant="torch-2.10-cu128",
            residency="invocation" if index == 0 else "session",
        )
        for index, region in enumerate(compilation.module.regions)
    }
    sources = generate_cpp_session(
        compilation.plan,
        compilation.module,
        artifact_regions=definitions,
        validators={
            "finite_trajectory": CppValidatorDefinition(
                "finite_trajectory", "return data != nullptr && size_bytes > 0u;"
            )
        },
    ).as_dict()

    header = sources["session_generated.h"]
    source = sources["session_generated.cpp"]
    cmake = sources["CMakeLists.txt"]
    assert "vlaforge_model_session_create_from_bundle" in header + source
    assert "VerifyArtifactFile" in source
    assert "vlaforge_aoti_region_executable_value_api" in source
    assert "api->load" in source
    assert "api->run" in source
    assert "kArtifactInvocationResident0 =\n    true" in source
    assert "FailArtifactRegion" in source
    assert "DestroyRegion(0u);" in source
    assert "VLAFORGE_BUILD_AOTI_BACKEND ON" in cmake
    assert "RunRegion0" not in source

    bad = dict(definitions)
    first_name = next(iter(bad))
    bad[first_name] = CppArtifactRegionDefinition(
        region_name=first_name,
        backend="aoti",
        artifact_path=f"artifacts/{first_name}.pt2",
        artifact_sha256="f" * 64,
        artifact_size_bytes=1,
        io_schema_digest="0" * 64,
        target="sm_86",
        device="cuda:0",
    )
    with pytest.raises(CodegenUnsupportedError, match="schema digest mismatch"):
        generate_cpp_session(
            compilation.plan,
            compilation.module,
            artifact_regions=bad,
            validators={
                "finite_trajectory": CppValidatorDefinition(
                    "finite_trajectory", "return true;"
                )
            },
        )


def test_generated_runner_is_clean_and_matches_python(tmp_path: Path):
    fixture, _, sources = _sources()
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
            "-G",
            "Ninja",
            f"-DVLAFORGE_RUNTIME_ROOT={runtime_root}",
            "-DBUILD_TESTING=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "4"],
        check=True,
        capture_output=True,
        text=True,
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHOME": "/definitely/not/a/python/home",
            "PYTHONPATH": "/definitely/not/a/python/path",
        }
    )
    runner = build_dir / "vlaforge_generated_runner"
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

    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
    )
    expected = [
        runtime.run(inputs=item.inputs).committed_outputs.output("action")
        for item in fixture.runs
    ]
    rows = [
        tuple(float(item) for item in line.split(",")[2:])
        for line in completed.stdout.splitlines()
        if line.startswith("OUTPUT,")
    ]
    assert len(rows) == len(expected) == 3
    for row, reference in zip(rows, expected, strict=True):
        typed = row[:2]
        generic = row[2:]
        assert typed == pytest.approx(reference, abs=1e-6, rel=0)
        assert generic == pytest.approx(reference, abs=1e-6, rel=0)
        assert typed == generic


def test_generated_stateful_smolvla_matches_python_and_resets(
    tmp_path: Path,
):
    fixture = build_smolvla_fixture()
    plan = physicalize_plan(lower_to_plan(fixture.module))
    sources = generate_cpp_session(
        plan,
        fixture.module,
        regions=smolvla_fixture_regions(),
        validators=smolvla_fixture_validators(),
        runner_source=smolvla_fixture_runner_source(),
        initial_state=fixture.initial_state,
    )
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
            "-G",
            "Ninja",
            f"-DVLAFORGE_RUNTIME_ROOT={runtime_root}",
            "-DBUILD_TESTING=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "4"],
        check=True,
        capture_output=True,
        text=True,
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHOME": "/definitely/not/a/python/home",
            "PYTHONPATH": "/definitely/not/a/python/path",
        }
    )
    runner = build_dir / "vlaforge_generated_runner"
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

    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    expected = [
        runtime.run(inputs=item.inputs).committed_outputs.output("action")
        for item in fixture.runs
    ]
    runtime.reset_episode(1)
    expected.append(
        runtime.run(
            inputs=fixture.runs[0].inputs
        ).committed_outputs.output("action")
    )
    rows = [
        tuple(float(item) for item in line.split(",")[2:])
        for line in completed.stdout.splitlines()
        if line.startswith("OUTPUT,")
    ]
    assert len(rows) == len(expected) == 7
    for row, reference in zip(rows, expected, strict=True):
        typed = row[:2]
        generic = row[2:]
        assert typed == pytest.approx(reference, abs=1e-6, rel=0)
        assert generic == pytest.approx(reference, abs=1e-6, rel=0)
        assert typed == generic
    assert rows[-1][:2] == rows[0][:2]


def test_generated_hybrid_driving_session_covers_external_region_and_io_contracts(
    tmp_path: Path,
):
    fixture = build_hybrid_external_feature_fixture()
    plan = physicalize_plan(lower_to_plan(fixture.module))
    sources = generate_cpp_session(
        plan,
        fixture.module,
        regions=hybrid_external_feature_regions(),
        validators=hybrid_external_feature_validators(),
        runner_source=hybrid_external_feature_runner_source(),
    )
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
            "-G",
            "Ninja",
            f"-DVLAFORGE_RUNTIME_ROOT={runtime_root}",
            "-DBUILD_TESTING=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "4"],
        check=True,
        capture_output=True,
        text=True,
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHOME": "/definitely/not/a/python/home",
            "PYTHONPATH": "/definitely/not/a/python/path",
        }
    )
    runner = build_dir / "vlaforge_generated_runner"
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
    assert "CACHE,1,3" in completed.stdout

    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
    )
    expected_runs = [item.inputs for item in fixture.runs]
    explicit = dict(fixture.runs[-1].inputs)
    stamp = InputStamp(revision=42)
    explicit["external_bev"] = InputBinding(
        explicit["external_bev"].value,
        stamp,
    )
    explicit["route_command"] = InputBinding(
        explicit["route_command"].value,
        stamp,
    )
    explicit["agent_features"] = InputBinding(
        TensorView(
            (
                (0.5, 0.1, 0.0),
                (1.5, -0.2, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            (6, 3),
            "f32",
        ),
        stamp,
    )
    explicit["agent_valid_count"] = InputBinding(2, stamp)
    expected_runs.append(explicit)
    expected = []
    for inputs in expected_runs:
        result = runtime.run(inputs=inputs).committed_outputs
        trajectory = result.output("trajectory")
        prediction = result.output("agent_prediction")
        expected.append(
            (
                trajectory[0][0],
                trajectory[0][1],
                prediction[0][0],
                prediction[0][1],
                result.output("vqa_token"),
            )
        )

    semantic_trace = tuple(
        event.as_tuple()
        for event in normalize_plan_trace_for_runtime(
            runtime.trace,
            plan,
            fixture.module,
        )
    )
    plan_runtime = PlanExecutor(
        plan,
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
    )
    for inputs in expected_runs:
        plan_runtime.run(inputs=inputs)
    plan_trace = tuple(
        event.as_tuple()
        for event in normalize_plan_trace_for_runtime(
            plan_runtime.trace,
            plan,
            fixture.module,
        )
    )
    cpp_trace = tuple(
        tuple(int(item) for item in line.split(",")[1:])
        for line in completed.stdout.splitlines()
        if line.startswith("TRACE,")
    )
    assert semantic_trace == plan_trace == cpp_trace

    rows = [
        tuple(float(item) for item in line.split(",")[2:])
        for line in completed.stdout.splitlines()
        if line.startswith("OUTPUT,")
    ]
    assert len(rows) == len(expected) == 4
    for row, reference in zip(rows, expected, strict=True):
        typed = row[:5]
        generic = row[5:]
        assert typed == pytest.approx(reference, abs=1e-6, rel=0)
        assert generic == pytest.approx(reference, abs=1e-6, rel=0)
        assert typed == generic


def test_generated_driving_diffusion_session_preserves_candidates_scores_and_trace(
    tmp_path: Path,
):
    fixture = build_driving_diffusion_fixture()
    plan = physicalize_plan(lower_to_plan(fixture.module))
    sources = generate_cpp_session(
        plan,
        fixture.module,
        regions=driving_diffusion_regions(),
        validators=driving_diffusion_validators(),
        runner_source=driving_diffusion_runner_source(),
    )
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
            "-G",
            "Ninja",
            f"-DVLAFORGE_RUNTIME_ROOT={runtime_root}",
            "-DBUILD_TESTING=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "4"],
        check=True,
        capture_output=True,
        text=True,
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHOME": "/definitely/not/a/python/home",
            "PYTHONPATH": "/definitely/not/a/python/path",
        }
    )
    runner = build_dir / "vlaforge_generated_runner"
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

    semantic = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
    )
    expected: list[tuple[float, ...]] = []
    for run in fixture.runs:
        outputs = semantic.run(inputs=run.inputs).committed_outputs
        candidates = outputs.output("candidate_trajectories")
        scores = outputs.output("candidate_scores")
        trajectory = outputs.output("trajectory")
        expected.append(
            tuple(
                value
                for candidate in candidates
                for point in candidate
                for value in point
            )
            + tuple(scores)
            + tuple(value for point in trajectory for value in point)
        )

    scheduled = PlanExecutor(
        plan,
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
    )
    for run in fixture.runs:
        scheduled.run(inputs=run.inputs)
    semantic_trace = tuple(
        event.as_tuple()
        for event in normalize_plan_trace_for_runtime(
            semantic.trace,
            plan,
            fixture.module,
        )
    )
    plan_trace = tuple(
        event.as_tuple()
        for event in normalize_plan_trace_for_runtime(
            scheduled.trace,
            plan,
            fixture.module,
        )
    )
    cpp_trace = tuple(
        tuple(int(item) for item in line.split(",")[1:])
        for line in completed.stdout.splitlines()
        if line.startswith("TRACE,")
    )
    assert semantic_trace == plan_trace == cpp_trace

    values: dict[int, list[tuple[float, float]]] = {
        index: [] for index in range(3)
    }
    for line in completed.stdout.splitlines():
        if not line.startswith("VALUE,"):
            continue
        _, run, _, _, typed, generic = line.split(",")
        values[int(run)].append((float(typed), float(generic)))
    for run, reference in enumerate(expected):
        typed = tuple(value[0] for value in values[run])
        generic = tuple(value[1] for value in values[run])
        assert len(typed) == len(reference) == 51
        assert typed == pytest.approx(reference, abs=1e-6, rel=0)
        assert generic == pytest.approx(reference, abs=1e-6, rel=0)
        assert typed == generic
