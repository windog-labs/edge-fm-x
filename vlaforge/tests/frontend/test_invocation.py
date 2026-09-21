from __future__ import annotations

import importlib.util
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from vlaforge.analysis import verify
from vlaforge.codegen import CppRegionDefinition, generate_compiled_cpp_session
from vlaforge.compiler import compile_module
from vlaforge.frontend import InvocationBuilder, capture_region, tensor_region
from vlaforge.interpreter import InputBinding, InputStamp, Interpreter, TensorView
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType
from vlaforge.plan import PlanExecutor, PlanModule, verify_plan


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "iterative_frontend",
    ROOT / "examples" / "iterative_frontend.py",
)
examples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(examples)
V = examples.VECTOR
FLAG = examples.FLAG


def bindings(program, *, revision=1, accepted=True, observation=(1.0, 2.0)):
    values = {
        "observation": observation,
        "noise": (0.5, -0.25),
        "token": (4,),
        "accepted": (accepted,),
    }
    return {
        port.name: InputBinding(
            TensorView(values[port.name], port.payload.shape, port.payload.dtype),
            InputStamp(revision=revision),
        )
        for port in program.module.inputs
    }


def runtimes(program, profile="verified"):
    compilation = compile_module(program.module, profile=profile)
    initial = {state.name: (0.0, 0.0) for state in program.module.states}
    options = dict(
        regions=program.regions, validators=program.validators, initial_state=initial
    )
    return (
        compilation,
        Interpreter(compilation.module, **options),
        PlanExecutor(
            compilation.plan,
            compilation.module,
            **options,
        ),
    )


@pytest.mark.parametrize(
    "factory", [examples.continuous_program, examples.autoregressive_program]
)
@pytest.mark.parametrize("profile", ["off", "verified"])
def test_two_paradigms_use_the_same_ir_compiler_and_memory_planner(factory, profile):
    program = factory()
    compilation, semantic, scheduled = runtimes(program, profile)
    assert verify(compilation.module, raise_on_error=False) == ()
    assert verify_plan(compilation.plan, raise_on_error=False) == ()
    assert PlanModule.from_dict(compilation.plan.to_dict()) == compilation.plan
    for revision in (1, 1, 2, None):
        inputs = bindings(program, revision=revision)
        left, right = semantic.run(inputs=inputs), scheduled.run(inputs=inputs)
        assert left.committed_outputs == right.committed_outputs
        assert left.state == right.state
    assert semantic.trace.to_data() == scheduled.trace.to_data()
    assert compilation.plan.arena is not None
    loop = next(task for task in compilation.plan.tasks if task.opcode == "vla.for")
    assert len(loop.outputs) == len(loop.attributes["carry_scratch"]) == 2


def test_generated_context_dependencies_exclude_unrelated_inputs():
    program = examples.continuous_program()
    compilation, runtime, _ = runtimes(program)
    (cache,) = compilation.certificate.caches
    assert cache.input_ids == (0,)
    assert cache.state_ids == ()
    runtime.run(inputs=bindings(program))
    noise_change = bindings(program)
    noise_change["noise"] = InputBinding(
        TensorView((9.0, 8.0), (2,), "f32"), InputStamp(2)
    )
    runtime.run(inputs=noise_change)
    assert runtime.cache.hits == 1
    runtime.run(inputs=bindings(program, revision=2, observation=(3.0, 4.0)))
    assert runtime.cache.misses == 2
    runtime.run(inputs=bindings(program, revision=None))
    runtime.run(inputs=bindings(program, revision=None))
    assert runtime.cache.misses == 4


def test_context_depends_on_committed_state_and_failed_run_does_not_publish():
    program = examples.autoregressive_program()
    compilation, semantic, scheduled = runtimes(program)
    (cache,) = compilation.certificate.caches
    assert cache.input_ids == (0,)
    assert cache.state_ids == (0,)
    for runtime in (semantic, scheduled):
        first = runtime.run(inputs=bindings(program))
        before = runtime.state_store.versions("history")
        with pytest.raises(RuntimeError, match="validation"):
            runtime.run(inputs=bindings(program, accepted=False))
        assert runtime.state_store.versions("history") == before
        second = runtime.run(inputs=bindings(program))
        assert second.committed_outputs != first.committed_outputs
        runtime.reset_episode(42)
        reset = runtime.run(inputs=bindings(program))
        assert reset.committed_outputs.output(
            "features"
        ) == first.committed_outputs.output("features")


def _builder():
    return InvocationBuilder(
        "test",
        inputs=(InputPort("x", V), InputPort("ok", FLAG)),
        outputs=(OutputPort("y", V),),
    )


def test_loop_callback_is_traced_once_and_loop_local_values_cannot_escape():
    b = _builder()
    x, ok = b.input("x"), b.input("ok")
    seen = []

    def body(index, carry):
        seen.append(carry)
        return b.call(examples.encode, carry)

    (result,) = b.iterate((x,), body, steps=100)
    assert len(seen) == 1
    with pytest.raises(ValueError, match="escaped"):
        b.finish({"y": seen[0]}, accepted=ok)
    assert b.finish({"y": result}, accepted=ok).module.invocations


def test_invalid_cache_scope_foreign_values_and_python_branch_are_rejected():
    b, other = _builder(), _builder()
    x = b.input("x")
    with pytest.raises(TypeError, match="control flow"):
        bool(x)
    with pytest.raises(ValueError, match="another invocation"):
        b.call(examples.encode, other.input("x"))
    with pytest.raises(ValueError, match="loop-dependent"):
        b.iterate(
            (x,),
            lambda index, carry: b.call(examples.encode, carry, cache=True),
            steps=2,
        )
    with pytest.raises(ValueError, match="callback failed"):
        b.input("x")


@pytest.mark.parametrize("kind", ["arity", "type"])
def test_invalid_carried_results_are_rejected(kind):
    b = _builder()
    x = b.input("x")
    with pytest.raises(ValueError, match="carried"):
        b.iterate(
            (x,), lambda index, carry: () if kind == "arity" else (index,), steps=2
        )


def test_memory_pins_uncached_prefill_through_the_whole_loop():
    program = examples.continuous_program(cache=False)
    compilation = compile_module(program.module)
    loop = next(t for t in compilation.plan.tasks if t.opcode == "vla.for")
    encode = next(
        t for t in compilation.plan.tasks if t.attributes.get("region") == "encode"
    )
    end = max(compilation.plan.block(loop.blocks[0]).tasks)
    allocation = next(
        a
        for a in compilation.plan.arena.physical_buffers
        if encode.outputs[0] in a.logical_buffers
    )
    assert allocation.last_task >= end


def test_variadic_verifier_rejects_mismatched_yields():
    program = examples.continuous_program()
    invocation = program.module.invocations[0]
    loop = next(op for op in invocation.body.operations if op.opcode == "vla.for")
    body = loop.regions[0]
    bad_yield = replace(body.operations[-1], operands=body.operations[-1].operands[:1])
    broken_loop = replace(
        loop, regions=(replace(body, operations=(*body.operations[:-1], bad_yield)),)
    )
    broken = replace(
        program.module,
        invocations=(
            replace(
                invocation,
                body=replace(
                    invocation.body,
                    operations=tuple(
                        broken_loop if op is loop else op
                        for op in invocation.body.operations
                    ),
                ),
            ),
        ),
    )
    assert "control.for_yield" in {d.rule for d in verify(broken, raise_on_error=False)}


_CPP = {
    "encode": """
for (int i = 0; i < 2; ++i) Output<float>(executable, 0)[i] = 2 * Input<float>(executable, 0)[i];
return vlaforge_status_ok();
""",
    "merge_context": """
for (int i = 0; i < 2; ++i) Output<float>(executable, 0)[i] = Input<float>(executable, 0)[i] + Input<float>(executable, 1)[i];
return vlaforge_status_ok();
""",
    "flow_step": """
for (int i = 0; i < 2; ++i) {
  const float v = Input<float>(executable, 0)[i] + 0.125f * Input<float>(executable, 1)[i];
  Output<float>(executable, 0)[i] = Input<float>(executable, 1)[i] - 0.25f * v + 0.0625f * Input<float>(executable, 2)[i];
  Output<float>(executable, 1)[i] = v;
}
return vlaforge_status_ok();
""",
    "token_step": """
Output<std::int64_t>(executable, 0)[0] = (Input<std::int64_t>(executable, 1)[0] + Input<std::int64_t>(executable, 3)[0] + 1) % 64;
for (int i = 0; i < 2; ++i) Output<float>(executable, 1)[i] = Input<float>(executable, 2)[i] + 0.125f * Input<float>(executable, 0)[i];
return vlaforge_status_ok();
""",
}


def _runner(program):
    lines = [
        '#include "session_generated.h"',
        "#include <cstdint>",
        "#include <cstdio>",
        "int main() {",
        "vlaforge_generated::ModelSession session;",
        "vlaforge_generated::ModelInputs inputs{};",
    ]
    data = {
        "observation": "1, 2",
        "noise": "0.5f, -0.25f",
        "token": "4",
        "accepted": "1",
        "x": "1, 2",
        "z": "3, 4",
        "ok": "1",
    }
    for port in program.module.inputs:
        dtype = {"f32": "float", "i64": "std::int64_t", "bool": "std::uint8_t"}[
            port.payload.dtype
        ]
        enum = {"f32": "F32", "i64": "I64", "bool": "BOOL"}[port.payload.dtype]
        name = port.name
        lines.extend(
            (
                f"{dtype} {name}[] = {{{data[name]}}};",
                f"const std::int64_t {name}_shape[] = {{{port.payload.shape[0]}}};",
                f"inputs.{name} = {{sizeof(VLAForgeBoundTensor), {{{name}, sizeof({name}), "
                f"{name}_shape, 1u, VLAFORGE_DTYPE_{enum}, {{VLAFORGE_DEVICE_CPU, 0}}}}, "
                "VLAFORGE_LAYOUT_CONTIGUOUS, 1u};",
            )
        )
    lines.extend(
        (
            "for (int run = 0; run < 3; ++run) {",
            "vlaforge_generated::ModelOutputs outputs{};",
            "const auto status = session.Run(inputs, &outputs);",
            'if (!status.ok()) { std::fprintf(stderr, "%s\\n", status.message); return 1; }',
        )
    )
    for port in program.module.outputs:
        dtype = {"f32": "float", "i64": "std::int64_t"}[port.payload.dtype]
        for index in range(port.payload.shape[0]):
            lines.append(
                f'std::printf("%.12g\\n", static_cast<double>(static_cast<const {dtype}*>'
                f"(outputs.{port.name}.tensor.data)[{index}]));"
            )
    lines.extend(("}", "return 0;", "}"))
    return "\n".join(lines)


def _build_and_run(program, tmp_path, *, runner_source=None):
    compilation, reference, _ = runtimes(program)
    definitions = {
        name: CppRegionDefinition(name, _CPP[fn.__name__])
        for name, fn in program.regions.items()
    }
    sources = generate_compiled_cpp_session(
        compilation,
        regions=definitions,
        validators=program.cpp_validators(),
        runner_source=_runner(program) if runner_source is None else runner_source,
        initial_state={s.name: (0.0, 0.0) for s in program.module.states},
    )
    sources.write(tmp_path / "src")
    for command in (
        [
            "cmake",
            "-S",
            str(tmp_path / "src"),
            "-B",
            str(tmp_path / "build"),
            "-G",
            "Ninja",
            f"-DVLAFORGE_RUNTIME_ROOT={ROOT}",
            "-DBUILD_TESTING=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        ["cmake", "--build", str(tmp_path / "build"), "--parallel", "4"],
    ):
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    binary = tmp_path / "build" / "vlaforge_generated_runner"
    completed = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHOME": "/nonexistent", "PYTHONPATH": "/nonexistent"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    linked = subprocess.run(
        ["ldd", str(binary)], capture_output=True, text=True, check=True
    )
    assert "libpython" not in linked.stdout.lower()
    return [float(line) for line in completed.stdout.splitlines()], reference


@pytest.mark.parametrize(
    "factory", [examples.continuous_program, examples.autoregressive_program]
)
def test_new_frontend_compiles_and_runs_without_python(factory, tmp_path):
    program = factory()
    actual, reference = _build_and_run(program, tmp_path)
    expected = []
    for _ in range(3):
        result = reference.run(inputs=bindings(program, revision=None))
        for port in program.module.outputs:
            expected.extend(result.committed_outputs.output(port.name))
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_cpp_carry_swap_is_simultaneous_and_uses_planned_scratch(tmp_path):
    b = InvocationBuilder(
        "swap",
        inputs=(InputPort("x", V), InputPort("z", V), InputPort("ok", FLAG)),
        outputs=(OutputPort("left", V), OutputPort("right", V)),
    )
    x, z, ok = b.input("x"), b.input("z"), b.input("ok")
    left, right = b.iterate((x, z), lambda i, a, c: (c, a), steps=3)
    program = b.finish({"left": left, "right": right}, accepted=ok)
    actual, _ = _build_and_run(program, tmp_path)
    assert actual == [3, 4, 1, 2] * 3


def test_nested_loops_and_scalar_carries_share_the_same_lowering(tmp_path):
    b = _builder()
    x, ok = b.input("x"), b.input("ok")

    def outer(index, value):
        value, _ = b.iterate(
            (value, index),
            lambda inner, v, i: (b.call(examples.encode, v)[0], inner),
            steps=2,
        )
        return (value,)

    (result,) = b.iterate((x,), outer, steps=3)
    program = b.finish({"y": result}, accepted=ok)
    _, semantic, scheduled = runtimes(program)
    inputs = {
        "x": InputBinding(TensorView((1.0, 2.0), (2,), "f32")),
        "ok": InputBinding(TensorView((True,), (1,), "bool")),
    }
    for runtime in (semantic, scheduled):
        assert runtime.run(inputs=inputs).committed_outputs.output("y") == (64.0, 128.0)
    actual, _ = _build_and_run(program, tmp_path)
    assert actual == [64.0, 128.0] * 3


def test_plan_rejects_aliasing_carry_scratch():
    compilation, _, _ = runtimes(examples.continuous_program())
    loop = next(t for t in compilation.plan.tasks if t.opcode == "vla.for")
    broken = replace(
        loop, attributes={**loop.attributes, "carry_scratch": list(loop.outputs)}
    )
    plan = replace(
        compilation.plan,
        tasks=tuple(broken if t is loop else t for t in compilation.plan.tasks),
    )
    assert "loop.carry_scratch" in {
        d.rule for d in verify_plan(plan, raise_on_error=False)
    }


def test_annotated_tensor_stages_keep_the_existing_torch_export_path():
    torch = pytest.importorskip("torch")
    time_type = TensorType((1,), "f32")

    @tensor_region("tensor_encode", inputs=(Value("x", V),), outputs=(V,))
    def encode_tensor(x):
        return x * 2

    @tensor_region(
        "tensor_step",
        inputs=(Value("context", V), Value("sample", V), Value("time", time_type)),
        outputs=(V, time_type),
    )
    def step_tensor(context, sample, time):
        return sample - 0.25 * (context + sample * time), time - 0.25

    @tensor_region(
        "tensor_check", inputs=(Value("x", V),), outputs=(TensorType((), "bool"),)
    )
    def check_tensor(x):
        return x.isfinite().all()

    b = InvocationBuilder(
        "tensor_stages",
        inputs=(InputPort("x", V), InputPort("noise", V), InputPort("time", time_type)),
        outputs=(OutputPort("action", V),),
    )
    x, noise, time = b.input("x"), b.input("noise"), b.input("time")
    (context,) = b.call(encode_tensor, x, cache=True)
    action, _ = b.iterate(
        (noise, time),
        lambda index, sample, t: b.call(step_tensor, context, sample, t),
        steps=4,
    )
    (accepted,) = b.call(check_tensor, action)
    program = b.finish({"action": action}, accepted=accepted)
    args = {
        "tensor_encode": (torch.tensor([1.0, 2.0]),),
        "tensor_step": (
            torch.tensor([2.0, 4.0]),
            torch.tensor([0.5, -0.25]),
            torch.ones(1),
        ),
        "tensor_check": (torch.tensor([1.0, 2.0]),),
    }
    for region in program.module.regions:
        function = program.regions[region.name]
        original_name = function.__vlaforge_region__.name
        capture = capture_region(region, function, args[original_name])
        assert capture.supported, capture.report
        assert capture.evidence.maximum_absolute_error == 0.0


def test_cpp_revision_domains_cannot_reuse_stale_context(tmp_path):
    program = examples.continuous_program()
    runner = _runner(program)
    original = "for (int run = 0; run < 3; ++run) {"
    replacement = (
        original
        + """
  observation[0] = run == 1 ? 2.0f : 1.0f;
  inputs.observation_stamp.struct_size = sizeof(VLAForgeInputStamp);
  inputs.observation_stamp.has_revision = run == 1 ? 0u : 1u;
  inputs.observation_stamp.revision = 1u;
  inputs.noise_stamp.has_revision = 1u;
  inputs.noise_stamp.struct_size = sizeof(VLAForgeInputStamp);
  inputs.noise_stamp.revision = 10u;
  inputs.accepted_stamp.has_revision = 1u;
  inputs.accepted_stamp.struct_size = sizeof(VLAForgeInputStamp);
  inputs.accepted_stamp.revision = 10u;
"""
    )
    # The explicit=1, automatic=1, explicit=1 sequence deliberately collides.
    runner = runner.replace(original, replacement)
    actual, _ = _build_and_run(program, tmp_path, runner_source=runner)
    expected = []
    for run in range(3):
        _, fresh, _ = runtimes(program)
        observation = (2.0 if run == 1 else 1.0, 2.0)
        expected.extend(
            fresh.run(
                inputs=bindings(program, revision=None, observation=observation)
            ).committed_outputs.output("action")
        )
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_cpp_uncached_context_survives_every_iteration(tmp_path):
    program = examples.continuous_program(cache=False)
    actual, reference = _build_and_run(program, tmp_path)
    expected = reference.run(inputs=bindings(program)).committed_outputs.output(
        "action"
    )
    assert actual == pytest.approx(list(expected) * 3, rel=1e-6, abs=1e-6)


def test_optional_default_has_distinct_cache_identity_from_explicit_zero(tmp_path):
    b = InvocationBuilder(
        "optional_context",
        inputs=(
            InputPort("x", V, required=False, default=(1.0, 2.0)),
            InputPort("ok", FLAG),
        ),
        outputs=(OutputPort("y", V),),
    )
    x, ok = b.input("x"), b.input("ok")
    (y,) = b.call(examples.encode, x, cache=True)
    program = b.finish({"y": y}, accepted=ok)
    _, semantic, scheduled = runtimes(program)
    expected = [(2.0, 4.0), (6.0, 4.0), (2.0, 4.0)]
    for runtime in (semantic, scheduled):
        for run in range(3):
            inputs = {"ok": InputBinding(TensorView((True,), (1,), "bool"))}
            if run == 1:
                inputs["x"] = InputBinding(
                    TensorView((3.0, 2.0), (2,), "f32"), InputStamp(revision=0)
                )
            assert (
                runtime.run(inputs=inputs).committed_outputs.output("y")
                == expected[run]
            )
    runner = _runner(program).replace(
        "for (int run = 0; run < 3; ++run) {",
        """for (int run = 0; run < 3; ++run) {
  x[0] = 3.0f;
  inputs.has_x = run == 1;
  inputs.x_stamp.struct_size = sizeof(VLAForgeInputStamp);
  inputs.x_stamp.has_revision = 1u;
  inputs.x_stamp.revision = 0u;
""",
    )
    actual, _ = _build_and_run(program, tmp_path, runner_source=runner)
    assert actual == [value for row in expected for value in row]


def test_cpp_state_and_outputs_are_atomic_across_failure_and_reset(tmp_path):
    program = examples.autoregressive_program()
    runner = _runner(program).replace(
        "for (int run = 0; run < 3; ++run) {",
        """for (int run = 0; run < 4; ++run) {
  accepted[0] = run == 1 ? 0u : 1u;
  if (run == 3 && !session.ResetEpisode(42u).ok()) return 2;
""",
    )
    runner = runner.replace(
        'if (!status.ok()) { std::fprintf(stderr, "%s\\n", status.message); return 1; }',
        """if (run == 1) {
  if (status.ok()) return 3;
  if (!session.ReadOutputTensor(0u, &outputs.token).ok()) return 4;
  if (!session.ReadOutputTensor(1u, &outputs.features).ok()) return 5;
} else if (!status.ok()) return 6;
""",
    )
    actual, reference = _build_and_run(program, tmp_path, runner_source=runner)
    first = reference.run(inputs=bindings(program)).committed_outputs
    second = reference.run(inputs=bindings(program)).committed_outputs
    expected = [
        item
        for result in (first, first, second, first)
        for port in program.module.outputs
        for item in result.output(port.name)
    ]
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)
