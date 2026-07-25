"""Deterministic driving fixtures covering four deployment paradigms."""

from __future__ import annotations

import math

from vlaforge.adapters.common import AdapterFixture, FixtureRun
from vlaforge.frontend.builder import ModuleBuilder
from vlaforge.interpreter import InputBinding, InputStamp, TensorView
from vlaforge.ir import ops
from vlaforge.ir.program import (
    Block,
    InputPort,
    Invocation,
    Operation,
    OutputPort,
    TensorRegion,
    Value,
)
from vlaforge.ir.types import PendingOutputType, ScalarType, TensorType


I32 = ScalarType("i32")
I64 = ScalarType("i64")
BOOL = ScalarType("bool")


def _tensor(data: object, type_: TensorType) -> TensorView:
    return TensorView(
        data,
        tuple(int(item) for item in type_.shape),
        type_.dtype,
        type_.layout,
    )


def _transactional_outputs(
    outputs: tuple[tuple[str, str, TensorType | ScalarType], ...],
    *,
    group: str,
    transaction: str,
    condition: str,
) -> tuple[Operation, ...]:
    pending_types = tuple(
        PendingOutputType(port, type_) for _, port, type_ in outputs
    )
    return (
        *(
            ops.output_create(f"pending_{port}", value, type_, port)
            for value, port, type_ in outputs
        ),
        ops.output_group(
            "pending_outputs",
            group,
            tuple(
                (f"pending_{port}", pending)
                for (_, port, _), pending in zip(
                    outputs, pending_types, strict=True
                )
            ),
        ),
        ops.transaction_commit(
            "committed_outputs",
            pending_types,
            group,
            transaction,
            "pending_outputs",
            condition,
        ),
        ops.return_values("committed_outputs"),
    )


def build_driving_trajectory_fixture() -> AdapterFixture:
    """Multi-camera + ego history -> one stateless trajectory."""

    cameras_t = TensorType((3, 2), "f32")
    camera_mask_t = TensorType((3,), "bool")
    ego_history_t = TensorType((4, 3), "f32")
    route_t = TensorType((3,), "f32")
    condition_t = TensorType((4,), "f32")
    trajectory_t = TensorType((6, 3), "f32")

    builder = ModuleBuilder("driving_trajectory_fixture")
    builder.add_input(InputPort("multi_camera", cameras_t))
    builder.add_input(InputPort("camera_valid_mask", camera_mask_t))
    builder.add_input(InputPort("ego_history", ego_history_t))
    builder.add_input(
        InputPort(
            "ego_valid_count",
            I32,
            value_range=(1, 4),
            valid_for="ego_history",
        )
    )
    builder.add_input(InputPort("route_command", route_t))
    builder.add_output(
        OutputPort("trajectory", trajectory_t, group="planning")
    )
    builder.add_region(
        TensorRegion(
            "encode_driving_context",
            (
                Value("cameras", cameras_t),
                Value("camera_mask", camera_mask_t),
                Value("ego_history", ego_history_t),
                Value("ego_count", I32),
                Value("route", route_t),
            ),
            (condition_t,),
            metadata={"memoize": True, "template": "StatelessTrajectory"},
        )
    )
    builder.add_region(
        TensorRegion(
            "trajectory_head",
            (Value("condition", condition_t),),
            (trajectory_t,),
        )
    )
    body = Block.of(
        (
            ops.input_read("cameras", "cameras_rev", cameras_t, "multi_camera"),
            ops.input_read(
                "camera_mask",
                "camera_mask_rev",
                camera_mask_t,
                "camera_valid_mask",
            ),
            ops.input_read(
                "ego_history_value",
                "ego_history_rev",
                ego_history_t,
                "ego_history",
            ),
            ops.input_read(
                "ego_count",
                "ego_count_rev",
                I32,
                "ego_valid_count",
            ),
            ops.input_read("route", "route_rev", route_t, "route_command"),
            ops.transaction_begin("txn"),
            ops.invoke(
                ("condition",),
                (condition_t,),
                "encode_driving_context",
                (
                    "cameras",
                    "camera_mask",
                    "ego_history_value",
                    "ego_count",
                    "route",
                ),
            ),
            ops.invoke(
                ("trajectory",),
                (trajectory_t,),
                "trajectory_head",
                ("condition",),
            ),
            ops.validate(
                "trajectory_valid",
                "trajectory",
                "finite_trajectory",
            ),
            *_transactional_outputs(
                (("trajectory", "trajectory", trajectory_t),),
                group="planning",
                transaction="txn",
                condition="trajectory_valid",
            ),
        )
    )
    builder.add_invocation(Invocation("act", body))

    def encode(cameras, camera_mask, ego_history, ego_count, route):
        camera_sum = sum(
            sum(cameras[index])
            for index in range(3)
            if camera_mask[index]
        )
        ego = ego_history[ego_count - 1]
        return (
            float(camera_sum),
            float(ego[0]),
            float(route[0]),
            float(route[1]),
        )

    def head(condition):
        return tuple(
            (
                condition[1] + 0.5 * step,
                condition[2] + 0.1 * condition[0],
                condition[3] * 0.05,
            )
            for step in range(6)
        )

    first_inputs = {
        "multi_camera": InputBinding(
            _tensor(((0.1, 0.2), (0.3, 0.4), (0.0, 0.0)), cameras_t),
            InputStamp(revision=10),
        ),
        "camera_valid_mask": InputBinding(
            _tensor((True, True, False), camera_mask_t),
            InputStamp(revision=10),
        ),
        "ego_history": InputBinding(
            _tensor(
                (
                    (0.0, 0.0, 0.0),
                    (0.2, 0.0, 0.0),
                    (0.4, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                ),
                ego_history_t,
            ),
            InputStamp(revision=10),
        ),
        "ego_valid_count": InputBinding(3, InputStamp(revision=10)),
        "route_command": InputBinding(
            _tensor((1.0, 0.2, 0.0), route_t),
            InputStamp(revision=10),
        ),
    }
    second_inputs = {
        name: InputBinding(binding.value, InputStamp(revision=11))
        for name, binding in first_inputs.items()
    }
    return AdapterFixture(
        module=builder.build(),
        regions={
            "encode_driving_context": encode,
            "trajectory_head": head,
        },
        validators={"finite_trajectory": _all_finite},
        initial_state={},
        runs=(
            FixtureRun(first_inputs),
            FixtureRun(first_inputs),
            FixtureRun(second_inputs),
        ),
    )


def build_driving_ar_fixture() -> AdapterFixture:
    """AutoVLA-like bounded trajectory tokens with fast/slow branch."""

    context_input_t = TensorType((4,), "f32")
    condition_t = TensorType((3,), "f32")
    trajectory_t = TensorType((5, 2), "f32")

    builder = ModuleBuilder("driving_ar_fixture")
    builder.add_input(InputPort("scene_context", context_input_t))
    builder.add_input(InputPort("force_slow", BOOL))
    builder.add_output(
        OutputPort("trajectory", trajectory_t, group="planning")
    )
    builder.add_output(OutputPort("planner_mode", I32, group="planning"))
    builder.add_region(
        TensorRegion(
            "ar_prefill",
            (Value("scene", context_input_t),),
            (condition_t,),
            metadata={
                "memoize": True,
                "template": "AutoregressiveTrajectory",
            },
        )
    )
    builder.add_region(
        TensorRegion(
            "choose_fast_path",
            (
                Value("condition", condition_t),
                Value("force_slow_value", BOOL),
            ),
            (BOOL,),
        )
    )
    builder.add_region(
        TensorRegion(
            "fast_trajectory",
            (Value("condition", condition_t),),
            (trajectory_t,),
        )
    )
    builder.add_region(TensorRegion("fast_mode", (), (I32,)))
    builder.add_region(
        TensorRegion(
            "initial_trajectory_token",
            (Value("condition", condition_t),),
            (I64,),
        )
    )
    builder.add_region(
        TensorRegion(
            "next_trajectory_token",
            (
                Value("condition", condition_t),
                Value("token", I64),
                Value("step", ScalarType("index")),
            ),
            (I64,),
        )
    )
    builder.add_region(
        TensorRegion(
            "trajectory_detokenize",
            (
                Value("condition", condition_t),
                Value("token", I64),
            ),
            (trajectory_t,),
        )
    )
    builder.add_region(TensorRegion("slow_mode", (), (I32,)))

    fast_branch = Block.of(
        (
            ops.invoke(
                ("fast_result",),
                (trajectory_t,),
                "fast_trajectory",
                ("condition",),
            ),
            ops.invoke(("fast_mode_value",), (I32,), "fast_mode", ()),
            ops.yield_values("fast_result", "fast_mode_value"),
        )
    )
    token_loop = Block.of(
        (
            ops.invoke(
                ("next_token",),
                (I64,),
                "next_trajectory_token",
                ("condition", "token_iter", "token_step"),
            ),
            ops.yield_values("next_token"),
        )
    )
    slow_branch = Block.of(
        (
            ops.invoke(
                ("initial_token",),
                (I64,),
                "initial_trajectory_token",
                ("condition",),
            ),
            ops.for_loop(
                Value("final_token", I64),
                "initial_token",
                Value("token_step", ScalarType("index")),
                Value("token_iter", I64),
                token_loop,
                lower=0,
                upper=4,
            ),
            ops.invoke(
                ("slow_result",),
                (trajectory_t,),
                "trajectory_detokenize",
                ("condition", "final_token"),
            ),
            ops.invoke(("slow_mode_value",), (I32,), "slow_mode", ()),
            ops.yield_values("slow_result", "slow_mode_value"),
        )
    )
    body = Block.of(
        (
            ops.input_read("scene", "scene_rev", context_input_t, "scene_context"),
            ops.input_read(
                "force_slow_value",
                "force_slow_rev",
                BOOL,
                "force_slow",
            ),
            ops.transaction_begin("txn"),
            ops.invoke(
                ("condition",),
                (condition_t,),
                "ar_prefill",
                ("scene",),
            ),
            ops.invoke(
                ("use_fast",),
                (BOOL,),
                "choose_fast_path",
                ("condition", "force_slow_value"),
            ),
            ops.if_op(
                (
                    Value("trajectory", trajectory_t),
                    Value("planner_mode", I32),
                ),
                "use_fast",
                fast_branch,
                slow_branch,
            ),
            ops.validate(
                "trajectory_valid",
                "trajectory",
                "finite_trajectory",
            ),
            *_transactional_outputs(
                (
                    ("trajectory", "trajectory", trajectory_t),
                    ("planner_mode", "planner_mode", I32),
                ),
                group="planning",
                transaction="txn",
                condition="trajectory_valid",
            ),
        )
    )
    builder.add_invocation(
        Invocation(
            "act",
            body,
            metadata={
                "adapter_template": "AutoregressiveTrajectory",
                "evidence_level": "L1",
                "source_contract": "AutoVLA-like",
                "core_op_delta": 0,
            },
        )
    )

    def prefill(scene):
        return float(scene[0]), float(scene[1]), float(sum(scene))

    def fast_path(condition, force_slow):
        return not force_slow and condition[2] < 3.0

    def fast_trajectory(condition):
        return tuple(
            (condition[0] + step * 0.4, condition[1])
            for step in range(5)
        )

    def initial_token(condition):
        return int(abs(condition[2]) * 7) % 31

    def next_token(condition, token, step):
        return (token * 5 + int(condition[0] * 10) + step) % 31

    def detokenize(condition, token):
        lateral = (token - 15) / 30.0
        return tuple(
            (condition[0] + step * 0.25, condition[1] + lateral)
            for step in range(5)
        )

    def bindings(revision: int, force_slow: bool):
        return {
            "scene_context": InputBinding(
                _tensor((0.2, 0.1, 0.3, 0.4), context_input_t),
                InputStamp(revision=revision),
            ),
            "force_slow": InputBinding(
                force_slow,
                InputStamp(revision=revision),
            ),
        }

    return AdapterFixture(
        module=builder.build(),
        regions={
            "ar_prefill": prefill,
            "choose_fast_path": fast_path,
            "fast_trajectory": fast_trajectory,
            "fast_mode": lambda: 0,
            "initial_trajectory_token": initial_token,
            "next_trajectory_token": next_token,
            "trajectory_detokenize": detokenize,
            "slow_mode": lambda: 1,
        },
        validators={"finite_trajectory": _all_finite},
        initial_state={},
        runs=(
            FixtureRun(bindings(20, False)),
            FixtureRun(bindings(20, False)),
            FixtureRun(bindings(21, True)),
        ),
    )


def build_driving_diffusion_fixture() -> AdapterFixture:
    """DiffusionDrive-like two-step K-candidate planner."""

    scene_t = TensorType((4,), "f32")
    agents_t = TensorType((8, 4), "f32")
    route_t = TensorType((3,), "f32")
    condition_t = TensorType((4,), "f32")
    candidates_t = TensorType((3, 6, 2), "f32")
    scores_t = TensorType((3,), "f32")
    trajectory_t = TensorType((6, 2), "f32")

    builder = ModuleBuilder("driving_diffusion_fixture")
    builder.add_input(InputPort("scene_feature", scene_t))
    builder.add_input(InputPort("agent_features", agents_t))
    builder.add_input(
        InputPort(
            "agent_valid_count",
            I32,
            value_range=(0, 8),
            valid_for="agent_features",
        )
    )
    builder.add_input(InputPort("route_command", route_t))
    builder.add_output(
        OutputPort("candidate_trajectories", candidates_t, group="planning")
    )
    builder.add_output(
        OutputPort("candidate_scores", scores_t, group="planning")
    )
    builder.add_output(
        OutputPort("trajectory", trajectory_t, group="planning")
    )
    builder.add_region(
        TensorRegion(
            "diffusion_condition",
            (
                Value("scene", scene_t),
                Value("agents", agents_t),
                Value("agent_count", I32),
                Value("route", route_t),
            ),
            (condition_t,),
            metadata={"memoize": True, "template": "DiffusionPlanner"},
        )
    )
    builder.add_region(
        TensorRegion(
            "initialize_candidates",
            (Value("condition", condition_t),),
            (candidates_t,),
        )
    )
    builder.add_region(
        TensorRegion(
            "denoise_candidates",
            (
                Value("condition", condition_t),
                Value("candidates", candidates_t),
                Value("step", ScalarType("index")),
            ),
            (candidates_t,),
        )
    )
    builder.add_region(
        TensorRegion(
            "score_candidates",
            (Value("candidates", candidates_t),),
            (scores_t, trajectory_t),
        )
    )
    denoise_body = Block.of(
        (
            ops.invoke(
                ("candidates_next",),
                (candidates_t,),
                "denoise_candidates",
                ("condition", "candidates_iter", "denoise_step"),
            ),
            ops.yield_values("candidates_next"),
        )
    )
    body = Block.of(
        (
            ops.input_read("scene", "scene_rev", scene_t, "scene_feature"),
            ops.input_read(
                "agents",
                "agents_rev",
                agents_t,
                "agent_features",
            ),
            ops.input_read(
                "agent_count",
                "agent_count_rev",
                I32,
                "agent_valid_count",
            ),
            ops.input_read("route", "route_rev", route_t, "route_command"),
            ops.transaction_begin("txn"),
            ops.invoke(
                ("condition",),
                (condition_t,),
                "diffusion_condition",
                ("scene", "agents", "agent_count", "route"),
            ),
            ops.invoke(
                ("candidates_initial",),
                (candidates_t,),
                "initialize_candidates",
                ("condition",),
            ),
            ops.for_loop(
                Value("candidate_trajectories", candidates_t),
                "candidates_initial",
                Value("denoise_step", ScalarType("index")),
                Value("candidates_iter", candidates_t),
                denoise_body,
                lower=0,
                upper=2,
            ),
            ops.invoke(
                ("candidate_scores", "trajectory"),
                (scores_t, trajectory_t),
                "score_candidates",
                ("candidate_trajectories",),
            ),
            ops.validate(
                "trajectory_valid",
                "trajectory",
                "finite_trajectory",
            ),
            *_transactional_outputs(
                (
                    (
                        "candidate_trajectories",
                        "candidate_trajectories",
                        candidates_t,
                    ),
                    ("candidate_scores", "candidate_scores", scores_t),
                    ("trajectory", "trajectory", trajectory_t),
                ),
                group="planning",
                transaction="txn",
                condition="trajectory_valid",
            ),
        )
    )
    builder.add_invocation(Invocation("act", body))

    def condition(scene, agents, count, route):
        crowd = sum(agents[index][0] for index in range(count))
        return scene[0], route[0], route[1], crowd

    def initialize(condition_value):
        return tuple(
            tuple(
                (step * 0.5, condition_value[2] + candidate * 0.2)
                for step in range(6)
            )
            for candidate in range(3)
        )

    def denoise(condition_value, candidates, step):
        scale = 0.5 / (step + 1)
        return tuple(
            tuple(
                (
                    point[0] + scale * condition_value[0],
                    point[1] - scale * condition_value[3],
                )
                for point in candidate
            )
            for candidate in candidates
        )

    def score(candidates):
        scores = tuple(
            -sum(abs(point[1]) for point in candidate)
            for candidate in candidates
        )
        best = max(range(len(scores)), key=scores.__getitem__)
        return scores, candidates[best]

    agent_values = tuple(
        (0.1 * index, 0.0, 0.0, 1.0) for index in range(8)
    )

    def bindings(revision: int):
        return {
            "scene_feature": InputBinding(
                _tensor((0.2, 0.1, 0.0, 0.3), scene_t),
                InputStamp(revision=revision),
            ),
            "agent_features": InputBinding(
                _tensor(agent_values, agents_t),
                InputStamp(revision=revision),
            ),
            "agent_valid_count": InputBinding(
                3, InputStamp(revision=revision)
            ),
            "route_command": InputBinding(
                _tensor((1.0, 0.2, 0.0), route_t),
                InputStamp(revision=revision),
            ),
        }

    return AdapterFixture(
        module=builder.build(),
        regions={
            "diffusion_condition": condition,
            "initialize_candidates": initialize,
            "denoise_candidates": denoise,
            "score_candidates": score,
        },
        validators={"finite_trajectory": _all_finite},
        initial_state={},
        runs=(
            FixtureRun(bindings(30)),
            FixtureRun(bindings(30)),
            FixtureRun(bindings(31)),
        ),
    )


def build_hybrid_external_feature_fixture() -> AdapterFixture:
    """DriveVLM-Dual-like external BEV feature + multi-task outputs."""

    external_bev_t = TensorType((4, 4), "f32")
    agents_t = TensorType((6, 3), "f32")
    route_t = TensorType((3,), "f32")
    bev_token_t = TensorType((4,), "f32")
    trajectory_t = TensorType((6, 2), "f32")
    prediction_t = TensorType((6, 2), "f32")

    builder = ModuleBuilder("hybrid_external_feature_fixture")
    builder.add_input(
        InputPort("external_bev", external_bev_t, extension=True)
    )
    builder.add_input(
        InputPort(
            "agent_features",
            agents_t,
            required=False,
            default=tuple((0.0, 0.0, 0.0) for _ in range(6)),
        )
    )
    builder.add_input(
        InputPort(
            "agent_valid_count",
            I32,
            required=False,
            default=0,
            value_range=(0, 6),
            valid_for="agent_features",
        )
    )
    builder.add_input(InputPort("route_command", route_t))
    builder.add_output(
        OutputPort("trajectory", trajectory_t, group="planning")
    )
    builder.add_output(
        OutputPort("agent_prediction", prediction_t, group="planning")
    )
    builder.add_output(OutputPort("vqa_token", I64, group="planning"))
    builder.add_region(
        TensorRegion(
            "external_bev_preprocess",
            (Value("external_bev_value", external_bev_t),),
            (bev_token_t,),
            metadata={
                "memoize": True,
                "external_cpp_region": True,
                "plugin_abi": "vlaforge.region_executable/2",
                "artifact_provider": "customer",
            },
        )
    )
    builder.add_region(
        TensorRegion(
            "hybrid_planner",
            (
                Value("bev_token", bev_token_t),
                Value("agents", agents_t),
                Value("agent_count", I32),
                Value("route", route_t),
            ),
            (trajectory_t, prediction_t, I64),
            metadata={"template": "MultiTaskDriving"},
        )
    )
    body = Block.of(
        (
            ops.input_read(
                "external_bev_value",
                "external_bev_rev",
                external_bev_t,
                "external_bev",
            ),
            ops.input_read(
                "agents",
                "agents_rev",
                agents_t,
                "agent_features",
            ),
            ops.input_read(
                "agent_count",
                "agent_count_rev",
                I32,
                "agent_valid_count",
            ),
            ops.input_read("route", "route_rev", route_t, "route_command"),
            ops.transaction_begin("txn"),
            ops.invoke(
                ("bev_token",),
                (bev_token_t,),
                "external_bev_preprocess",
                ("external_bev_value",),
            ),
            ops.invoke(
                ("trajectory", "agent_prediction", "vqa_token"),
                (trajectory_t, prediction_t, I64),
                "hybrid_planner",
                ("bev_token", "agents", "agent_count", "route"),
            ),
            ops.validate(
                "trajectory_valid",
                "trajectory",
                "finite_trajectory",
            ),
            *_transactional_outputs(
                (
                    ("trajectory", "trajectory", trajectory_t),
                    (
                        "agent_prediction",
                        "agent_prediction",
                        prediction_t,
                    ),
                    ("vqa_token", "vqa_token", I64),
                ),
                group="planning",
                transaction="txn",
                condition="trajectory_valid",
            ),
        )
    )
    builder.add_invocation(Invocation("act", body))

    def preprocess(bev):
        return tuple(sum(row[index] for row in bev) for index in range(4))

    def planner(bev_token, agents, count, route):
        trajectory = tuple(
            (
                step * 0.4 + bev_token[0] * 0.01,
                route[1] + bev_token[1] * 0.01,
            )
            for step in range(6)
        )
        prediction = tuple(
            (
                (
                    agents[index % count][0] + step * 0.1
                    if count
                    else step * 0.1
                ),
                agents[index % count][1] if count else 0.0,
            )
            for index, step in enumerate(range(6))
        )
        token = int(abs(sum(bev_token) + route[0]) * 10) % 1024
        return trajectory, prediction, token

    def bindings(revision: int):
        return {
            "external_bev": InputBinding(
                _tensor(
                    tuple(
                        tuple((row + column) * 0.1 for column in range(4))
                        for row in range(4)
                    ),
                    external_bev_t,
                ),
                InputStamp(revision=revision),
            ),
            "route_command": InputBinding(
                _tensor((1.0, 0.3, 0.0), route_t),
                InputStamp(revision=revision),
            ),
        }

    return AdapterFixture(
        module=builder.build(),
        regions={
            "external_bev_preprocess": preprocess,
            "hybrid_planner": planner,
        },
        validators={"finite_trajectory": _all_finite},
        initial_state={},
        runs=(
            FixtureRun(bindings(40)),
            FixtureRun(bindings(40)),
            FixtureRun(bindings(41)),
        ),
    )


def _all_finite(value: object) -> bool:
    if isinstance(value, tuple | list):
        return all(_all_finite(item) for item in value)
    return isinstance(value, int | float) and math.isfinite(value)


DRIVING_FIXTURES = (
    build_driving_trajectory_fixture,
    build_driving_ar_fixture,
    build_driving_diffusion_fixture,
    build_hybrid_external_feature_fixture,
)
