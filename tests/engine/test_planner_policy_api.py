import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.edge_fm_build_paths import prepend_built_python_paths

prepend_built_python_paths(project_root)

import edge_fm


def _write_config(tmp_path: Path, payload: dict) -> str:
    path = tmp_path / "engine_config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _tensor_from_np(array: np.ndarray) -> edge_fm.Tensor:
    contiguous = np.ascontiguousarray(array)
    dtype = {
        np.dtype("float32"): edge_fm.DType.Float32,
        np.dtype("int32"): edge_fm.DType.Int32,
    }[contiguous.dtype]
    return edge_fm.Tensor(
        int(contiguous.ctypes.data),
        list(contiguous.shape),
        dtype,
        edge_fm.Device.CPU,
        0,
        True,
    )


def _dump_values(tmp_path: Path, tensor: edge_fm.Tensor, dtype=float):
    path = tmp_path / "tensor_dump.txt"
    tensor.dump(str(path))
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        values.extend(dtype(item) for item in line.split())
    return values


def test_planner_policy_api_surface():
    assert hasattr(edge_fm.EdgeFM, "plan")
    assert hasattr(edge_fm.EdgeFM, "run_stage")
    assert hasattr(edge_fm.EdgeFM, "last_plan_metrics")
    assert hasattr(edge_fm.EdgeFM, "last_stage_metrics")
    assert hasattr(edge_fm.EdgeFM, "prefill")
    assert hasattr(edge_fm.EdgeFM, "decode")


def test_token_generation_task_aliases_normalize_before_model_path_validation(tmp_path: Path):
    for task in ["token_generation", "text_generation", "vlm_generation"]:
        task_dir = tmp_path / task
        task_dir.mkdir()
        config_path = _write_config(
            task_dir,
            {
                "task": task,
                "model_name": "Qwen2.5",
                "runtime": {"device": "cpu"},
            },
        )
        try:
            edge_fm.EdgeFM(config_path)
        except Exception as exc:
            assert "prefill_model_path" in str(exc)
        else:
            raise AssertionError(f"expected token generation alias {task} to require a model path")


def test_stage_execution_prefill_matches_run_stage(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "stage_execution",
            "model_name": "trajectory_planner",
            "runtime": {"device": "cpu"},
            "stages": {
                "prefill": {
                    "backend": "mock",
                    "outputs": {
                        "prefix_state": {
                            "dtype": "float32",
                            "shape": [1, 2],
                            "values": [3.0, 4.0],
                        }
                    },
                }
            },
        },
    )

    engine = edge_fm.EdgeFM(config_path)
    prefill_outputs = engine.prefill(0, {})
    stage_outputs = engine.run_stage(0, "prefill", {})

    assert sorted(prefill_outputs) == ["prefix_state"]
    assert sorted(stage_outputs) == ["prefix_state"]
    assert _dump_values(tmp_path, prefill_outputs["prefix_state"]) == [3.0, 4.0]
    assert _dump_values(tmp_path, stage_outputs["prefix_state"]) == [3.0, 4.0]
    assert engine.last_stage_metrics()["stage_outputs"] == 1.0


def test_stage_execution_uses_request_local_state_with_explicit_override(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "stage_execution",
            "model_name": "trajectory_planner",
            "runtime": {"device": "cpu"},
            "stages": {
                "context": {
                    "backend": "mock",
                    "outputs": {
                        "planner_state": {
                            "dtype": "float32",
                            "shape": [1, 2],
                            "values": [5.0, 6.0],
                        }
                    },
                },
                "consume": {
                    "backend": "mock",
                    "outputs": {
                        "used_state": {
                            "source": "planner_state",
                        }
                    },
                },
            },
        },
    )
    engine = edge_fm.EdgeFM(config_path)

    engine.run_stage(10, "context", {})
    cached_outputs = engine.run_stage(10, "consume", {})
    explicit_state = _tensor_from_np(np.array([[9.0, 10.0]], dtype=np.float32))
    explicit_outputs = engine.run_stage(10, "consume", {"planner_state": explicit_state})

    assert _dump_values(tmp_path, cached_outputs["used_state"]) == [5.0, 6.0]
    assert _dump_values(tmp_path, explicit_outputs["used_state"]) == [9.0, 10.0]
    try:
        engine.run_stage(11, "consume", {})
    except Exception as exc:
        assert "planner_state" in str(exc)
    else:
        raise AssertionError("expected request-local planner state to be isolated")


def test_stage_execution_uses_stage_defaults_after_cache(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "stage_execution",
            "model_name": "trajectory_planner",
            "runtime": {"device": "cpu"},
            "stages": {
                "consume": {
                    "backend": "mock",
                    "defaults": {
                        "planner_state": {
                            "dtype": "float32",
                            "shape": [1, 2],
                            "values": [1.5, 2.5],
                        }
                    },
                    "outputs": {
                        "used_state": {
                            "source": "planner_state",
                        }
                    },
                },
                "context": {
                    "backend": "mock",
                    "outputs": {
                        "planner_state": {
                            "dtype": "float32",
                            "shape": [1, 2],
                            "values": [5.0, 6.0],
                        }
                    },
                },
            },
        },
    )
    engine = edge_fm.EdgeFM(config_path)

    default_outputs = engine.run_stage(20, "consume", {})
    engine.run_stage(20, "context", {})
    cached_outputs = engine.run_stage(20, "consume", {})

    assert _dump_values(tmp_path, default_outputs["used_state"]) == [1.5, 2.5]
    assert _dump_values(tmp_path, cached_outputs["used_state"]) == [5.0, 6.0]


def test_stage_execution_mock_supports_common_integer_tensors(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "stage_execution",
            "model_name": "trajectory_planner",
            "runtime": {"device": "cpu"},
            "stages": {
                "mask": {
                    "backend": "mock",
                    "defaults": {
                        "attention_mask": {
                            "dtype": "uint8",
                            "shape": [1, 3],
                            "values": [1, 0, 1],
                        },
                        "position_ids": {
                            "dtype": "int64",
                            "shape": [1, 3],
                            "values": [7, 8, 9],
                        },
                    },
                    "outputs": {
                        "attention_mask": {
                            "source": "attention_mask",
                        },
                        "position_ids": {
                            "source": "position_ids",
                        },
                    },
                }
            },
        },
    )

    outputs = edge_fm.EdgeFM(config_path).run_stage(0, "mask", {})

    assert outputs["attention_mask"].dtype() == edge_fm.DType.UInt8
    assert outputs["position_ids"].dtype() == edge_fm.DType.Int64
    assert _dump_values(tmp_path, outputs["attention_mask"], int) == [1, 0, 1]
    assert _dump_values(tmp_path, outputs["position_ids"], int) == [7, 8, 9]


def test_single_stage_planner_returns_trajectory(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "trajectory_planning",
            "model_name": "trajectory_planner",
            "runtime": {"device": "cpu"},
            "planner": {
                "kind": "single_stage",
                "stage": "plan",
                "output_tensor": "trajectory",
            },
            "stages": {
                "plan": {
                    "backend": "mock",
                    "outputs": {
                        "trajectory": {
                            "dtype": "float32",
                            "shape": [1, 2, 2],
                            "values": [1.0, 2.0, 3.0, 4.0],
                        }
                    },
                }
            },
        },
    )

    outputs = edge_fm.EdgeFM(config_path).plan(7, {})

    assert sorted(outputs) == ["trajectory"]
    assert outputs["trajectory"].shape() == [1, 2, 2]
    assert _dump_values(tmp_path, outputs["trajectory"]) == [1.0, 2.0, 3.0, 4.0]


def test_planner_model_name_without_task_infers_trajectory_planning(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "model_name": "sparsedrive_v2",
            "runtime": {"device": "cpu"},
            "planner": {
                "kind": "single_stage",
                "stage": "plan",
                "output_tensor": "trajectory",
            },
            "stages": {
                "plan": {
                    "backend": "mock",
                    "outputs": {
                        "trajectory": {
                            "dtype": "float32",
                            "shape": [1, 1, 2],
                            "values": [8.0, 9.0],
                        }
                    },
                }
            },
        },
    )

    outputs = edge_fm.EdgeFM(config_path).plan(0, {})

    assert _dump_values(tmp_path, outputs["trajectory"]) == [8.0, 9.0]


def test_candidate_scoring_planner_selects_argmax(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "trajectory_planning",
            "model_name": "sparsedrive_v2",
            "runtime": {"device": "cpu"},
            "planner": {
                "kind": "candidate_scoring",
                "stage": "score",
                "sampler": "argmax",
                "candidate_tensor": "candidate_trajectories",
                "score_tensor": "candidate_scores",
            },
            "stages": {
                "score": {
                    "backend": "mock",
                    "outputs": {
                        "candidate_trajectories": {
                            "dtype": "float32",
                            "shape": [1, 3, 2, 2],
                            "values": [
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                                10.0,
                                10.0,
                                10.0,
                                11.0,
                                20.0,
                                20.0,
                                20.0,
                                21.0,
                            ],
                        },
                        "candidate_scores": {
                            "dtype": "float32",
                            "shape": [1, 3],
                            "values": [0.1, 0.9, 0.2],
                        },
                    },
                }
            },
        },
    )

    outputs = edge_fm.EdgeFM(config_path).plan(0, {})

    assert outputs["trajectory"].shape() == [1, 2, 2]
    assert outputs["selected_index"].shape() == [1]
    assert _dump_values(tmp_path, outputs["trajectory"]) == [10.0, 10.0, 10.0, 11.0]
    assert _dump_values(tmp_path, outputs["selected_index"], int) == [1]


def test_planner_method_scoring_alias_selects_argmax(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "trajectory_planning",
            "model_name": "trajectory_planner",
            "runtime": {"device": "cpu"},
            "planner": {
                "method": "scoring",
                "stage": "score",
            },
            "stages": {
                "score": {
                    "backend": "mock",
                    "outputs": {
                        "candidate_trajectories": {
                            "dtype": "float32",
                            "shape": [1, 2, 1, 2],
                            "values": [1.0, 2.0, 7.0, 8.0],
                        },
                        "candidate_scores": {
                            "dtype": "float32",
                            "shape": [1, 2],
                            "values": [-1.0, 3.0],
                        },
                    },
                }
            },
        },
    )

    outputs = edge_fm.EdgeFM(config_path).plan(0, {})

    assert _dump_values(tmp_path, outputs["trajectory"]) == [7.0, 8.0]
    assert _dump_values(tmp_path, outputs["selected_index"], int) == [1]


def test_candidate_scoring_selects_argmax_per_batch(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "trajectory_planning",
            "model_name": "sparsedrive_v2",
            "runtime": {"device": "cpu"},
            "planner": {
                "kind": "candidate_scoring",
                "stage": "score",
            },
            "stages": {
                "score": {
                    "backend": "mock",
                    "outputs": {
                        "candidate_trajectories": {
                            "dtype": "float32",
                            "shape": [2, 2, 1, 2],
                            "values": [
                                1.0,
                                2.0,
                                3.0,
                                4.0,
                                5.0,
                                6.0,
                                7.0,
                                8.0,
                            ],
                        },
                        "candidate_scores": {
                            "dtype": "float32",
                            "shape": [2, 2],
                            "values": [0.1, 0.9, 4.0, 2.0],
                        },
                    },
                }
            },
        },
    )

    outputs = edge_fm.EdgeFM(config_path).plan(0, {})

    assert outputs["trajectory"].shape() == [2, 1, 2]
    assert _dump_values(tmp_path, outputs["selected_index"], int) == [1, 0]
    assert _dump_values(tmp_path, outputs["trajectory"]) == [3.0, 4.0, 5.0, 6.0]


def test_candidate_scoring_rejects_empty_candidate_axis(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "trajectory_planning",
            "model_name": "sparsedrive_v2",
            "runtime": {"device": "cpu"},
            "planner": {
                "kind": "candidate_scoring",
                "stage": "score",
            },
            "stages": {
                "score": {
                    "backend": "mock",
                    "outputs": {
                        "candidate_trajectories": {
                            "dtype": "float32",
                            "shape": [1, 0, 1, 2],
                            "values": [],
                        },
                        "candidate_scores": {
                            "dtype": "float32",
                            "shape": [1, 0],
                            "values": [],
                        },
                    },
                }
            },
        },
    )

    try:
        edge_fm.EdgeFM(config_path).plan(0, {})
    except Exception as exc:
        assert "at least one candidate" in str(exc)
    else:
        raise AssertionError("expected empty candidate axis to be rejected")


def test_iterative_euler_flow_planner_updates_state(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "trajectory_planning",
            "model_name": "lingxi_sparsedrive_planner",
            "runtime": {"device": "cpu"},
            "planner": {
                "kind": "iterative_denoise",
                "method": "flow",
                "sampler": "euler_flow",
                "num_steps": 3,
                "state_tensor": "current_actions",
                "step_stage": "step",
                "step_output_tensor": "velocity",
                "output_tensor": "trajectory",
            },
            "stages": {
                "step": {
                    "backend": "mock",
                    "outputs": {
                        "velocity": {
                            "dtype": "float32",
                            "shape": [1, 2, 2],
                            "values": [3.0, 6.0, 9.0, 12.0],
                        }
                    },
                }
            },
        },
    )
    init_actions = _tensor_from_np(np.zeros((1, 2, 2), dtype=np.float32))

    engine = edge_fm.EdgeFM(config_path)
    outputs = engine.plan(0, {"current_actions": init_actions})

    assert outputs["trajectory"].shape() == [1, 2, 2]
    assert _dump_values(tmp_path, outputs["trajectory"]) == [3.0, 6.0, 9.0, 12.0]
    metrics = engine.last_plan_metrics()
    assert metrics["plan_steps"] == 3.0


def test_iterative_denoise_uses_context_stage_cache(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "trajectory_planning",
            "model_name": "trajectory_planner",
            "runtime": {"device": "cpu"},
            "planner": {
                "kind": "iterative_denoise",
                "sampler": "euler_flow",
                "num_steps": 1,
                "context_stage": "context",
                "state_tensor": "current_actions",
                "step_stage": "step",
                "step_output_tensor": "velocity",
            },
            "stages": {
                "context": {
                    "backend": "mock",
                    "outputs": {
                        "cached_velocity": {
                            "dtype": "float32",
                            "shape": [1, 1, 2],
                            "values": [10.0, 20.0],
                        }
                    },
                },
                "step": {
                    "backend": "mock",
                    "outputs": {
                        "velocity": {
                            "source": "cached_velocity",
                        }
                    },
                },
            },
        },
    )
    init_actions = _tensor_from_np(np.ones((1, 1, 2), dtype=np.float32))

    outputs = edge_fm.EdgeFM(config_path).plan(0, {"current_actions": init_actions})

    assert _dump_values(tmp_path, outputs["trajectory"]) == [11.0, 21.0]


def test_flow_matching_kind_alias_runs_iterative_denoise(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "trajectory_planning",
            "model_name": "trajectory_planner",
            "runtime": {"device": "cpu"},
            "planner": {
                "kind": "flow_matching",
                "sampler": "euler_flow",
                "num_steps": 2,
                "state_tensor": "current_actions",
                "step_stage": "step",
                "step_output_tensor": "velocity",
                "output_tensor": "trajectory",
            },
            "stages": {
                "step": {
                    "backend": "mock",
                    "outputs": {
                        "velocity": {
                            "dtype": "float32",
                            "shape": [1, 1, 2],
                            "values": [2.0, 4.0],
                        }
                    },
                }
            },
        },
    )
    init_actions = _tensor_from_np(np.zeros((1, 1, 2), dtype=np.float32))

    outputs = edge_fm.EdgeFM(config_path).plan(0, {"current_actions": init_actions})

    assert _dump_values(tmp_path, outputs["trajectory"]) == [2.0, 4.0]


def test_iterative_ddim_initializes_state_from_trajectory_shape(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "trajectory_planning",
            "model_name": "trajectory_planner",
            "runtime": {"device": "cpu"},
            "planner": {
                "kind": "iterative_denoise",
                "sampler": "ddim",
                "num_steps": 2,
                "trajectory_shape": [1, 1, 2],
                "noise_sigma": 0.0,
                "state_tensor": "current_actions",
                "step_stage": "step",
                "step_output_tensor": "x0_pred",
            },
            "stages": {
                "step": {
                    "backend": "mock",
                    "outputs": {
                        "x0_pred": {
                            "dtype": "float32",
                            "shape": [1, 1, 2],
                            "values": [4.0, 5.0],
                        }
                    },
                }
            },
        },
    )
    engine = edge_fm.EdgeFM(config_path)

    outputs = engine.plan(0, {})

    assert _dump_values(tmp_path, outputs["trajectory"]) == [4.0, 5.0]
    assert engine.last_plan_metrics()["plan_steps"] == 2.0


def test_iterative_noise_initialization_is_deterministic(tmp_path: Path):
    payload = {
        "task": "trajectory_planning",
        "model_name": "trajectory_planner",
        "runtime": {"device": "cpu"},
        "planner": {
            "kind": "iterative_denoise",
            "sampler": "ddim",
            "num_steps": 1,
            "trajectory_shape": [1, 1, 4],
            "noise_sigma": 0.5,
            "seed": 123,
            "state_tensor": "current_actions",
            "step_stage": "step",
            "step_output_tensor": "current_actions",
        },
        "stages": {
            "step": {
                "backend": "mock",
                "outputs": {
                    "current_actions": {
                        "source": "current_actions",
                    }
                },
            }
        },
    }
    config_path = _write_config(tmp_path, payload)

    first = edge_fm.EdgeFM(config_path).plan(0, {})["trajectory"]
    second = edge_fm.EdgeFM(config_path).plan(0, {})["trajectory"]

    assert _dump_values(tmp_path, first) == _dump_values(tmp_path, second)
    assert _dump_values(tmp_path, first) != [0.0, 0.0, 0.0, 0.0]


def test_iterative_denoise_passes_configured_timestep(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {
            "task": "trajectory_planning",
            "model_name": "trajectory_planner",
            "runtime": {"device": "cpu"},
            "planner": {
                "kind": "iterative_denoise",
                "sampler": "ddim",
                "num_steps": 3,
                "trajectory_shape": [1],
                "noise_sigma": 0.0,
                "timestep_start": 2.0,
                "timestep_end": 4.0,
                "timestep_tensor": "planner_t",
                "step_stage": "step",
                "step_output_tensor": "planner_t",
            },
            "stages": {
                "step": {
                    "backend": "mock",
                    "outputs": {
                        "planner_t": {
                            "source": "planner_t",
                        }
                    },
                }
            },
        },
    )

    outputs = edge_fm.EdgeFM(config_path).plan(0, {})

    assert _dump_values(tmp_path, outputs["trajectory"]) == [4.0]
