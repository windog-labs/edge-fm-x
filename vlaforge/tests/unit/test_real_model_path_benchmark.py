from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "benchmark_real_model_paths.py"
    )
    specification = importlib.util.spec_from_file_location(
        "benchmark_real_model_paths",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_latency_summary_uses_nearest_rank_percentiles() -> None:
    benchmark = _module()
    result = benchmark._latency_summary(list(range(1, 101)))

    assert result["count"] == 100
    assert result["mean_ns"] == pytest.approx(50.5)
    assert result["p50_ns"] == 50
    assert result["p90_ns"] == 90
    assert result["p99_ns"] == 99


def test_aoti_sequence_parser_preserves_bindings_and_liveness(
    tmp_path: Path,
) -> None:
    benchmark = _module()
    path = tmp_path / "region.vfseq"
    path.write_text(
        "\n".join(
            (
                "VLAFORGE_AOTI_SEQUENCE 1",
                "region test_region",
                "target sm_86",
                "device cuda:0",
                "values 3",
                "value 0 input 0 f32 1 4",
                "value 1 temporary -1 f32 1 4",
                "value 2 output 0 f32 1 4",
                "artifacts 2",
                "artifact 0 physical/first.so " + "a" * 64 + " 10",
                "artifact 1 physical/second.so " + "b" * 64 + " 20",
                "nodes 2",
                "node 0 1 0 1 1",
                "node 1 1 1 1 2",
                "end",
                "",
            )
        ),
        encoding="utf-8",
    )

    manifest = benchmark._parse_aoti_sequence(path)

    assert manifest.region == "test_region"
    assert manifest.target == "sm_86"
    assert manifest.input_count == 1
    assert manifest.output_count == 1
    assert manifest.nodes[0].inputs == (0,)
    assert manifest.nodes[1].outputs == (2,)


def test_first_run_can_be_counted_inside_stateful_warmup() -> None:
    benchmark = _module()

    class Scalar:
        def reshape(self, *_shape: object) -> "Scalar":
            return self

        def __getitem__(self, _index: int) -> "Scalar":
            return self

        def item(self) -> float:
            return 1.0

    class Cuda:
        @staticmethod
        def mem_get_info() -> tuple[int, int]:
            return 80, 100

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def memory_allocated() -> int:
            return 10

        @staticmethod
        def memory_reserved() -> int:
            return 20

        @staticmethod
        def reset_peak_memory_stats() -> None:
            return None

        @staticmethod
        def max_memory_allocated() -> int:
            return 11

        @staticmethod
        def max_memory_reserved() -> int:
            return 21

    class Torch:
        cuda = Cuda()

    calls = 0

    def run() -> Scalar:
        nonlocal calls
        calls += 1
        return Scalar()

    samples, _ = benchmark._benchmark(
        Torch(),
        run,
        warmup=3,
        samples=2,
        first_run_counts_as_warmup=True,
    )

    assert calls == 5
    assert len(samples) == 2


def test_minddrive_replays_only_the_stochastic_meta_action_token() -> None:
    torch = pytest.importorskip("torch")
    benchmark = _module()
    candidate = torch.tensor([1, 2, 151665])
    reference = torch.tensor([1, 2, 151671])

    replay = benchmark._minddrive_meta_action_replay(
        torch,
        candidate,
        reference,
        allowed_token_ids=set(range(151665, 151672)),
    )

    assert replay == {
        "sampled_token_id": 151665,
        "replayed_token_id": 151671,
        "token_offset": 2,
        "allowed_token_ids": list(range(151665, 151672)),
    }
    assert (
        benchmark._minddrive_meta_action_replay(
            torch,
            reference,
            reference,
            allowed_token_ids=set(range(151665, 151672)),
        )
        is None
    )
    with pytest.raises(ValueError, match="outside its stochastic"):
        benchmark._minddrive_meta_action_replay(
            torch,
            torch.tensor([9, 2, 151665]),
            reference,
            allowed_token_ids=set(range(151665, 151672)),
        )


def test_minddrive_projects_upstream_state_to_fixed_retention() -> None:
    torch = pytest.importorskip("torch")
    benchmark = _module()

    class Payload:
        shape = (1, 2, 3)

    class Head:
        memory = torch.arange(12).reshape(1, 4, 3)

    class Model:
        pts_bbox_head = Head()
        map_head = Head()

    result = benchmark._normalize_minddrive_upstream_state(
        Model(),
        (("map_memory", Payload()),),
        {"map_memory": "map.memory"},
    )

    assert result["map_memory"] == {
        "observed_shape": [1, 4, 3],
        "committed_shape": [1, 2, 3],
        "truncated": True,
    }
    assert tuple(Model.map_head.memory.shape) == (1, 2, 3)

    Model.map_head.memory = torch.zeros(2, 2, 3)
    with pytest.raises(RuntimeError, match="cannot project"):
        benchmark._normalize_minddrive_upstream_state(
            Model(),
            (("map_memory", Payload()),),
            {"map_memory": "map.memory"},
        )


def test_minddrive_clones_views_but_borrows_tensor_storage() -> None:
    torch = pytest.importorskip("torch")
    benchmark = _module()
    tensor = torch.arange(3).reshape(1, 3)
    original = {
        "outer": [[tensor]],
        "metadata": ({"scene": "one"},),
    }

    invocation = benchmark._clone_minddrive_invocation(original)
    invocation["outer"][0].append("mutated")
    invocation["metadata"][0]["scene"] = "two"

    assert original["outer"] == [[tensor]]
    assert original["metadata"][0]["scene"] == "one"
    borrowed = invocation["outer"][0][0]
    assert borrowed is not tensor
    assert borrowed.data_ptr() == tensor.data_ptr()
    borrowed.squeeze_(0)
    assert tuple(borrowed.shape) == (3,)
    assert tuple(tensor.shape) == (1, 3)
