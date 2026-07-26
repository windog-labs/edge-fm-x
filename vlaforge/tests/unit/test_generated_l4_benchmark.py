from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "benchmark_generated_l4.py"
    )
    specification = importlib.util.spec_from_file_location(
        "benchmark_generated_l4",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_latency_summary_uses_nearest_rank_percentiles() -> None:
    benchmark = _module()
    result = benchmark._summary(list(range(1, 101)))

    assert result["count"] == 100
    assert result["mean_ns"] == pytest.approx(50.5)
    assert result["p50_ns"] == 50
    assert result["p90_ns"] == 90
    assert result["p99_ns"] == 99


def test_runner_output_parser_preserves_trace_and_samples() -> None:
    benchmark = _module()
    samples, summary = benchmark._parse_output(
        "\n".join(
            (
                "SAMPLE,0,100,7,1,0.25",
                "SAMPLE,1,90,8,1,0.5",
                "STATE_VERSIONS," + ",".join(
                    str(index) for index in range(32)
                ),
                "SUMMARY,10,11,19,20,21,22,29,30,31,32,0.75,"
                "2,3,4,5,6,0,6,1,7,8",
            )
        )
    )

    assert [item["latency_ns"] for item in samples] == [100, 90]
    assert summary["cache_hits"] == 3
    assert summary["cache_misses"] == 4
    assert summary["transaction_aborts"] == 0
    assert summary["state_1_version"] == 8
    assert summary["first_run_ns"] == 11
    assert summary["state_versions"] == list(range(32))


def test_diffusiondrive_same_revision_requires_exact_cache_hits() -> None:
    benchmark = _module()
    samples = [
        {
            "index": 0,
            "latency_ns": 100,
            "revision": 7,
            "revision_present": True,
            "output_probe": 0.25,
        }
    ]
    runtime = {
        "transaction_commits": 1,
        "transaction_aborts": 0,
        "output_commits": 1,
        "cache_hits": 1,
        "cache_misses": 0,
    }

    benchmark._validate_runtime(
        model="diffusiondrive",
        mode="same",
        sample_count=1,
        samples=samples,
        runtime=runtime,
    )

    runtime["cache_hits"] = 0
    with pytest.raises(RuntimeError, match="revision/cache"):
        benchmark._validate_runtime(
            model="diffusiondrive",
            mode="same",
            sample_count=1,
            samples=samples,
            runtime=runtime,
        )


def test_minddrive_requires_cache_and_all_state_commits() -> None:
    benchmark = _module()
    samples = [
        {
            "index": index,
            "latency_ns": 100,
            "revision": 1000,
            "revision_present": True,
            "output_probe": 0.25,
        }
        for index in range(3)
    ]
    runtime = {
        "transaction_commits": 3,
        "transaction_aborts": 0,
        "output_commits": 3,
        "cache_hits": 3,
        "cache_misses": 0,
        "state_commits": 48,
        "state_versions": [4] * 16 + [0] * 16,
    }
    benchmark._validate_runtime(
        model="minddrive",
        mode="same",
        sample_count=3,
        warmup=1,
        samples=samples,
        runtime=runtime,
    )

    runtime["state_commits"] = 47
    with pytest.raises(RuntimeError, match="16 authoritative"):
        benchmark._validate_runtime(
            model="minddrive",
            mode="same",
            sample_count=3,
            warmup=1,
            samples=samples,
            runtime=runtime,
        )


def test_minddrive_source_supports_real_frames_and_missing_revision(
    tmp_path: Path,
) -> None:
    benchmark = _module()
    runner = tmp_path / "runner.cpp"
    runner.write_text("int main() { return 0; }\n", encoding="utf-8")
    source = benchmark._minddrive_source(runner)

    assert '"frame_00400", "frame_00401", "frame_00402"' in source
    assert "bound.camera_images_stamp.has_revision = 0u;" in source
    assert "outputs.trajectory.tensor.data" in source
    assert "iteration % inputs.size()" in source


def test_bundle_memory_contract_separates_state_cache_and_arena(
    tmp_path: Path,
) -> None:
    benchmark = _module()
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "input_schema.json").write_text(
        """
        {"inputs":[
          {"payload":{"kind":"tensor","dtype":"f32","shape":[2,3]}}
        ]}
        """,
        encoding="utf-8",
    )
    (metadata / "output_schema.json").write_text(
        """
        {"outputs":[
          {"payload":{"kind":"scalar","name":"i64"}}
        ]}
        """,
        encoding="utf-8",
    )
    (metadata / "state_schema.json").write_text(
        """
        {"states":[
          {"payload":{"dtype":"f32","shape":[2,3]},"retention":2}
        ]}
        """,
        encoding="utf-8",
    )
    (metadata / "physical_memory_plan.json").write_text(
        """
        {
          "states":[{
            "offset":64,"slot_size_bytes":32,"slot_capacity":2
          }],
          "arena":{
            "size_bytes":1024,
            "physical_buffers":[
              {"buffer_class":"derived_cache","size_bytes":128},
              {"buffer_class":"ssa","size_bytes":256}
            ]
          }
        }
        """,
        encoding="utf-8",
    )

    contract = benchmark._bundle_memory_contract(tmp_path)
    assert contract["external_input_bytes_per_invocation"] == 24
    assert contract["external_output_bytes_per_invocation"] == 8
    assert contract["per_run_static_arena_bytes"] == 1024
    assert contract["authoritative_state_payload_bytes"] == 24
    assert contract["authoritative_state_retained_payload_bytes"] == 48
    assert contract["authoritative_state_arena_capacity_bytes"] == 128
    assert contract["derived_cache_physical_capacity_bytes"] == 128
