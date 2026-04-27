import json
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.horizon.j6m_rewrite import (
    prepare_j6m_rewrite_artifacts,
    should_enable_j6m_rewrite,
)


def _write_compile_spec(tmp_path: Path, model_name: str = "smolvla") -> Path:
    module_path = tmp_path / "horizon_model.py"
    module_path.write_text(
        "class DummyModel:\n"
        "    pass\n\n"
        "def build_model(stage='prefill'):\n"
        "    return DummyModel()\n",
        encoding="utf-8",
    )
    compile_spec = {
        "schema": "edgefm_horizon_compile_spec_v2",
        "backend": "horizon",
        "model_name": model_name,
        "model_variant": model_name,
        "model_config": {
            "num_steps": 3,
            "chunk_size": 4,
            "max_action_dim": 5,
        },
        "graph_tuning": {
            "target_hw_constraints": {
                "backend_target": "horizon",
                "runtime_device": "horizon",
                "resolved_hw_profile": "j6m",
            }
        },
        "horizon_rewrite": {
            "enabled": True,
            "smolvla": {
                "num_steps": 3,
                "chunk_size": 4,
                "max_action_dim": 5,
            },
        },
        "generated_module": {
            "module_path": str(module_path),
            "module_name": "edgefm_horizon_smolvla_model",
            "model_class": "DummyModel",
            "factory_function": "build_model",
        },
        "compile_entry": {
            "module_path": str(module_path),
            "factory_function": "build_model",
            "default_kwargs": {"stage": "prefill"},
        },
        "artifact": {
            "backend": "horizon",
            "artifact_type": "hbm",
            "artifact_path": str(tmp_path / "model.hbm"),
            "manifest_path": str(tmp_path / "compile_spec.json"),
            "metadata": {},
        },
        "engine_config": {
            "runtime": {"device": "horizon", "hw_profile": "j6m"},
            "kvcache": {"requests": [{"request_id": 0, "prefix_token_ids": [1, 2], "max_tokens": 8}]},
        },
    }
    path = tmp_path / "compile_spec.json"
    path.write_text(json.dumps(compile_spec, indent=2), encoding="utf-8")
    return path


def test_should_enable_j6m_rewrite_modes():
    assert not should_enable_j6m_rewrite(
        {"model_name": "qwen2_5", "engine_config": {"runtime": {"hw_profile": "cuda_sm80"}}},
        "auto",
    )
    assert should_enable_j6m_rewrite(
        {"model_name": "smolvla", "engine_config": {"runtime": {"device": "horizon"}}},
        "auto",
    )
    assert should_enable_j6m_rewrite({"model_name": "qwen2_5"}, "on")
    assert not should_enable_j6m_rewrite(
        {"model_name": "smolvla", "horizon_rewrite": {"enabled": True}},
        "off",
    )


def test_prepare_j6m_rewrite_artifacts_writes_manifest(tmp_path: Path):
    compile_spec_path = _write_compile_spec(tmp_path)
    compile_spec = json.loads(compile_spec_path.read_text(encoding="utf-8"))

    result = prepare_j6m_rewrite_artifacts(
        compile_spec_path=compile_spec_path,
        compile_spec=compile_spec,
        output_dir=tmp_path,
        stage="prefill",
        lerobot_root=tmp_path / "lerobot",
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    rewrite_ids = {entry["id"] for entry in manifest["operator_rewrites"]}
    assert result["enabled"] is True
    assert "smolvla_make_att_2d_masks_int16_cumsum" in rewrite_ids
    assert "smolvla_eager_attention_bounded_mask_fill" in rewrite_ids
    assert "smolvla_flow_matching_loop_bins" in rewrite_ids
    assert manifest["target"]["public_tensor_policy"] == "copy_in_copy_out"

    scale_config = json.loads(Path(result["scale_config_path"]).read_text(encoding="utf-8"))
    assert scale_config["int16_range"] == [-32768, 32767]
    assert scale_config["mask_fill_value"] == -32760.0

    flow_plan = json.loads(Path(result["flow_plan_path"]).read_text(encoding="utf-8"))
    assert flow_plan["enabled"] is True
    assert flow_plan["num_steps"] == 3
    assert flow_plan["logical_shapes"]["x_t"] == ["batch", 4, 5]


def test_compile_helper_dry_run_records_j6m_rewrite(tmp_path: Path):
    compile_spec_path = _write_compile_spec(tmp_path)
    helper = project_root / "scripts" / "horizon" / "compile_horizon_from_spec.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(helper),
            str(compile_spec_path),
            "--dry-run",
            "--horizon-rewrite",
            "on",
            "--skip-model-init",
            "--lerobot-root",
            str(tmp_path / "lerobot"),
        ],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "prepared"
    assert payload["horizon_rewrite"]["enabled"] is True

    prep_manifest = json.loads((tmp_path / "compile_prep.json").read_text(encoding="utf-8"))
    assert prep_manifest["horizon_rewrite"]["enabled"] is True
    assert Path(prep_manifest["horizon_rewrite"]["manifest_path"]).exists()

    model_summary = json.loads((tmp_path / "model_summary.json").read_text(encoding="utf-8"))
    assert model_summary["model_init_skipped"] is True
