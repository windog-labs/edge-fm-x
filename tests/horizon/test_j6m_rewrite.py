import json
import subprocess
import sys
from pathlib import Path

import pytest

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
        "stages": [
            {
                "name": "prefill",
                "artifact_path": str(tmp_path / "smolvla_prefill.hbm"),
                "factory_kwargs": {"stage": "prefill"},
                "inputs": [
                    {"name": "prefix_embeds", "shape": [1, 6, 8], "dtype": "float32"},
                    {"name": "prefix_attention_mask", "shape": [1, 6, 6], "dtype": "uint8"},
                    {"name": "prefix_position_ids", "shape": [1, 6], "dtype": "int32"},
                ],
                "outputs": [
                    {"name": "prefix_kv_layer_0", "shape": [2, 6, 1, 8], "dtype": "float32"},
                ],
                "kv_layout": "packed_layer_kv_v1",
            },
            {
                "name": "decode",
                "artifact_path": str(tmp_path / "smolvla_decode.hbm"),
                "factory_kwargs": {"stage": "decode"},
                "inputs": [
                    {"name": "suffix_embeds", "shape": [1, 4, 6], "dtype": "float32"},
                    {"name": "denoise_attention_mask", "shape": [1, 4, 10], "dtype": "uint8"},
                    {"name": "suffix_position_ids", "shape": [1, 4], "dtype": "int32"},
                    {"name": "prefix_kv_layer_0", "shape": [2, 6, 1, 8], "dtype": "float32"},
                ],
                "outputs": [
                    {"name": "expert_hidden", "shape": [1, 4, 6], "dtype": "float32"},
                ],
                "kv_layout": "packed_layer_kv_v1",
            },
        ],
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


def _write_onnx_compile_spec(tmp_path: Path) -> Path:
    module_path = tmp_path / "horizon_model_for_onnx.py"
    module_path.write_text(
        "import torch\n\n"
        "class DummyModel(torch.nn.Module):\n"
        "    def forward(self, suffix_embeds, denoise_attention_mask, suffix_position_ids, prefix_kv_layer_0):\n"
        "        keep = denoise_attention_mask.to(torch.float32).sum() * 0.0\n"
        "        keep = keep + suffix_position_ids.to(torch.float32).sum() * 0.0\n"
        "        keep = keep + prefix_kv_layer_0.sum() * 0.0\n"
        "        return suffix_embeds + keep\n\n"
        "def build_model(stage='decode'):\n"
        "    return DummyModel()\n",
        encoding="utf-8",
    )
    compile_spec_path = _write_compile_spec(tmp_path)
    compile_spec = json.loads(compile_spec_path.read_text(encoding="utf-8"))
    compile_spec["generated_module"]["module_path"] = str(module_path)
    compile_spec["compile_entry"]["module_path"] = str(module_path)
    compile_spec_path.write_text(json.dumps(compile_spec, indent=2), encoding="utf-8")
    return compile_spec_path


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


def test_compile_helper_uses_smolvla_stage_io(tmp_path: Path):
    compile_spec_path = _write_compile_spec(tmp_path)
    helper = project_root / "scripts" / "horizon" / "compile_horizon_from_spec.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(helper),
            str(compile_spec_path),
            "--dry-run",
            "--horizon-rewrite",
            "off",
            "--skip-model-init",
            "--stage",
            "decode",
        ],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["artifact_path"].endswith("smolvla_decode.hbm")

    prep_manifest = json.loads((tmp_path / "compile_prep.json").read_text(encoding="utf-8"))
    assert prep_manifest["stage"] == "decode"
    assert prep_manifest["example_inputs"]["kv_layout"] == "packed_layer_kv_v1"
    assert prep_manifest["example_inputs"]["inputs"]["suffix_embeds"] == [1, 4, 6]
    assert prep_manifest["example_inputs"]["inputs"]["prefix_kv_layer_0"] == [2, 6, 1, 8]
    assert prep_manifest["example_inputs"]["outputs"]["expert_hidden"] == [1, 4, 6]


def test_compile_helper_rejects_legacy_smolvla_stage_name(tmp_path: Path):
    compile_spec_path = _write_compile_spec(tmp_path)
    helper = project_root / "scripts" / "horizon" / "compile_horizon_from_spec.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(helper),
            str(compile_spec_path),
            "--dry-run",
            "--skip-model-init",
            "--stage",
            "expert_denoise",
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "invalid choice" in completed.stderr
    assert "decode" in completed.stderr


def test_compile_helper_exports_stage_onnx(tmp_path: Path):
    pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")

    compile_spec_path = _write_onnx_compile_spec(tmp_path)
    helper = project_root / "scripts" / "horizon" / "compile_horizon_from_spec.py"
    onnx_path = tmp_path / "decode.onnx"

    completed = subprocess.run(
        [
            sys.executable,
            str(helper),
            str(compile_spec_path),
            "--stage",
            "decode",
            "--horizon-rewrite",
            "off",
            "--export-onnx",
            "--onnx-path",
            str(onnx_path),
        ],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "prepared"
    assert payload["onnx_path"] == str(onnx_path)
    assert onnx_path.exists()

    model = onnx.load(str(onnx_path))
    assert model.ir_version <= 9
    assert [item.name for item in model.graph.input] == [
        "suffix_embeds",
        "denoise_attention_mask",
        "suffix_position_ids",
        "prefix_kv_layer_0",
    ]
    assert [item.name for item in model.graph.output] == ["expert_hidden"]
