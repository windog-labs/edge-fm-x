from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _tool() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "run_high_memory_autovla.py"
    )


def _module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "run_high_memory_autovla",
        _tool(),
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_target_and_torch_arch_are_destination_specific() -> None:
    handoff = _module()
    assert handoff._target_from_capability(8, 0) == "sm_80"
    assert handoff._target_from_capability(9, 0) == "sm_90"
    assert handoff._torch_arch_list("sm_80") == "8.0"
    assert handoff._torch_arch_list("sm_90") == "9.0"
    with pytest.raises(ValueError, match="invalid CUDA target"):
        handoff._torch_arch_list("sm86")


def test_environment_template_separates_qwen_model_and_eval_configs() -> None:
    template = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "high_memory"
        / "autovla.env.example"
    ).read_text(encoding="utf-8")
    config_line = next(
        line
        for line in template.splitlines()
        if line.startswith("VLAFORGE_AUTOVLA_QWEN_CONFIG=")
    )
    assert config_line.endswith("/Qwen2.5-VL-3B-Instruct/config.json")
    assert "nusc-sft-eval.yaml" not in config_line


def test_print_plan_is_portable_and_does_not_probe_gpu(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        str(_tool()),
        "--source-root",
        str(tmp_path / "source"),
        "--checkpoint",
        str(tmp_path / "checkpoint.ckpt"),
        "--codebook",
        str(tmp_path / "codebook.pkl"),
        "--qwen-config",
        str(tmp_path / "config.yaml"),
        "--output-root",
        str(tmp_path / "output"),
        "--target",
        "sm_80",
        "--through",
        "partition-l3",
        "--print-plan",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["target"] == "sm_80"
    assert plan["torch_cuda_arch_list"] == "8.0"
    assert [stage["name"] for stage in plan["stages"]] == [
        "partition-l2",
        "partition-compile",
        "partition-l3",
    ]
    compile_command = plan["stages"][1]["command"]
    assert "--device" in compile_command
    assert "--inductor-profile" in compile_command
    frontend_command = plan["stages"][0]["command"]
    precision_index = frontend_command.index("--precision-mode") + 1
    assert frontend_command[precision_index] == "fp32-internal"
    audit_command = plan["stages"][2]["command"]
    expected_index = audit_command.index("--expected-target") + 1
    assert audit_command[expected_index] == "sm_80"
    assert "sm_86" not in json.dumps(plan)


def test_packaged_source_uses_pinned_content_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff = _module()
    for relative, expected in handoff.AUTOVLA_SOURCE_SHA256.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        monkeypatch.setitem(
            handoff.AUTOVLA_SOURCE_SHA256,
            relative,
            handoff._sha256(path),
        )
    record, errors = handoff._source_record(tmp_path)
    assert errors == []
    assert record["repository"]["git_checkout"] is False
    assert (
        record["repository"]["identity_mode"]
        == "pinned-content-sha256"
    )
