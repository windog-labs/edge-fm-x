from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _assembler_module() -> object:
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "tools" / "assemble_real_openvla_l4.py"
    specification = importlib.util.spec_from_file_location(
        "assemble_real_openvla_l4",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_stable_openvla_artifact_maps_wrapper_and_runtime_files(
    tmp_path: Path,
) -> None:
    assembler = _assembler_module()
    package = tmp_path / "artifacts" / "decode.pt2"
    package.parent.mkdir()
    package.write_bytes(b"verified package")
    digest = assembler._sha256(package)
    runtime = (
        tmp_path
        / "extracted_artifacts"
        / "decode"
        / "decode"
        / "data"
        / "aotinductor"
        / "model"
    )
    runtime.mkdir(parents=True)
    wrapper = runtime / "model.wrapper.so"
    wrapper.write_bytes(b"shared library")
    cubin = runtime / "kernel.cubin"
    cubin.write_bytes(b"cuda binary")
    runtime.parents[3].joinpath(".package-sha256").write_text(
        digest + "\n",
        encoding="utf-8",
    )

    source, relative, auxiliary, record = (
        assembler._stable_model_artifact(
            l3_root=tmp_path,
            name="decode",
            package_path=package,
        )
    )

    assert source == wrapper
    assert relative == "artifacts/decode/model.wrapper.so"
    assert auxiliary == {"artifacts/decode/kernel.cubin": cubin}
    assert record["package_sha256"] == digest
    assert record["runtime_file_count"] == 1


@pytest.mark.cuda_aoti
@pytest.mark.real_model
@pytest.mark.skipif(
    os.environ.get("VLAFORGE_RUN_REAL_OPENVLA_L4") != "1",
    reason="set VLAFORGE_RUN_REAL_OPENVLA_L4=1 for real L4 audit",
)
def test_real_openvla_generated_cuda_session(tmp_path: Path) -> None:
    capture_root = os.getenv("VLAFORGE_OPENVLA_CAPTURE_ROOT")
    l3_root = os.getenv("VLAFORGE_OPENVLA_L3_ROOT")
    support_root = os.getenv("VLAFORGE_OPENVLA_L4_SUPPORT_ROOT")
    if not capture_root or not l3_root:
        pytest.skip(
            "set VLAFORGE_OPENVLA_CAPTURE_ROOT and "
            "VLAFORGE_OPENVLA_L3_ROOT"
        )

    project_root = Path(__file__).resolve().parents[2]
    repository_root = project_root.parent
    l3_report = Path(
        os.getenv(
            "VLAFORGE_OPENVLA_L3_REPORT",
            str(
                repository_root
                / "doc"
                / "reports"
                / "vlaforge_real_v03"
                / "openvla_artifact_l3.json"
            ),
        )
    )
    report = tmp_path / "openvla-real-l4.json"
    subprocess.run(
        [
            sys.executable,
            str(project_root / "tools" / "assemble_real_openvla_l4.py"),
            "--capture-root",
            capture_root,
            "--l3-root",
            l3_root,
            "--l3-report",
            str(l3_report),
            "--support-root",
            support_root or str(tmp_path / "support"),
            "--bundle-root",
            str(tmp_path / "bundle"),
            "--input-root",
            str(tmp_path / "inputs"),
            "--report",
            str(report),
            "--python",
            sys.executable,
            "--target",
            "sm_86",
            "--runs",
            "1",
        ],
        check=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "L4"
    assert payload["semantic_ir"]["core_op_delta"] == 0
    assert payload["semantic_ir"]["persistent_state_slots"] == 0
    assert payload["artifacts"]["invocation_resident"] == 36
    assert payload["artifacts"]["session_resident"] == 2
    assert payload["correctness"]["typed_generic_equal"]
    assert payload["correctness"]["action_maximum_absolute_error"] <= 1e-12
    assert payload["transaction"]["previous_output_preserved"]
    assert payload["transaction"]["transaction_abort_delta"] == 1
    assert payload["transaction"]["recovered"]
    assert payload["bundle"]["invalid_python_environment"]
    assert payload["bundle"]["links_libpython"] is False
