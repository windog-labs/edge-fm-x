from pathlib import Path

from scripts.profile.owner_a_phase3_decision_gates import (
    build_owner_a_phase3_decision_report,
    format_owner_a_phase3_decision_report,
)


def test_owner_a_phase3_decision_report_keeps_runtime_paths_default_off(tmp_path):
    flashinfer_deepgemm = tmp_path / "third_party" / "flashinfer" / "flashinfer" / "deep_gemm.py"
    flashinfer_deepgemm.parent.mkdir(parents=True)
    flashinfer_deepgemm.write_text("# probe fixture\n", encoding="utf-8")

    report = build_owner_a_phase3_decision_report(
        project_root=tmp_path,
        cuda_capability=(8, 6),
        flashinfer_importable=False,
    )

    assert report["lm_head_top1"]["default_runtime"] == "full_logits"
    assert report["lm_head_top1"]["integration_status"] == "experimental_default_off"
    assert report["lm_head_top1"]["runtime_flag"] == "runtime.lm_head_top1.enabled"
    assert report["deepgemm"]["source_present"] is True
    assert report["deepgemm"]["runtime_default"] == "disabled"
    assert report["deepgemm"]["eligible_scope"] == "prefill_dense_linear_or_fp8_w8a8"
    assert report["deepgemm"]["runtime_binding_ready"] is False
    assert report["deepgemm"]["sm"] == 86
    assert report["deepgemm"]["shape_support_matrix"]
    assert all(
        item["default_runtime_safe"] is False
        for item in report["deepgemm"]["shape_support_matrix"]
    )
    assert report["prefix_kv"]["implementation_status"] == "implemented"
    assert "Continuous per-request/per-layer KV slot" in report["prefix_kv"]["runtime_contract"]
    assert report["prefix_kv"]["coverage"]
    assert report["int8_kv"]["implementation_status"] == "deferred"


def test_owner_a_phase3_decision_report_marks_missing_deepgemm_source(tmp_path):
    report = build_owner_a_phase3_decision_report(
        project_root=tmp_path,
        cuda_capability=None,
        flashinfer_importable=False,
    )

    assert report["deepgemm"]["source_present"] is False
    assert report["deepgemm"]["hardware_supported"] is False
    assert report["deepgemm"]["runtime_binding_ready"] is False
    assert any(
        "source is missing" in blocker
        for item in report["deepgemm"]["shape_support_matrix"]
        for blocker in item["blockers"]
    )
    assert "DeepGEMM" in format_owner_a_phase3_decision_report(report)


def test_owner_a_phase3_decision_report_keeps_deepgemm_default_off_on_supported_sm(tmp_path):
    flashinfer_deepgemm = tmp_path / "third_party" / "flashinfer" / "flashinfer" / "deep_gemm.py"
    flashinfer_deepgemm.parent.mkdir(parents=True)
    flashinfer_deepgemm.write_text(
        "@supported_compute_capability([100, 103])\n"
        "def m_grouped_fp8_gemm_nt_contiguous():\n"
        "    pass\n",
        encoding="utf-8",
    )

    report = build_owner_a_phase3_decision_report(
        project_root=tmp_path,
        cuda_capability=(10, 0),
        flashinfer_importable=True,
    )

    assert report["deepgemm"]["hardware_supported"] is True
    assert report["deepgemm"]["flashinfer_adapter_hardware_supported"] is True
    assert report["deepgemm"]["runtime_binding_ready"] is False
    prefill_rows = [
        item for item in report["deepgemm"]["shape_support_matrix"]
        if item["stage"] == "prefill"
    ]
    assert prefill_rows
    assert all(item["local_flashinfer_adapter_supported"] is True for item in prefill_rows)
    assert all(item["default_runtime_safe"] is False for item in prefill_rows)
