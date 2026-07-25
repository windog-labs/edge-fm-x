import os
from pathlib import Path

import pytest

from vlaforge.adapters.openvla_real import (
    RealOpenVLAConfig,
    build_real_openvla_action_program,
    run_real_openvla,
)
from vlaforge.analysis import verify


def test_real_openvla_source_contract_uses_invocation_ir_v02():
    module = build_real_openvla_action_program(
        action_dim=7,
        token_length=19,
        device="cpu",
    )
    assert module.schema_version == "0.2"
    assert module.states == ()
    assert [item.name for item in module.outputs] == ["action"]
    assert verify(module, raise_on_error=False) == ()


@pytest.mark.real_model
def test_real_openvla_checkpoint_matches_ir(tmp_path):
    checkpoint = os.getenv("VLAFORGE_OPENVLA_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set VLAFORGE_OPENVLA_CHECKPOINT")

    evidence = run_real_openvla(
        RealOpenVLAConfig(
            checkpoint_path=Path(checkpoint),
            revision=os.getenv("VLAFORGE_OPENVLA_REVISION", "unknown"),
            device=os.getenv("VLAFORGE_MODEL_DEVICE", "cuda:0"),
            unnorm_key=os.getenv("VLAFORGE_OPENVLA_UNNORM_KEY", "bridge_orig"),
            tolerance=1e-6,
            load_in_4bit=os.getenv("VLAFORGE_OPENVLA_LOAD_IN_4BIT", "1") == "1",
        ),
        trace_path=tmp_path / "trace.json",
    )

    assert evidence.evidence_kind == "real_checkpoint"
    assert evidence.schema == "vlaforge.real_model_evidence/0.2"
    assert evidence.checkpoint_revision != "unknown"
    assert evidence.input_fixture["name"] == "rgb_coordinate_grid_v1"
    assert evidence.input_fixture["source_image_shape"] == [224, 224, 3]
    assert evidence.input_fixture["processed_inputs"]["input_ids"]["shape"] == [
        1,
        19,
    ]
    assert evidence.action_shape == (7,)
    assert len(evidence.generated_token_ids) == 7
    assert evidence.token_ids_equal
    assert evidence.action_max_abs_error <= 1e-6
    assert evidence.trace_events > 0
    assert evidence.passed
