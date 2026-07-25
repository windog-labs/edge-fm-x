import os
from pathlib import Path

import pytest

from vlaforge.adapters.smolvla_real import (
    RealSmolVLAConfig,
    build_real_smolvla_action_program,
    run_real_smolvla,
)
from vlaforge.analysis import verify


def test_real_smolvla_source_contract_uses_tensor_ports_and_adapter_state():
    module = build_real_smolvla_action_program(
        chunk_size=50,
        max_action_dim=32,
        output_action_dim=6,
        num_steps=10,
        token_length=48,
        device="cpu",
    )
    assert [item.name for item in module.inputs] == [
        "image",
        "state",
        "instruction_tokens",
        "instruction_mask",
        "noise",
    ]
    assert [item.name for item in module.states] == [
        "action_queue",
        "queue_cursor",
    ]
    assert verify(module, raise_on_error=False) == ()


@pytest.mark.real_model
def test_real_smolvla_checkpoint_matches_ir(tmp_path):
    policy_path = os.getenv("VLAFORGE_SMOLVLA_POLICY_PATH")
    vlm_path = os.getenv("VLAFORGE_SMOLVLA_VLM_PATH")
    if not policy_path or not vlm_path:
        pytest.skip(
            "set VLAFORGE_SMOLVLA_POLICY_PATH and VLAFORGE_SMOLVLA_VLM_PATH"
        )

    evidence = run_real_smolvla(
        RealSmolVLAConfig(
            policy_path=Path(policy_path),
            vlm_path=Path(vlm_path),
            device=os.getenv("VLAFORGE_MODEL_DEVICE", "cuda"),
            num_steps=int(os.getenv("VLAFORGE_SMOLVLA_NUM_STEPS", "10")),
            tolerance=1e-5,
            lerobot_revision=os.getenv(
                "VLAFORGE_LEROBOT_REVISION", "unknown"
            ),
        ),
        trace_path=tmp_path / "trace.json",
    )

    assert evidence.evidence_kind == "real_checkpoint"
    assert evidence.schema == "vlaforge.real_model_evidence/0.2"
    assert evidence.num_steps == 10
    assert evidence.action_shape == (1, 50, 6)
    assert evidence.action_max_abs_error <= 1e-5
    assert evidence.solver_max_abs_errors == (0.0,) * 10
    assert evidence.action_queue_max_abs_errors == (0.0,) * 3
    assert evidence.trace_events > 0
    assert evidence.passed
