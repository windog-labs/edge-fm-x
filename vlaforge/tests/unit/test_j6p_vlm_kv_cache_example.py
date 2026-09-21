"""Contract tests for the J6P VLM Python orchestration example."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[2] / "examples/j6p_vlm_kv_cache.py"
_SPEC = importlib.util.spec_from_file_location("j6p_vlm_kv_cache_example", _PATH)
example = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = example
_SPEC.loader.exec_module(example)

CompleteHbmManifest = example.CompleteHbmManifest
DecodeResult = example.DecodeResult
J6PVLMDeployment = example.J6PVLMDeployment
PreparedVLMInput = example.PreparedVLMInput
PrefillResult = example.PrefillResult
StateTensorSpec = example.StateTensorSpec
input_identity = example.input_identity


class FakeSession:
    model_id = "fake-qwen35-08b"
    max_sequence_length = 8
    state_schema = (StateTensorSpec("state", (1,), "bf16"),)
    prefill_manifest = SimpleNamespace(state_schema=state_schema)
    decode_manifest = SimpleNamespace(state_schema=state_schema)

    def __init__(self, *, fail_step: int | None = None):
        self.fail_step = fail_step
        self.reset_count = 0
        self.steps = []

    def reset(self):
        self.reset_count += 1

    def prefill(self, **_):
        return PrefillResult(first_token=101, rope_deltas=0, state=(0,))

    def decode(self, *, token, step, rope_deltas, state):
        del rope_deltas
        self.steps.append((token, step, state))
        if self.fail_step == step:
            raise RuntimeError("decode failed")
        return DecodeResult(next_token=token + 1, state=(state[0] + 1,))


def inputs(prompt_length=3):
    return PreparedVLMInput(
        input_ids=[1, 2, 3],
        attention_mask=[1, 1, 1],
        pixel_values=b"image",
        image_grid_thw=[1, 2, 2],
        mm_token_type_ids=[0, 0, 1],
        prompt_length=prompt_length,
    )


def test_input_identity_changes_when_prompt_or_image_changes():
    first = input_identity({"prompt": [1, 2], "image": b"a"})
    second = input_identity({"prompt": [1, 3], "image": b"a"})
    third = input_identity({"prompt": [1, 2], "image": b"b"})
    assert len({first, second, third}) == 3


def test_generation_uses_prefill_then_one_token_decode_steps():
    session = FakeSession()
    result = J6PVLMDeployment(session).generate(inputs(), new_tokens=4)
    assert result.tokens == (101, 102, 103, 104)
    assert [step for _, step, _ in session.steps] == [1, 2, 3]
    assert session.steps[0][2] == (0,)


def test_j6p_provider_requires_two_complete_standard_op_hbms(tmp_path):
    prefill = tmp_path / "prefill.hbm"
    decode = tmp_path / "decode.hbm"
    prefill.write_bytes(b"prefill")
    decode.write_bytes(b"decode")
    state = (StateTensorSpec("state", (1,), "bf16"),)
    prefill_manifest = CompleteHbmManifest(
        "prefill", prefill, ("input_ids",), ("first_token",), state
    )
    decode_manifest = CompleteHbmManifest(
        "decode", decode, ("token", "state"), ("next_token", "state"), state
    )
    assert prefill_manifest.stage == "prefill"
    assert decode_manifest.stage == "decode"
    with pytest.raises(ValueError, match="custom operators"):
        CompleteHbmManifest(
            "decode", decode, ("token",), ("next_token",), state, True
        )


def test_generation_rejects_cache_capacity_before_provider_call():
    session = FakeSession()
    with pytest.raises(ValueError, match="cache capacity"):
        J6PVLMDeployment(session).generate(inputs(prompt_length=7), new_tokens=3)
    assert session.steps == []


def test_failed_decode_resets_the_session_and_does_not_commit_state():
    session = FakeSession(fail_step=2)
    deployment = J6PVLMDeployment(session)
    with pytest.raises(RuntimeError, match="decode failed"):
        deployment.generate(inputs(), new_tokens=4)
    assert session.reset_count == 2  # before prefill and after the failure
    assert deployment._committed_key is None
