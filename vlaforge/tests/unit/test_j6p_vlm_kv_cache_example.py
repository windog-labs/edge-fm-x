"""Two whole-model calls, explicit cache ABI and fresh-request failure tests."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[2] / "examples/j6p_vlm_kv_cache.py"
_SPEC = importlib.util.spec_from_file_location("j6p_vlm_kv_cache_example", _PATH)
example = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = example
_SPEC.loader.exec_module(example)

Spec = example.StateTensorSpec


class FakeSession:
    """Control-flow fixture only; values are not model outputs."""

    model_id = "test-fixture"
    prompt_length = 3
    max_sequence_length = 8
    prefill_manifest = SimpleNamespace(
        state_outputs=(Spec("present_k", (1,), "bf16"), Spec("present_v", (1,), "bf16")),
    )
    decode_manifest = SimpleNamespace(
        state_inputs=(Spec("past_v", (1,), "bf16"), Spec("past_k", (1,), "bf16")),
        state_outputs=(Spec("updated_k", (1,), "bf16"), Spec("updated_v", (1,), "bf16")),
    )
    prefill_to_decode = {"past_k": "present_k", "past_v": "present_v"}
    decode_to_decode = {"past_k": "updated_k", "past_v": "updated_v"}

    def __init__(self, *, fail_position=None):
        self.fail_position = fail_position
        self.reset_count = 0
        self.prefills = 0
        self.calls = []

    def reset(self):
        self.reset_count += 1

    def prefill(self, **_):
        return example.PrefillResult(first_token=101, rope_deltas=0, state=(10, 20))

    def decode(self, *, token, cache_position, rope_deltas, state):
        self.calls.append((token, cache_position, state))
        if self.fail_position == cache_position:
            raise RuntimeError("decode failed")
        # Output K,V ABI differs from the V,K input ABI.
        return example.DecodeResult(next_token=token + 1, state=(state[1] + 1, state[0] + 1))


def inputs():
    return example.PreparedVLMInput(
        input_ids=[1, 2, 3], attention_mask=[1, 1, 1], pixel_values=b"image",
        image_grid_thw=[1, 2, 2], mm_token_type_ids=[0, 0, 1], prompt_length=3,
    )


def test_generation_maps_different_state_ports_and_advances_absolute_position():
    session = FakeSession()
    result = example.J6PVLMDeployment(session).generate(inputs(), new_tokens=4)
    assert result.tokens == (101, 102, 103, 104)
    assert session.calls == [(101, 3, (20, 10)), (102, 4, (21, 11)), (103, 5, (22, 12))]


def test_first_token_needs_only_prefill_and_new_request_always_resets():
    session = FakeSession()
    deployment = example.J6PVLMDeployment(session)
    deployment.generate(inputs(), new_tokens=1)
    deployment.generate(replace(inputs(), pixel_values=b"new image"), new_tokens=1)
    assert session.reset_count == 2
    assert not session.calls


def test_generation_rejects_capacity_before_any_provider_call():
    session = FakeSession()
    deployment = example.J6PVLMDeployment(session)
    with pytest.raises(ValueError, match="cache capacity"):
        deployment.generate(inputs(), new_tokens=7)
    assert not session.calls and session.reset_count == 0
    deployment.generate(inputs(), new_tokens=6)
    assert session.calls[-1][1] == 7


@pytest.mark.parametrize("new_tokens", [True, 1.5, 0, -1])
def test_bad_generation_lengths_reject(new_tokens):
    with pytest.raises(ValueError, match="positive"):
        example.J6PVLMDeployment(FakeSession()).generate(inputs(), new_tokens=new_tokens)


def test_wrong_prompt_profile_rejects():
    with pytest.raises(ValueError, match="compiled profile"):
        example.J6PVLMDeployment(FakeSession()).generate(
            replace(inputs(), prompt_length=4), new_tokens=1,
        )


def test_incompatible_state_shape_rejects_before_hbm_execution():
    session = FakeSession()
    session.prefill_manifest = SimpleNamespace(
        state_outputs=(Spec("present_k", (2,), "bf16"), Spec("present_v", (1,), "bf16")),
    )
    with pytest.raises(ValueError, match="ABI mismatch"):
        example.J6PVLMDeployment(session)
    assert session.prefills == 0


def test_missing_feedback_binding_rejects():
    session = FakeSession()
    session.decode_to_decode = {"past_k": "updated_k"}
    with pytest.raises(ValueError, match="explicit binding"):
        example.J6PVLMDeployment(session)


def test_one_source_cannot_supply_two_cache_inputs():
    with pytest.raises(ValueError, match="exactly once"):
        example.state_order(
            (Spec("present_k", (1,), "bf16"),),
            (Spec("past_k", (1,), "bf16"), Spec("past_v", (1,), "bf16")),
            {"past_k": "present_k", "past_v": "present_k"},
        )


def test_failure_discards_request_state_and_retry_starts_with_prefill():
    session = FakeSession(fail_position=4)
    deployment = example.J6PVLMDeployment(session)
    with pytest.raises(RuntimeError, match="decode failed"):
        deployment.generate(inputs(), new_tokens=4)
    assert session.reset_count == 2
    assert deployment._committed_key is None
    session.fail_position = None
    assert deployment.generate(inputs(), new_tokens=2).tokens == (101, 102)
    assert session.calls[-1] == (101, 3, (20, 10))


def test_reset_failure_does_not_leave_old_commit_key():
    session = FakeSession()
    deployment = example.J6PVLMDeployment(session)
    deployment.generate(inputs(), new_tokens=1)
    def failed_reset():
        raise RuntimeError("reset failed")
    session.reset = failed_reset
    with pytest.raises(RuntimeError, match="reset failed"):
        deployment.reset()
    assert deployment._committed_key is None


def test_manifest_loads_distinct_io_and_rejects_custom_ops(tmp_path):
    # File existence only is tested here, not HBM legality or execution.
    (tmp_path / "prefill.hbm").write_bytes(b"metadata-test-fixture")
    payload = dict(stage="prefill", hbm="prefill.hbm", inputs=["input_ids"],
                   outputs=["first_token", "present_k"], state_inputs=[],
                   state_outputs=[dict(name="present_k", shape=[1], dtype="bf16", layout="contiguous")],
                   custom_ops=False)
    path = tmp_path / "prefill.json"
    path.write_text(json.dumps(payload))
    manifest = example.CompleteHbmManifest.from_json(path)
    assert manifest.state_inputs == () and manifest.state_outputs[0].name == "present_k"
    for bad in (True, "false", None):
        path.write_text(json.dumps({**payload, "custom_ops": bad}))
        with pytest.raises(ValueError, match="custom operators"):
            example.CompleteHbmManifest.from_json(path)


def test_identity_includes_shape_dtype_and_unambiguous_field_lengths():
    np = pytest.importorskip("numpy")
    values = np.array([1, 2], dtype=np.int32)
    variants = (values, values.reshape(1, 2), values.view(np.float32))
    assert len({example.input_identity({"x": v}) for v in variants}) == 3
    assert example.input_identity({"x": b"a\0y\0bytes:b"}) != example.input_identity({"x": b"a", "y": b"b"})


def test_bf16_identity_is_supported():
    torch = pytest.importorskip("torch")
    assert len(example.input_identity({"x": torch.ones(2, dtype=torch.bfloat16)})) == 64


def test_preparation_rejects_padding():
    np = pytest.importorskip("numpy")
    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"
        def __call__(self, **kwargs):
            return dict(input_ids=np.array([[1, 2, 0]]), attention_mask=np.array([[1, 1, 0]]))
    with pytest.raises(ValueError, match="padded prompts"):
        example.prepare_qwen35(Processor(), None, "hello")
