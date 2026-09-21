"""Pure output-lowering fixtures; actual official processor audits are separate."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vlaforge.adapters.openpi.openpi_output import (
    checked_output_module,
    tensor_absolute_actions,
    tensor_aloha_actions,
    tensor_unnormalize,
)


@pytest.mark.parametrize("quantiles", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalization_preserves_numpy_order_tail_and_dtype(quantiles, dtype):
    stats = SimpleNamespace(
        mean=np.array([0.3, -0.2], dtype=dtype),
        std=np.array([0.7, 1.3], dtype=dtype),
        q01=np.array([-0.4, -0.9], dtype=dtype),
        q99=np.array([0.8, 0.7], dtype=dtype),
    )
    actions = np.arange(12, dtype=np.float32).reshape(1, 3, 4) / 7
    state = np.arange(4, dtype=np.float64).reshape(1, 4) / 9
    module = tensor_unnormalize({"actions": stats, "state": stats}, use_quantiles=quantiles)
    actual = module(torch.from_numpy(state.copy()), torch.from_numpy(actions.copy()))
    for source, value in zip((state, actions), actual, strict=True):
        if quantiles:
            expected = np.concatenate(((source[..., :2] + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01, source[..., 2:]), axis=-1)
        else:
            expected = source * (np.pad(stats.std, (0, 2), constant_values=1) + 1e-6) + np.pad(stats.mean, (0, 2))
        assert value.numpy().dtype == expected.dtype
        np.testing.assert_array_equal(value.numpy(), expected)


def test_absolute_actions_retains_numpy_inplace_cast_and_unmasked_tail():
    state = np.array([[0.13, -0.17, 0.33]], dtype=np.float64)
    actions = np.array([[[0.19, 0.27, 0.73]]], dtype=np.float32)
    expected = actions.copy()
    expected[..., :2] += np.where([True, False], state[..., :2], 0)[:, None, :]
    _, actual = tensor_absolute_actions((True, False))(torch.from_numpy(state), torch.from_numpy(actions))
    assert actual.dtype == torch.float32
    np.testing.assert_array_equal(actual.numpy(), expected)
    np.testing.assert_array_equal(actions, np.array([[[0.19, 0.27, 0.73]]], dtype=np.float32))


@pytest.mark.parametrize("adapt", [False, True])
def test_aloha_slice_and_numpy_int64_promotion_are_explicit(adapt):
    mask = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=np.int64)
    actions = np.arange(32, dtype=np.float32).reshape(1, 1, 32) / 31
    expected = actions[..., :14]
    if adapt:
        expected = mask * expected
        expected[..., [6, 13]] = ((expected[..., [6, 13]] + 0.5476) - (-0.6213)) / (1.4910 - (-0.6213))
    module = tensor_aloha_actions(adapt_to_pi=adapt, joint_flip_mask=mask)
    _, actual = module(torch.zeros(1, 32), torch.from_numpy(actions))
    assert actual.numpy().dtype == expected.dtype
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("mask", [(True, 1), (), [False, "false"]])
def test_unknown_mask_semantics_fail_closed(mask):
    with pytest.raises(ValueError, match="boolean"):
        tensor_absolute_actions(mask)


def test_export_save_reload_keeps_original_padded_state_and_full_output(tmp_path):
    mask = np.ones(14, dtype=np.int64)
    module = tensor_aloha_actions(adapt_to_pi=True, joint_flip_mask=mask)
    inputs = (torch.zeros(1, 32, dtype=torch.float64), torch.arange(1600, dtype=torch.float32).reshape(1, 50, 32) / 1600)
    program = torch.export.export(module, inputs)
    path = tmp_path / "output.pt2"
    torch.export.save(program, path)
    actual = torch.export.load(path).module()(*inputs)
    expected = module(*inputs)
    assert actual[1].shape == (1, 50, 14)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_aloha_divisor_stays_a_tensor_buffer_in_export():
    module = tensor_aloha_actions(adapt_to_pi=True, joint_flip_mask=np.ones(14, dtype=np.int64))
    assert module.gripper_denominator.dtype == torch.float64
    assert module.gripper_denominator.shape == (1,)
    assert module.gripper_denominator.item() == 1.4910 - (-0.6213)
    program = torch.export.export(module, (torch.zeros(1, 32), torch.zeros(1, 50, 32)))
    divisions = [node for node in program.graph.nodes if node.target == torch.ops.aten.div.Tensor]
    assert len(divisions) == 2
    assert all(isinstance(node.args[1], torch.fx.Node) for node in divisions)


@pytest.mark.parametrize("location", ["retained", "discarded"])
def test_checked_output_rejects_nonfinite_normalized_values_even_in_discarded_tail(location):
    module = checked_output_module(tensor_aloha_actions(adapt_to_pi=False, joint_flip_mask=np.ones(14, dtype=np.int64)))
    # The ALOHA stage returns state/actions; the public composite returns only actions.
    class Composite(torch.nn.Module):
        def forward(self, state, actions):
            return module.output(state, actions)[1]
    checked = checked_output_module(Composite())
    state, actions = torch.zeros(1, 32), torch.zeros(1, 2, 32)
    assert checked(state, actions, torch.tensor([True]))[1].item() is True
    assert checked(state, actions, torch.tensor([False]))[1].item() is False
    actions[0, 0, 0 if location == "retained" else 31] = float("nan")
    assert checked(state, actions, torch.tensor([True]))[1].item() is False


def test_unsupported_statistics_dtype_is_not_silently_promoted():
    stats = SimpleNamespace(mean=np.array([0, 1]), std=np.array([1, 1]))
    with pytest.raises(TypeError, match="float32 or float64"):
        tensor_unnormalize({"actions": stats}, use_quantiles=False)


def test_inconsistent_statistics_vectors_fail_closed():
    stats = SimpleNamespace(mean=np.array([0.0, 1.0]), std=np.array([1.0]))
    with pytest.raises(ValueError, match="dimensions"):
        tensor_unnormalize({"actions": stats}, use_quantiles=False)
