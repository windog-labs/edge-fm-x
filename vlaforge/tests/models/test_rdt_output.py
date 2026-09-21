from types import SimpleNamespace

import pytest
import torch
from vlaforge.adapters.rdt.rdt_reference import official_agilex_output_transform


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_wrapper_method_is_reused_without_loading_models(dtype, tmp_path):
    class Wrapper:
        def __init__(self, *args, **kwargs):
            raise AssertionError("output adaptation must not load a model")

        def _unformat_action_to_joint(self, actions):
            values = actions[:, :, [3, 0, 2]]
            return values * torch.tensor([1, 11.8997, 13.9231], dtype=values.dtype,
                                         device=values.device)

    transform = official_agilex_output_transform(SimpleNamespace(
        wrapper_class=Wrapper, source_config={}))
    assert not tuple(transform.parameters()) and not tuple(transform.buffers())
    actions = torch.arange(16, dtype=dtype).reshape(1, 4, 4) / 7
    original = actions.clone()
    expected = object.__new__(Wrapper)._unformat_action_to_joint(actions)
    assert torch.equal(transform(actions), expected)
    exported = torch.export.export(transform, (actions,), strict=False)
    path = tmp_path / "output.pt2"
    torch.export.save(exported, path)
    assert torch.equal(torch.export.load(path).module()(actions), expected)
    assert torch.equal(actions, original)
