import os

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from vlaforge.adapters.rdt.rdt_scheduler import make_dpm_solver_step


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("steps", [3, 5, 10])
def test_official_solver_every_sample_and_explicit_history_exact(dtype, steps):
    scheduler = diffusers.DPMSolverMultistepScheduler(
        beta_schedule="squaredcos_cap_v2", prediction_type="sample",
    )
    scheduler.set_timesteps(steps)
    module = make_dpm_solver_step(scheduler, steps, device="cpu")
    generator = torch.Generator().manual_seed(372)
    expected = torch.randn(1, 4, 8, generator=generator).to(dtype)
    actual = expected.clone()
    older, previous = torch.zeros_like(actual), torch.zeros_like(actual)
    count, index = torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64)
    assert not list(module.parameters())
    for step, timestep in enumerate(scheduler.timesteps):
        current = torch.randn(actual.shape, generator=generator).to(dtype)
        expected = scheduler.step(current, timestep, expected).prev_sample.to(dtype)
        actual, older, previous, count, index = module(actual, current, older, previous, count, index)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert index.item() == step + 1 and count.item() == min(step + 1, 2)
        assert torch.equal(previous, scheduler.model_outputs[-1])
        if step:
            assert torch.equal(older, scheduler.model_outputs[-2])
        else:
            assert scheduler.model_outputs[-2] is None and torch.count_nonzero(older) == 0


@pytest.mark.parametrize("change", [{"prediction_type": "epsilon"}, {"solver_order": 3},
                                    {"solver_type": "heun"}, {"thresholding": True}])
def test_different_scheduler_requires_explicit_new_lowering(change):
    settings = {"prediction_type": "sample"}
    settings.update(change)
    scheduler = diffusers.DPMSolverMultistepScheduler(**settings)
    with pytest.raises(ValueError, match="outside"):
        make_dpm_solver_step(scheduler, 5, device="cpu")


def test_export_keeps_device_index_and_no_hidden_rng_or_host_scalar_read():
    if not hasattr(torch.export, "export") or torch.__version__.startswith("2.1."):
        pytest.skip("compiler-profile torch.export validation runs in Torch 2.10")
    scheduler = diffusers.DPMSolverMultistepScheduler(prediction_type="sample")
    module = make_dpm_solver_step(scheduler, 5, device="cpu")
    sample = torch.ones(1, 4, 8, dtype=torch.bfloat16)
    index = torch.zeros(1, dtype=torch.int64)
    example = (sample, sample, sample, sample, index, index)
    exported = torch.export.export(module, example, strict=False)
    targets = [str(node.target) for node in exported.graph.nodes if node.op == "call_function"]
    assert any("index_select" in name for name in targets)
    assert not any("_local_scalar_dense" in name or "rand" in name for name in targets)
    for step in range(5):
        args = (*example[:4], torch.tensor([min(step, 2)]), torch.tensor([step]))
        expected = module(*args)
        actual = exported.module()(*args)
        for left, right in zip(expected, actual, strict=True):
            assert torch.equal(left, right)


@pytest.mark.skipif(os.environ.get("VLAFORGE_RUN_CUDA_TESTS") != "1", reason="explicit exclusive CUDA test opt-in")
def test_cuda_cpu_scalar_opmath_rounding_is_preserved():
    scheduler = diffusers.DPMSolverMultistepScheduler(
        prediction_type="sample", beta_schedule="squaredcos_cap_v2",
    )
    scheduler.set_timesteps(5)
    module = make_dpm_solver_step(scheduler, 5, device="cuda:0")
    sample = torch.zeros(1, 1, 4, dtype=torch.bfloat16, device="cuda:0")
    current = torch.tensor([[[.75, .1875, -.375, .0234375]]], dtype=sample.dtype, device=sample.device)
    index = torch.zeros(1, dtype=torch.int64, device=sample.device)
    expected = scheduler.step(current, scheduler.timesteps[0], sample).prev_sample.to(sample.dtype)
    actual = module(sample, current, sample, sample, index, index)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
