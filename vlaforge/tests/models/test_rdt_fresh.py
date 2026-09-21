from types import SimpleNamespace

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from vlaforge.adapters.rdt.rdt_fresh import build_rdt_fresh_program
from vlaforge.compiler import compile_module
from vlaforge.interpreter import Interpreter


class Text(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 3).to(torch.bfloat16)

    def forward(self, input_ids, attention_mask):
        return {"last_hidden_state": self.embed(input_ids) * attention_mask.unsqueeze(-1)}


class Vision(torch.nn.Module):
    def forward(self, pixels):
        return {"last_hidden_state": pixels.mean((2, 3)).unsqueeze(1)}


class Denoiser(torch.nn.Module):
    def forward(self, trajectory, frequency, timestep, language, image, *, lang_mask):
        context = language.mean() * .01 + image.mean() * .01 + trajectory[:, :1, :4].mean() * .01
        return trajectory[:, 1:, :4] * .125 + context + (frequency + timestep).to(trajectory.dtype).reshape(1, 1, 1) * .0001


def fixture():
    torch.manual_seed(3)
    scheduler = diffusers.DPMSolverMultistepScheduler(prediction_type="sample", beta_schedule="squaredcos_cap_v2")
    policy = SimpleNamespace(
        pred_horizon=2, action_dim=4, num_inference_timesteps=5, noise_scheduler_sample=scheduler,
        lang_adaptor=torch.nn.Linear(3, 8).to(torch.bfloat16),
        img_adaptor=torch.nn.Linear(3, 8).to(torch.bfloat16),
        state_adaptor=torch.nn.Linear(8, 8).to(torch.bfloat16), model=Denoiser(),
    )
    loaded = SimpleNamespace(policy=policy, text_embedder=SimpleNamespace(model=Text()),
                             vision_tower=SimpleNamespace(config=SimpleNamespace(image_size=2), select_feature="patch", vision_tower=Vision()))
    supplied = {
        "token_ids": torch.tensor([[1, 2]]), "text_mask": torch.tensor([[1, 1]]),
        "pixels": torch.randn(6, 3, 2, 2).to(torch.bfloat16),
        "unified_state": torch.randn(1, 1, 4).to(torch.bfloat16),
        "action_mask": torch.tensor([[[1, 0, 1, 1]]], dtype=torch.bfloat16),
        "noise": torch.randn(1, 2, 4).to(torch.bfloat16), "control_frequency": torch.tensor([25]),
    }
    return loaded, supplied


def test_all_online_stages_shared_builder_and_solver_weight_isolation():
    loaded, inputs = fixture()
    built = build_rdt_fresh_program(loaded, inputs)
    assert built.eager_actions.shape == (1, 2, 4)
    assert torch.count_nonzero(built.eager_actions[..., 1]) == 0
    assert len(built.program.module.regions) == 7
    regions = {region.name: region for region in built.program.module.regions}
    cached = [region for region in regions.values() if region.metadata.get("memoize")]
    assert {tuple(region.metadata["cache_input_ports"]) for region in cached} == {
        ("text_mask", "token_ids"), ("pixels",),
        ("action_mask", "pixels", "text_mask", "token_ids", "unified_state"),
    }
    assert not list(built.program.regions["rdt_solver"].parameters())
    assert not list(built.program.regions["rdt_initialize"].parameters())
    assert all(not value.is_inference() for arguments in built.region_examples.values() for value in arguments)
    compiled = compile_module(built.program.module)
    runtime = Interpreter(compiled.module, regions=built.program.regions, validators=built.program.validators)
    with torch.inference_mode():
        output = runtime.run(inputs=built.bind_inputs()).committed_outputs.output("action_chunk")
    assert torch.equal(output, built.eager_actions)


def test_condition_examples_are_exportable_with_trainable_weights():
    if torch.__version__.startswith("2.1."):
        pytest.skip("compiler-profile export validation runs in Torch 2.10")
    loaded, inputs = fixture()
    built = build_rdt_fresh_program(loaded, inputs)
    name = next(name for name in built.program.regions if "conditions" in name)
    module, arguments = built.program.regions[name], built.region_examples[name]
    exported = torch.export.export(module, arguments, strict=False)
    with torch.no_grad():
        for expected, actual in zip(module(*arguments), exported.module()(*arguments), strict=True):
            assert torch.equal(expected, actual)


@pytest.mark.parametrize("change", ["missing_camera", "nonfinite", "wrong_dtype", "nonbinary_mask", "missing_noise"])
def test_explicit_input_profile_fails_closed(change):
    loaded, supplied = fixture()
    if change == "missing_camera":
        supplied["pixels"] = supplied["pixels"][:5]
    elif change == "nonfinite":
        supplied["noise"][0, 0, 0] = float("nan")
    elif change == "wrong_dtype":
        supplied["noise"] = supplied["noise"].float()
    elif change == "nonbinary_mask":
        supplied["action_mask"][0, 0, 0] = .5
    else:
        del supplied["noise"]
    with pytest.raises(ValueError):
        build_rdt_fresh_program(loaded, supplied)


@pytest.mark.parametrize("formula", ["permuted_scaled", "nonlinear"])
def test_explicit_output_transform_preserves_complete_unified_auxiliary(formula, tmp_path):
    class Output(torch.nn.Module):
        def forward(self, value):
            if formula == "permuted_scaled":
                return value[:, :, [3, 0]] * torch.tensor([2, .5], dtype=value.dtype, device=value.device)
            return value[:, :, :2].sigmoid().contiguous()

    loaded, supplied = fixture()
    original = build_rdt_fresh_program(loaded, supplied)
    transform = Output()
    built = build_rdt_fresh_program(loaded, supplied, output_transform=transform,
                                   output_space="explicit-test-embodiment")
    assert len(built.program.module.regions) == 8
    assert [port.name for port in built.program.module.outputs] == ["action_chunk", "unified_action_chunk"]
    assert torch.equal(built.eager_unified_actions, original.eager_actions)
    assert torch.equal(built.eager_actions, transform(original.eager_actions))
    assert len(built.eager_samples) == len(original.eager_samples) == 5
    assert all(torch.equal(a, b) for a, b in zip(built.eager_samples, original.eager_samples, strict=True))
    compiled = compile_module(built.program.module)
    runtime = Interpreter(compiled.module, regions=built.program.regions, validators=built.program.validators)
    with torch.inference_mode():
        result = runtime.run(inputs=built.bind_inputs()).committed_outputs
    assert torch.equal(result.output("action_chunk"), built.eager_actions)
    assert torch.equal(result.output("unified_action_chunk"), original.eager_actions)
    output_module = built.program.regions["rdt_output"]
    examples = built.region_examples["rdt_output"]
    exported = torch.export.export(output_module, examples)
    path = tmp_path / "output.pt2"
    torch.export.save(exported, path)
    reloaded = torch.export.load(path).module()
    with torch.no_grad():
        for expected, actual in zip(output_module(*examples), reloaded(*examples), strict=True):
            assert torch.equal(expected, actual)
        assert not reloaded(examples[0], torch.tensor([False]))[1].item()
        assert not reloaded(torch.full_like(examples[0], float("nan")), examples[1])[1].item()


@pytest.mark.parametrize("change", ["missing_space", "no_module", "space_without_transform", "truncated_horizon", "nonfinite"])
def test_output_transform_fails_closed(change):
    class Output(torch.nn.Module):
        def forward(self, value):
            if change == "truncated_horizon":
                return value[:, :1].contiguous()
            if change == "nonfinite":
                return value * float("nan")
            return value.clone()

    loaded, supplied = fixture()
    options = {"output_transform": Output(), "output_space": "selected-embodiment"}
    if change == "missing_space":
        options["output_space"] = None
    elif change == "no_module":
        options["output_transform"] = lambda x: x
    elif change == "space_without_transform":
        options["output_transform"] = None
    with pytest.raises(ValueError, match="output|transformed"):
        build_rdt_fresh_program(loaded, supplied, **options)
