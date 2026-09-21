from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vlaforge.adapters.smolvla.smolvla_fresh import (
    LANGUAGE_MASK,
    STATE,
    TOKENS,
    build_smolvla_fresh_program,
    capture_smolvla_fresh_regions,
)
from vlaforge.compiler import compile_module
from vlaforge.interpreter import Interpreter
from vlaforge.plan import PlanExecutor


def attention_masks(pad_masks, _att_masks):
    return pad_masks[:, :, None] & pad_masks[:, None, :]


class FixtureVLM(torch.nn.Module):
    def forward(
        self,
        *,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        fill_kv_cache,
    ):
        del past_key_values, use_cache, fill_kv_cache
        prefix = inputs_embeds[0]
        value = (
            prefix + position_ids.unsqueeze(-1) + attention_mask.sum(-1).unsqueeze(-1)
        )
        return (None, None), {0: {"key_states": prefix, "value_states": value}}


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm_with_expert = FixtureVLM()

    def embed_prefix(self, images, image_masks, tokens, masks, *, state):
        visual = [
            image.mean(dim=(1, 2, 3)).unsqueeze(1) * mask.unsqueeze(1)
            for image, mask in zip(images, image_masks)
        ]
        values = torch.cat(
            [*visual, tokens.float() * masks, state.sum(1, keepdim=True)], dim=1
        )
        pad = torch.cat(
            [
                *(mask.unsqueeze(1) for mask in image_masks),
                masks,
                torch.ones_like(masks[:, :1]),
            ],
            dim=1,
        )
        return values.unsqueeze(-1), pad, torch.zeros_like(pad)

    def denoise_step(self, *, prefix_pad_masks, past_key_values, x_t, timestep):
        context = (
            past_key_values[0]["key_states"] + past_key_values[0]["value_states"]
        ).sum((1, 2))
        offset = context * 0.001 + prefix_pad_masks.sum(1) * 0.002
        return x_t * 0.125 + timestep[:, None, None] * 0.03125 + offset[:, None, None]


class FixturePolicy(torch.nn.Module):
    def __init__(self, *, camera_count=3, steps=10):
        super().__init__()
        self.config = SimpleNamespace(
            num_steps=steps,
            chunk_size=4,
            max_action_dim=3,
            action_feature=SimpleNamespace(shape=(2,)),
            n_obs_steps=1,
            adapt_to_pi_aloha=False,
            rtc_config=None,
            use_cache=True,
            empty_cameras=0,
            image_features={
                f"observation.images.camera_{i}": None for i in range(camera_count)
            },
        )
        self.model = FixtureModel()
        self._queues = {"action": deque([torch.ones(1)])}

    def prepare_images(self, batch):
        keys = [key for key in self.config.image_features if key in batch]
        return [batch[key] * 2.0 - 1.0 for key in keys], [
            batch[f"{key}_padding_mask"] for key in keys
        ]

    def prepare_state(self, batch):
        return batch[STATE]

    def predict_action_chunk(self, batch, *, noise):
        self._queues["action"].append(torch.zeros(1))
        images, masks = self.prepare_images(batch)
        embeds, pad, att = self.model.embed_prefix(
            images,
            masks,
            batch[TOKENS],
            batch[LANGUAGE_MASK],
            state=self.prepare_state(batch),
        )
        _, cache = self.model.vlm_with_expert(
            attention_mask=attention_masks(pad, att),
            position_ids=torch.cumsum(pad, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[embeds, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        sample = noise
        dt = -1.0 / self.config.num_steps
        for step in range(self.config.num_steps):
            timestep = torch.tensor(
                1.0 + step * dt, dtype=torch.float32, device=sample.device
            ).expand(sample.shape[0])
            velocity = self.model.denoise_step(
                prefix_pad_masks=pad,
                past_key_values=cache,
                x_t=sample,
                timestep=timestep,
            )
            sample = sample + dt * velocity
        return sample[:, :, : self.config.action_feature.shape[0]]


def inputs(camera_count=2, batch_size=1):
    batch = {
        STATE: torch.full((batch_size, 3), 0.125),
        TOKENS: torch.tensor([[1, 2, 3]]).expand(batch_size, -1).clone(),
        LANGUAGE_MASK: torch.tensor([[True, True, False]])
        .expand(batch_size, -1)
        .clone(),
    }
    for index in range(camera_count):
        key = f"observation.images.camera_{index}"
        batch[key] = torch.full((batch_size, 3, 2, 2), 0.125 * (index + 1))
        batch[f"{key}_padding_mask"] = torch.full(
            (batch_size,), index == 0, dtype=torch.bool
        )
    return batch, torch.arange(batch_size * 12, dtype=torch.float32).reshape(
        batch_size, 4, 3
    ) / 16


def bundle(policy=None, batch=None, noise=None, **kwargs):
    if batch is None:
        batch, noise = inputs()
    return build_smolvla_fresh_program(
        policy or FixturePolicy(),
        batch,
        noise,
        make_attention_masks=attention_masks,
        **kwargs,
    )


def runtime(built, kind=Interpreter):
    compiled = compile_module(built.program.module)
    options = dict(regions=built.program.regions, validators=built.program.validators)
    if kind is PlanExecutor:
        return compiled, PlanExecutor(compiled.plan, compiled.module, **options)
    return compiled, Interpreter(compiled.module, **options)


@pytest.mark.parametrize("camera_count,batch_size", [(1, 1), (2, 1), (4, 2)])
@pytest.mark.parametrize("kind", [Interpreter, PlanExecutor])
def test_full_fresh_chunk_uses_every_supplied_camera_and_shared_builder(
    camera_count, batch_size, kind
):
    batch, noise = inputs(camera_count, batch_size)
    policy = FixturePolicy(camera_count=camera_count + 1)
    queues = policy._queues
    built = bundle(policy, batch, noise)
    compiled, executor = runtime(built, kind)
    result = executor.run(inputs=built.bind_inputs()).committed_outputs.output(
        "action_chunk"
    )
    assert tuple(result.shape) == (batch_size, 4, 2)
    assert built.program.module.outputs[0].device == str(noise.device)
    assert torch.equal(result, built.reference_action_chunk)
    assert policy._queues is queues and len(queues["action"]) == 1
    assert len(built.camera_keys) == camera_count
    assert len(built.program.module.inputs) == camera_count * 2 + 4
    assert built.program.module.states == ()
    assert built.program.module.metadata["missing_camera_keys"] == [
        f"observation.images.camera_{camera_count}"
    ]
    assert built.input_tensors["noise"].data_ptr() != noise.data_ptr()
    loop = next(task for task in compiled.plan.tasks if task.opcode == "vla.for")
    assert len(loop.outputs) == 2


@pytest.mark.parametrize("steps", [3, 7, 10, 100])
def test_step_table_is_bit_equal_to_each_official_scalar_conversion(steps):
    built = bundle(FixturePolicy(steps=steps))
    expected = torch.stack(
        [
            torch.tensor(1.0 + step * (-1.0 / steps), dtype=torch.float32)
            for step in range(steps)
        ]
    )
    assert torch.equal(
        built.timestep_table.view(torch.int32), expected.view(torch.int32)
    )
    if steps == 10:
        incremental = torch.tensor(1.0)
        accumulated = []
        for _ in range(steps):
            accumulated.append(incremental.clone())
            incremental = incremental + (-1.0 / steps)
        assert not torch.equal(expected, torch.stack(accumulated))


def test_solver_uses_device_indices_and_official_multiply_then_add_order():
    built = bundle(FixturePolicy(steps=7))
    name = next(name for name in built.program.regions if name.endswith("_step"))
    solver = built.program.regions[name]
    args = built.region_examples[name]
    timestep = solver.timesteps.index_select(0, args[2])
    velocity = solver.model.denoise_step(
        prefix_pad_masks=args[0],
        past_key_values={0: {"key_states": args[3], "value_states": args[4]}},
        x_t=args[1],
        timestep=timestep,
    )
    sample, index = solver(*args)
    assert torch.equal(sample, args[1] + (-1.0 / 7) * velocity)
    assert index.dtype == torch.int64 and index.device == args[1].device
    assert torch.equal(index, args[2] + 1)


def test_automatic_prefix_cache_includes_all_images_masks_state_and_language():
    built = bundle()
    compiled, executor = runtime(built)
    (cache,) = compiled.certificate.caches
    names = {built.program.module.inputs[index].name for index in cache.input_ids}
    assert names == set(built.batch_keys)
    assert "noise" not in names
    revisions = {port.name: 1 for port in built.program.module.inputs}
    batch, noise = inputs()
    executor.run(inputs=built.bind_inputs(batch, noise, revisions=revisions))
    executor.run(
        inputs=built.bind_inputs(
            batch, noise + 0.1, revisions={**revisions, "noise": 2}
        )
    )
    assert executor.cache.hits == 1
    for index, port in enumerate(sorted(names), start=2):
        revisions[port] = index
        executor.run(inputs=built.bind_inputs(batch, noise, revisions=revisions))
    assert executor.cache.misses == len(names) + 1


def test_each_actual_camera_and_mask_changes_the_complete_output():
    batch, noise = inputs()
    batch["observation.images.camera_1_padding_mask"].fill_(True)
    built = bundle(batch=batch, noise=noise)
    _, executor = runtime(built)
    base = executor.run(inputs=built.bind_inputs()).committed_outputs.output(
        "action_chunk"
    )
    for key in built.camera_keys:
        changed = {name: value.clone() for name, value in batch.items()}
        changed[key].add_(0.125)
        actual = executor.run(
            inputs=built.bind_inputs(changed, noise)
        ).committed_outputs.output("action_chunk")
        assert not torch.equal(base, actual)
        changed = {name: value.clone() for name, value in batch.items()}
        changed[f"{key}_padding_mask"].fill_(False)
        actual = executor.run(
            inputs=built.bind_inputs(changed, noise)
        ).committed_outputs.output("action_chunk")
        assert not torch.equal(base, actual)


@pytest.mark.parametrize(
    "bad",
    [
        "missing_mask",
        "float_mask",
        "nan_image",
        "nan_noise",
        "short_noise",
        "wrong_device_index_shape",
    ],
)
def test_invalid_or_nonfinite_input_profiles_fail_closed(bad):
    batch, noise = inputs()
    if bad == "missing_mask":
        del batch["observation.images.camera_1_padding_mask"]
    elif bad == "float_mask":
        batch["observation.images.camera_1_padding_mask"] = torch.ones(1)
    elif bad == "nan_image":
        batch["observation.images.camera_0"].fill_(float("nan"))
    elif bad == "nan_noise":
        noise.fill_(float("nan"))
    elif bad == "short_noise":
        noise = torch.zeros(1, 3, 3)
    else:
        noise = torch.tensor(1.0)
    with pytest.raises(ValueError):
        bundle(batch=batch, noise=noise)


@pytest.mark.parametrize(
    "setting,value",
    [
        ("n_obs_steps", 2),
        ("adapt_to_pi_aloha", True),
        ("rtc_config", object()),
        ("use_cache", False),
        ("num_steps", 0),
    ],
)
def test_unsupported_hidden_state_contracts_are_explicit(setting, value):
    policy = FixturePolicy()
    setattr(policy.config, setting, value)
    with pytest.raises(ValueError):
        bundle(policy)


def test_runtime_nonfinite_chunk_aborts_without_overwriting_committed_output():
    built = bundle()
    _, executor = runtime(built)
    before = executor.run(inputs=built.bind_inputs()).committed_outputs
    name = next(name for name in built.program.regions if name.endswith("_step"))
    built.program.regions[name].forward = lambda masks, sample, index, *cache: (
        torch.full_like(sample, float("nan")),
        index + 1,
    )
    with pytest.raises(RuntimeError, match="validation"):
        executor.run(inputs=built.bind_inputs())
    assert torch.isfinite(before.output("action_chunk")).all()


def test_rebinding_cannot_silently_change_camera_availability_or_profile():
    built = bundle()
    batch, noise = inputs(camera_count=3)
    with pytest.raises(ValueError, match="camera availability"):
        built.bind_inputs(batch, noise)
    batch, noise = inputs()
    batch[TOKENS] = batch[TOKENS].to(torch.int32)
    with pytest.raises(ValueError, match="locked shape/dtype/device"):
        built.bind_inputs(batch, noise)
    with pytest.raises(ValueError, match="revision keys"):
        built.bind_inputs(revisions={"not_a_port": 1})
    batch, noise = inputs()
    noise.fill_(float("inf"))
    with pytest.raises(ValueError, match="finite"):
        built.bind_inputs(batch, noise)


def test_all_regions_capture_save_and_replay_full_chunk_without_python_loop_timestep(
    tmp_path,
):
    built = bundle()
    captures = capture_smolvla_fresh_regions(built, tmp_path, strict=True)
    assert len(captures) == 4
    implementations = {}
    for outcome in captures:
        name = outcome.region.name
        exported = torch.export.load(tmp_path / f"{name}.pt2e")
        implementations[name] = exported.module()
        assert (tmp_path / f"{name}.capture.json").is_file()
        if name.endswith("_step"):
            targets = {
                str(node.target)
                for node in exported.graph.nodes
                if node.op == "call_function"
            }
            assert "aten.index_select.default" in targets
            assert not any(
                "_local_scalar_dense" in target or "lift_fresh_copy" in target
                for target in targets
            )
    executor = Interpreter(
        built.program.module,
        regions=implementations,
        validators=built.program.validators,
    )
    actual = executor.run(inputs=built.bind_inputs()).committed_outputs.output(
        "action_chunk"
    )
    assert torch.equal(actual, built.reference_action_chunk)
