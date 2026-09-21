"""CPU profile rejection checks; these are not real-model fidelity evidence."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vlaforge.adapters.cogact.cogact_partitioned import (
    build_cogact_partition,
    produce_rng_tape,
)


def example():
    return {"tokens": torch.zeros((1, 31), dtype=torch.int64),
            "dino": torch.zeros((1, 3, 224, 224)), "siglip": torch.zeros((1, 3, 224, 224)),
            "initial_noise": torch.zeros((1, 16, 7)), "step_noise": torch.zeros((10, 2, 16, 7)),
            "rng_states": torch.zeros((12, 32), dtype=torch.uint8),
            "rng_before": torch.zeros(32, dtype=torch.uint8)}


@pytest.mark.parametrize("name", tuple(example()))
def test_missing_explicit_input_rejected(name):
    values = example()
    del values[name]
    with pytest.raises(ValueError, match="complete explicit"):
        build_cogact_partition(None, values, unnorm_key="unused")


@pytest.mark.parametrize("name", tuple(example()))
def test_precision_profile_does_not_silently_cast(name):
    values = example()
    values[name] = values[name].double()
    with pytest.raises(ValueError, match="precision profile"):
        build_cogact_partition(None, values, unnorm_key="unused")


def test_all_eleven_consumption_records_required():
    values = example()
    values["rng_states"] = values["rng_states"][:-1]
    with pytest.raises(ValueError, match="all ten step states"):
        build_cogact_partition(None, values, unnorm_key="unused")


def test_cannot_replace_full_noise_chunk_with_one_action():
    values = example()
    values["step_noise"] = values["step_noise"][:, :, :1]
    with pytest.raises(ValueError, match="full-chunk"):
        build_cogact_partition(None, values, unnorm_key="unused")


def test_external_producer_cannot_claim_cpu_global_state_is_cuda():
    model = SimpleNamespace(vlm=SimpleNamespace(device=torch.device("cpu")))
    with pytest.raises(ValueError, match="real CUDA"):
        produce_rng_tape(model)


@pytest.mark.parametrize("attention,static_past", [("eager", False), ("sdpa", True)])
def test_fixed_prefill_rejects_different_attention_or_static_past(attention, static_past):
    scheduler = SimpleNamespace(num_timesteps=10, original_num_steps=100,
                                timestep_map=list(range(0, 100, 10)),
                                model_mean_type="ModelMeanType.EPSILON",
                                model_var_type="ModelVarType.FIXED_SMALL")
    layer = SimpleNamespace(self_attn=SimpleNamespace(**({"past_key_value": None} if static_past else {})))
    decoder = SimpleNamespace(config=SimpleNamespace(_attn_implementation=attention), layers=[layer])
    model = SimpleNamespace(vlm=SimpleNamespace(device=torch.device("cpu"),
                            llm_backbone=SimpleNamespace(llm=SimpleNamespace(model=decoder))),
                            action_model=SimpleNamespace(ddim_diffusion=scheduler))
    with pytest.raises(ValueError, match="native SDPA without static past state"):
        build_cogact_partition(model, example(), unnorm_key="unused")
