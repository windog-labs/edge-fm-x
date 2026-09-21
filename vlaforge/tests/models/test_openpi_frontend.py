"""Source/Interface tests are not pretrained-model evidence.

The final opt-in test requires actual converted weights and a saved input/noise
bundle. No random miniature VLA is used as a substitute for that test.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from vlaforge.adapters.openpi.openpi_checkpoint import (
    OPENPI_REVISION,
    file_digest,
    load_converted_state_dict,
    verify_gcs_checkpoint,
    verify_openpi_source,
)
from vlaforge.adapters.openpi.openpi_frontend import (
    OpenPIConfig,
    _image_memory_format,
    _restore_image_memory_format,
    build_openpi_frontend,
    capture_openpi_frontend,
    compose_openpi_invocation,
    load_openpi,
    prepare_openpi_inputs,
)
from vlaforge.compiler import compile_module
from vlaforge.frontend import capture_region, tensor_region


@pytest.mark.parametrize("channels_last", [False, True])
def test_image_boundary_restores_recorded_layout(channels_last):
    torch = pytest.importorskip("torch")
    image = torch.arange(60, dtype=torch.float32).reshape(1, 3, 4, 5)
    if channels_last:
        image = image.contiguous(memory_format=torch.channels_last)
    memory_format = _image_memory_format(image)
    assert memory_format == ("channels_last" if channels_last else "contiguous")
    restored = _restore_image_memory_format(image.contiguous(), memory_format)
    assert restored.stride() == image.stride()
    torch.testing.assert_close(restored, image, atol=0, rtol=0)


def test_image_boundary_rejects_unrepresented_layout():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="unsupported"):
        _image_memory_format(torch.zeros(1, 3, 4, 10)[..., ::2])
    with pytest.raises(ValueError, match="rank-four"):
        _image_memory_format(torch.zeros(3, 4, 5))
    with pytest.raises(ValueError, match="unsupported"):
        _restore_image_memory_format(torch.zeros(1, 3, 4, 5), "unknown")


from vlaforge.ir.program import InputPort, TensorRegion, Value
from vlaforge.ir.types import TensorType
from vlaforge.plan import PlanModule, verify_plan


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(
    os.getenv("VLAFORGE_OPENPI_SOURCE_ROOT", str(ROOT / "third_party/openpi"))
)


def _inventory(tmp_path):
    root = tmp_path / "pi0_base"
    items = []
    for name, payload in (
        ("params/part", b"checkpoint bytes"),
        ("assets/trossen/norm_stats.json", b'{"mean": [0]}'),
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        items.append(
            {
                "name": f"checkpoints/pi0_base/{name}",
                "size": str(len(payload)),
                "generation": "123",
                "md5Hash": base64.b64encode(hashlib.md5(payload).digest()).decode(),
            }
        )
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps({"items": items}))
    return root, manifest, items


def test_every_checkpoint_object_is_checked_before_conversion(tmp_path):
    root, manifest, _ = _inventory(tmp_path)
    result = verify_gcs_checkpoint(root, manifest, checkpoint_name="pi0_base")
    assert len(result["objects"]) == 2
    assert result["inventory"] == file_digest(manifest)
    for item in result["objects"]:
        assert len(item["local"]["sha256"]) == 64
    (root / "params/part").write_bytes(b"modified content")
    with pytest.raises(ValueError, match="mismatch"):
        verify_gcs_checkpoint(root, manifest, checkpoint_name="pi0_base")


@pytest.mark.parametrize(
    "failure",
    ["pagination", "duplicate", "escape", "foreign", "missing_md5", "no_assets"],
)
def test_incomplete_or_unsafe_inventory_is_rejected(tmp_path, failure):
    root, manifest, items = _inventory(tmp_path)
    payload = {"items": items}
    if failure == "pagination":
        payload["nextPageToken"] = "another-page"
    elif failure == "duplicate":
        items.append(items[0])
    elif failure == "escape":
        items[0]["name"] = "checkpoints/pi0_base/../outside"
    elif failure == "foreign":
        items[0]["name"] = "checkpoints/pi05_base/params/part"
    elif failure == "missing_md5":
        items[0].pop("md5Hash")
    elif failure == "no_assets":
        items.pop()
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        verify_gcs_checkpoint(root, manifest, checkpoint_name="pi0_base")


def test_strict_conversion_gate_rejects_missing_and_extra_real_parameters():
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(2, 2)
    state = module.state_dict()
    with pytest.raises(ValueError, match="missing executable"):
        load_converted_state_dict(module, {"weight": state["weight"]})
    with pytest.raises(ValueError, match="unexpected"):
        load_converted_state_dict(module, {**state, "extra": torch.zeros(1)})
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_converted_state_dict(module, {**state, "weight": torch.zeros(3, 3)})


def test_strict_conversion_resolves_only_actual_ties_and_declared_unused_weights():
    torch = pytest.importorskip("torch")
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(2))
    module.alias = module.weight
    module.unused = torch.nn.Parameter(torch.full((2,), 8.0))
    result = load_converted_state_dict(
        module, {"weight": torch.tensor([2.0, 3.0])}, unused_parameters=("unused",)
    )
    assert result["missing_required"] == []
    assert result["resolved_tied_aliases"] == {"alias": "weight"}
    assert result["zeroed_unused_parameters"] == {"unused": 2}
    assert torch.equal(module.weight, torch.tensor([2.0, 3.0]))
    assert torch.equal(module.unused, torch.zeros(2))
    with pytest.raises(ValueError, match="unused parameter declaration"):
        load_converted_state_dict(
            module, module.state_dict(), unused_parameters=("unknown",)
        )
    with pytest.raises(ValueError, match="conflicting tied"):
        load_converted_state_dict(
            module, {**module.state_dict(), "alias": torch.zeros(2)}
        )


@pytest.mark.parametrize("num_steps", [0, -1, 1.5, True])
def test_profile_requires_a_positive_bound(num_steps):
    with pytest.raises(ValueError, match="static"):
        OpenPIConfig(
            Path("source"), Path("weights"), "pi0_aloha", "0" * 64, num_steps=num_steps
        )


def test_profile_requires_a_pinned_weight_digest():
    with pytest.raises(ValueError, match="sha256"):
        OpenPIConfig(Path("source"), Path("weights"), "pi0_aloha", "not-a-hash")


@pytest.mark.parametrize(
    "failure",
    [
        "schema",
        "load_gate",
        "digest",
        "missing_assets",
        "asset_tamper",
        "tokenizer_missing",
        "tokenizer_tamper",
    ],
)
def test_checkpoint_load_rejects_incomplete_evidence_before_model_construction(
    tmp_path, failure, monkeypatch
):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path / "processor-cache"))
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"not real model bytes; integrity gate test only")
    asset = tmp_path / "assets/trossen/norm_stats.json"
    asset.parent.mkdir(parents=True)
    asset.write_text('{"mean": [0]}')
    report = {
        "schema": "vlaforge.openpi_conversion/1",
        "source": {"revision": OPENPI_REVISION},
        "load_gate": {"strict": True, "missing_required": [], "unexpected": []},
        "checkpoint": file_digest(weight),
        "assets": {"assets/trossen/norm_stats.json": file_digest(asset)},
    }
    expected_sha = report["checkpoint"]["sha256"]
    if failure == "schema":
        report["schema"] = "unverified"
    elif failure == "load_gate":
        report["load_gate"]["missing_required"] = ["action_out_proj.weight"]
    elif failure == "digest":
        expected_sha = "0" * 64
    elif failure == "missing_assets":
        report["assets"] = {}
    elif failure == "asset_tamper":
        asset.write_text('{"mean": [8]}')
    elif failure == "tokenizer_tamper":
        tokenizer = tmp_path / "processor-cache/big_vision/paligemma_tokenizer.model"
        tokenizer.parent.mkdir(parents=True)
        tokenizer.write_bytes(b"unverified tokenizer")
    (tmp_path / "vlaforge_conversion.json").write_text(json.dumps(report))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_openpi(OpenPIConfig(SOURCE, tmp_path, "pi0_aloha", expected_sha))


def _declaration(name, inputs, outputs):
    @tensor_region(name, inputs=inputs, outputs=outputs)
    def schema_only(*_):
        raise AssertionError("schema-only Interface test must not execute a model")

    return schema_only


def _stages(*, pi05):
    tokens = 200 if pi05 else 48
    image = TensorType((1, 3, 224, 224), "f32")
    mask = TensorType((1,), "bool")
    state = TensorType((1, 32), "f32")
    sample = TensorType((1, 50, 32), "f32")
    time = TensorType((), "f32")
    prefix_len = 3 * 256 + tokens
    padding = TensorType((1, prefix_len), "bool")
    cache = TensorType((1, 1, prefix_len, 256), "bf16")
    ports = (
        *(InputPort(f"image_{index}", image) for index in range(3)),
        *(InputPort(f"image_mask_{index}", mask) for index in range(3)),
        InputPort("language_tokens", TensorType((1, tokens), "i64")),
        InputPort("language_mask", TensorType((1, tokens), "bool")),
        InputPort("state", state),
        InputPort("noise", sample),
    )
    prefix_inputs = tuple(
        Value(port.name, port.payload) for port in ports[: 9 if pi05 else 8]
    )
    prefix = _declaration("openpi_prefix", prefix_inputs, (padding, *((cache,) * 36)))
    step = _declaration(
        "openpi_step",
        (
            Value("state", state),
            Value("sample", sample),
            Value("time", time),
            Value("prefix_padding", padding),
            *(Value(f"cache_{index}", cache) for index in range(36)),
        ),
        (sample, time),
    )
    initialize = _declaration("openpi_time", (Value("noise", sample),), (time,))
    finite = _declaration("openpi_finite", (Value("sample", sample),), (mask,))
    return dict(
        ports=ports, prefix=prefix, step=step, initialize=initialize, finite=finite
    )


@pytest.mark.parametrize("pi05", [False, True])
def test_both_source_profiles_use_generic_variadic_ir_and_exact_dependencies(pi05):
    program = compose_openpi_invocation(
        **_stages(pi05=pi05), num_steps=10, state_in_prefix=pi05
    )
    compilation = compile_module(program.module, profile="verified")
    assert verify_plan(compilation.plan, raise_on_error=False) == ()
    assert PlanModule.from_dict(compilation.plan.to_dict()) == compilation.plan
    (cache,) = compilation.certificate.caches
    assert sorted(cache.input_ids) == list(range(9 if pi05 else 8))
    assert cache.state_ids == ()
    loop = next(task for task in compilation.plan.tasks if task.opcode == "vla.for")
    assert len(loop.outputs) == 2
    assert len(loop.attributes["carry_scratch"]) == 2
    assert loop.attributes["upper"] == 10
    assert program.module.outputs[0].payload.shape == (1, 50, 32)
    assert (
        program.module.invocations[0].metadata["measurement_boundary"]
        == "prepared-tensors-to-normalized-complete-chunk"
    )


def test_state_tokenized_context_cannot_omit_state_dependency():
    with pytest.raises(ValueError, match="state dependency"):
        compose_openpi_invocation(
            **_stages(pi05=False), num_steps=10, state_in_prefix=True
        )


def test_actual_pinned_source_is_clean_when_available():
    if not SOURCE.is_dir():
        pytest.skip("optional pinned OpenPI source checkout not available")
    result = verify_openpi_source(SOURCE)
    assert result["revision"] == OPENPI_REVISION
    assert len(result["files"]) >= 8


def test_official_mask_and_timestep_utilities_export_not_a_real_model():
    if not SOURCE.is_dir():
        pytest.skip("optional pinned OpenPI source checkout not available")
    torch = pytest.importorskip("torch")
    path = SOURCE / "src/openpi/models_pytorch/pi0_pytorch.py"
    parsed = ast.parse(path.read_text())
    functions = {
        "get_safe_dtype",
        "create_sinusoidal_pos_embedding",
        "make_att_2d_masks",
    }
    selected = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in functions
    ]
    assert len(selected) == len(functions)
    namespace = {"torch": torch, "math": math, "Tensor": torch.Tensor}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )

    class OfficialUtilities(torch.nn.Module):
        def forward(self, padding, attention, time):
            return (
                namespace["make_att_2d_masks"](padding, attention),
                namespace["create_sinusoidal_pos_embedding"](
                    time, 8, 4e-3, 4.0, device=time.device
                ),
            )

    region = TensorRegion(
        "source_utilities",
        (
            Value("padding", TensorType((1, 4), "bool")),
            Value("attention", TensorType((1, 4), "i64")),
            Value("time", TensorType((1,), "f32")),
        ),
        (TensorType((1, 4, 4), "bool"), TensorType((1, 8), "f32")),
    )
    inputs = (
        torch.tensor([[True, True, True, False]]),
        torch.tensor([[0, 0, 1, 0]]),
        torch.tensor([1.0]),
    )
    result = capture_region(region, OfficialUtilities(), inputs, strict=True)
    result.require_supported()
    mask, _ = result.exported_program.module()(*inputs)
    assert not mask[0, 0, 2].item()
    assert mask[0, 2, 0].item()
    assert not mask[0, :, 3].any().item()


def test_real_openpi_saved_input_full_chunk_and_capture():
    config_path = os.environ.get("VLAFORGE_OPENPI_REAL_CONFIG")
    if not config_path:
        pytest.skip(
            "requires real OpenPI converted checkpoint and saved observation/noise NPZ"
        )
    from vlaforge.adapters.openpi.openpi_inputs import load_openpi_input_pack

    run = json.loads(Path(config_path).read_text())
    config = OpenPIConfig(**run["adapter"])
    loaded = load_openpi(config)
    observation, noise, _ = load_openpi_input_pack(
        run["input_manifest"], source_root=config.source_root
    )
    prepared = prepare_openpi_inputs(loaded, observation, noise=noise)
    frontend = build_openpi_frontend(loaded, prepared, **run["tolerances"])
    assert frontend.normalized_reference.shape[1] == loaded.model.config.action_horizon
    for outcome in capture_openpi_frontend(frontend, **run["tolerances"]):
        outcome.require_supported()


def test_official_processor_float64_state_has_a_lossless_ir_type():
    torch = pytest.importorskip("torch")
    from vlaforge.adapters.openpi.openpi_frontend import _tensor_type

    assert _tensor_type(torch.empty(1, 32, dtype=torch.float64)) == TensorType(
        (1, 32), "f64"
    )
