from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vlaforge.deployment import (  # noqa: E402
    ArtifactKind,
    ArtifactIdentity,
    BackendCapability,
    WorkspaceContract,
)
from vlaforge.frontend import (  # noqa: E402
    DynamicDimension,
    FrontendUnsupportedError,
    ModelFrontendAudit,
    PersistentStateEvidence,
    RegionAuditRecord,
    ShapeProfile,
    capture_annotated_region,
    capture_region,
    finalize_region_artifact,
    lift_persistent_states,
    load_exported_region,
    make_compile_request,
    save_exported_region,
    tensor_region,
)
from vlaforge.ir.attrs import Ownership  # noqa: E402
from vlaforge.ir.program import TensorRegion, Value  # noqa: E402
from vlaforge.ir.types import ScalarType, TensorType  # noqa: E402


F32_2X3 = TensorType((2, 3), "f32")


class PureRegion(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sin(value) * 2.0


class DynamicRegion(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.cos() + 1.0


class CorrelatedDynamicRegion(torch.nn.Module):
    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        return left + right


class HiddenRandomRegion(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + torch.randn_like(value)


class MutatingRegion(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value.add_(1.0)
        return value


class EvalDropoutRegion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropout(value)


def _capability(*, dynamic: bool = False) -> BackendCapability:
    return BackendCapability(
        backend="cpu_fixture",
        target="x86_64",
        supported_dtypes=("f32",),
        supports_dynamic_shapes=dynamic,
    )


def test_capture_pure_region_and_save_round_trip(tmp_path: Path) -> None:
    region = TensorRegion(
        "pure",
        (Value("value", F32_2X3),),
        (F32_2X3,),
    )
    example = torch.linspace(-1, 1, 6, dtype=torch.float32).reshape(2, 3)
    capture = capture_region(region, PureRegion(), (example,))

    assert capture.supported
    assert capture.report.supported
    assert capture.evidence is not None
    assert capture.evidence.effect_audit.passed
    assert capture.evidence.inputs[0].type == F32_2X3

    program = tmp_path / "pure.pt2"
    evidence = tmp_path / "pure.capture.json"
    save_exported_region(capture, program_path=program, evidence_path=evidence)
    loaded = load_exported_region(program)
    torch.testing.assert_close(
        loaded.module()(example),
        PureRegion()(example),
        atol=0,
        rtol=0,
    )
    assert evidence.read_text().endswith("\n")


def test_load_legacy_export_uses_content_addressed_mmap_cache(
    tmp_path: Path,
) -> None:
    from torch._export.serde.schema import SCHEMA_VERSION
    from torch._export.serde.serialize import serialize

    example = torch.linspace(-1, 1, 6, dtype=torch.float32).reshape(2, 3)
    exported = torch.export.export(PureRegion(), (example,))
    artifact = serialize(exported)
    legacy = tmp_path / "pure.pt2e"
    with zipfile.ZipFile(
        legacy,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr(
            "serialized_exported_program.json",
            artifact.exported_program,
        )
        archive.writestr(
            "serialized_state_dict.pt",
            artifact.state_dict,
        )
        archive.writestr(
            "serialized_constants.pt",
            artifact.constants,
        )
        archive.writestr(
            "serialized_example_inputs.pt",
            artifact.example_inputs,
        )
        archive.writestr(
            "version",
            ".".join(str(item) for item in SCHEMA_VERSION),
        )

    cache = tmp_path / "mmap-cache"
    first = load_exported_region(legacy, mmap_cache=cache)
    second = load_exported_region(legacy, mmap_cache=cache)
    torch.testing.assert_close(
        first.module()(example),
        PureRegion()(example),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        second.module()(example),
        PureRegion()(example),
        atol=0,
        rtol=0,
    )
    cached = tuple(cache.iterdir())
    assert len(cached) == 3
    assert all(path.is_file() for path in cached)


def test_capture_dynamic_shape_uses_bounded_profile() -> None:
    tensor = TensorType((None, 3), "f32")
    region = TensorRegion(
        "dynamic",
        (Value("value", tensor),),
        (tensor,),
    )
    profile = ShapeProfile(
        (DynamicDimension("value", 0, "batch", 1, 2, 4),)
    )
    output_dimensions = (("output_0", 0, "output_batch", 1, 2, 4),)
    example = torch.ones(2, 3)
    capture = capture_region(
        region,
        DynamicRegion(),
        (example,),
        shape_profile=profile,
        output_dynamic_dimensions=output_dimensions,
    )

    assert capture.supported
    assert capture.exported_program is not None
    result = capture.exported_program.module()(torch.ones(4, 3))
    assert tuple(result.shape) == (4, 3)
    assert capture.evidence is not None
    assert capture.evidence.inputs[0].dimensions[0].symbol == "batch"


def test_capture_reuses_symbol_for_correlated_dynamic_dimensions() -> None:
    tensor = TensorType((None, 3), "f32")
    region = TensorRegion(
        "correlated_dynamic",
        (
            Value("left", tensor),
            Value("right", tensor),
        ),
        (tensor,),
    )
    profile = ShapeProfile(
        (
            DynamicDimension("left", 0, "batch", 1, 2, 4),
            DynamicDimension("right", 0, "batch", 1, 2, 4),
        )
    )
    capture = capture_region(
        region,
        CorrelatedDynamicRegion(),
        (torch.ones(2, 3), torch.ones(2, 3)),
        shape_profile=profile,
        output_dynamic_dimensions=(
            ("output_0", 0, "batch", 1, 2, 4),
        ),
    )

    assert capture.supported
    assert capture.exported_program is not None
    result = capture.exported_program.module()(
        torch.ones(4, 3),
        torch.ones(4, 3),
    )
    assert tuple(result.shape) == (4, 3)


def test_shape_profile_rejects_conflicting_shared_symbol_bounds() -> None:
    with pytest.raises(ValueError, match="conflicting bounds"):
        ShapeProfile(
            (
                DynamicDimension("left", 0, "batch", 1, 2, 4),
                DynamicDimension("right", 0, "batch", 1, 2, 8),
            )
        )


def test_capture_rejects_hidden_rng_without_fallback() -> None:
    region = TensorRegion(
        "hidden_rng",
        (Value("value", F32_2X3),),
        (F32_2X3,),
    )
    outcome = capture_region(region, HiddenRandomRegion(), (torch.ones(2, 3),))

    assert not outcome.supported
    assert outcome.exported_program is None
    assert outcome.report.stage == "effect_audit"
    assert {item.code for item in outcome.report.items} == {
        "frontend.hidden_rng"
    }
    with pytest.raises(FrontendUnsupportedError):
        outcome.require_supported()


def test_capture_rejects_mutation_without_fallback() -> None:
    region = TensorRegion(
        "mutating",
        (Value("value", F32_2X3),),
        (F32_2X3,),
    )
    outcome = capture_region(region, MutatingRegion(), (torch.ones(2, 3),))

    assert not outcome.supported
    assert outcome.report.stage == "effect_audit"
    assert "frontend.hidden_mutation" in {
        item.code for item in outcome.report.items
    }


def test_eval_dropout_is_deterministic_and_allowed() -> None:
    region = TensorRegion(
        "eval_dropout",
        (Value("value", F32_2X3),),
        (F32_2X3,),
    )
    outcome = capture_region(
        region,
        EvalDropoutRegion().eval(),
        (torch.ones(2, 3),),
    )
    assert outcome.supported
    assert outcome.evidence is not None
    assert not outcome.evidence.effect_audit.hidden_rng


def test_plain_function_mutable_closure_is_rejected() -> None:
    hidden: list[int] = []

    def implementation(value: torch.Tensor) -> torch.Tensor:
        hidden.append(1)
        return value

    region = TensorRegion(
        "closure",
        (Value("value", F32_2X3),),
        (F32_2X3,),
    )
    outcome = capture_region(region, implementation, (torch.ones(2, 3),))
    assert not outcome.supported
    assert outcome.report.stage == "preflight"
    assert outcome.report.items[0].code == "frontend.nonserializable_closure"


def test_annotated_function_capture() -> None:
    @tensor_region(
        "annotated",
        inputs=(Value("value", F32_2X3),),
        outputs=(F32_2X3,),
    )
    def implementation(value: torch.Tensor) -> torch.Tensor:
        return value.relu()

    outcome = capture_annotated_region(
        implementation, (torch.tensor([[-1.0, 0, 1]]).repeat(2, 1),)
    )
    assert outcome.supported


def test_compile_request_and_artifact_finalization(tmp_path: Path) -> None:
    region = TensorRegion(
        "compiled",
        (Value("value", F32_2X3),),
        (F32_2X3,),
    )
    capture = capture_region(region, PureRegion(), (torch.ones(2, 3),))
    request = make_compile_request(
        capture,
        region_id=4,
        artifact_kind=ArtifactKind.CPU_FIXTURE,
        output_path="artifacts/compiled.bin",
        capability=_capability(),
        io_schema_digest="2" * 64,
        identity=ArtifactIdentity(
            model_name="frontend-fixture",
            upstream_revision="fixture-revision",
            checkpoint_identity="fixture:no-checkpoint",
            graph_sha256=capture.evidence.graph_digest,
        ),
        workspace=WorkspaceContract(64, 64, "cpu"),
        backend_options={"opt_level": "2"},
        backend_variant="fixture",
    )

    artifact_path = tmp_path / request.output_path
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"compiled fixture")
    artifact = finalize_region_artifact(request, tmp_path)

    assert artifact.region_id == 4
    assert artifact.artifact_sha256 == hashlib.sha256(
        b"compiled fixture"
    ).hexdigest()
    assert request == type(request).from_dict(request.to_dict())
    assert request.digest() == type(request).from_dict(request.to_dict()).digest()


def test_state_lifting_requires_source_evidence() -> None:
    state = lift_persistent_states(
        (
            PersistentStateEvidence(
                name="action_queue",
                payload=TensorType((4, 2), "f32"),
                source_location="policy.py:select_action",
                cross_run_reason="queue survives successive Session::Run calls",
                retention=3,
            ),
        )
    )[0]
    assert state.name == "action_queue"
    assert state.ownership is Ownership.HOST
    assert state.reset_on_episode

    with pytest.raises(ValueError, match="source location"):
        PersistentStateEvidence(
            name="invented",
            payload=ScalarType("i64"),
            source_location="",
            cross_run_reason="no evidence",
            retention=1,
        )


def test_shape_profile_rejects_example_outside_bounds() -> None:
    tensor = TensorType((None, 3), "f32")
    region = TensorRegion(
        "dynamic",
        (Value("value", tensor),),
        (tensor,),
    )
    outcome = capture_region(
        region,
        DynamicRegion(),
        (torch.ones(5, 3),),
        shape_profile=ShapeProfile(
            (DynamicDimension("value", 0, "batch", 1, 2, 4),)
        ),
    )
    assert not outcome.supported
    assert outcome.report.stage == "shape_profile"


def test_model_audit_report_is_versioned_and_deterministic(
    tmp_path: Path,
) -> None:
    record = RegionAuditRecord(
        name="forward",
        source_location="model.py:10",
        major_compute=True,
        supported=True,
        graph_digest="a" * 64,
        graph_nodes=7,
        export_seconds=0.5,
        maximum_absolute_error=0.0,
        effect_audit={"hidden_mutation": False},
        unsupported_report=None,
    )
    report = ModelFrontendAudit(
        model="fixture",
        checkpoint_path="/fixture/model.safetensors",
        checkpoint_revision="revision",
        checkpoint_digests=(("model.safetensors", "b" * 64),),
        torch_version=torch.__version__,
        device="cpu",
        persistent_states=(),
        persistent_state_evidence_complete=True,
        regions=(record,),
        validation_checks=(("outputs_equal", "true"),),
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    report.write(first)
    report.write(second)

    assert report.passed
    assert first.read_bytes() == second.read_bytes()
    assert b'"schema": "vlaforge.frontend_model_audit/2"' in first.read_bytes()

    with pytest.raises(ValueError, match="sorted"):
        ModelFrontendAudit(
            model="fixture",
            checkpoint_path="/fixture/model.safetensors",
            checkpoint_revision="revision",
            checkpoint_digests=(
                ("z.safetensors", "c" * 64),
                ("a.safetensors", "d" * 64),
            ),
            torch_version=torch.__version__,
            device="cpu",
            persistent_states=(),
            persistent_state_evidence_complete=True,
            regions=(record,),
        )
