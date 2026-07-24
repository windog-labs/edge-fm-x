"""Explicit torch.export capture for a declared pure TensorRegion."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from vlaforge.deployment.contract import EffectAudit, ValueContract
from vlaforge.frontend.effect_audit import (
    audit_callable_closure,
    audit_exported_program,
)
from vlaforge.frontend.shape_profile import ShapeProfile
from vlaforge.frontend.unsupported import (
    FrontendUnsupportedError,
    UnsupportedItem,
    UnsupportedReport,
)
from vlaforge.ir.program import TensorRegion, Value


CAPTURE_SCHEMA = "vlaforge.frontend_capture/1"


@dataclass(frozen=True, slots=True)
class CaptureEvidence:
    region_name: str
    graph_digest: str
    torch_version: str
    strict_export: bool
    export_seconds: float
    maximum_absolute_error: float
    inputs: tuple[ValueContract, ...]
    outputs: tuple[ValueContract, ...]
    effect_audit: EffectAudit
    schema: str = CAPTURE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CAPTURE_SCHEMA:
            raise ValueError(f"unsupported capture schema: {self.schema!r}")
        if len(self.graph_digest) != 64:
            raise ValueError("capture graph digest must be SHA-256")
        if self.export_seconds < 0 or self.maximum_absolute_error < 0:
            raise ValueError("capture timing and error must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "region_name": self.region_name,
            "graph_digest": self.graph_digest,
            "torch_version": self.torch_version,
            "strict_export": self.strict_export,
            "export_seconds": self.export_seconds,
            "maximum_absolute_error": self.maximum_absolute_error,
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "effect_audit": self.effect_audit.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CaptureOutcome:
    region: TensorRegion
    exported_program: Any | None
    evidence: CaptureEvidence | None
    report: UnsupportedReport

    @property
    def supported(self) -> bool:
        return (
            self.exported_program is not None
            and self.evidence is not None
            and self.report.supported
        )

    def require_supported(self) -> "CaptureOutcome":
        if not self.supported:
            raise FrontendUnsupportedError(self.report)
        return self


def capture_region(
    region: TensorRegion,
    implementation: Any,
    example_args: tuple[object, ...],
    *,
    shape_profile: ShapeProfile = ShapeProfile(),
    input_devices: Mapping[str, str] | None = None,
    output_devices: Mapping[str, str] | None = None,
    input_alignments: Mapping[str, int] | None = None,
    output_alignments: Mapping[str, int] | None = None,
    output_dynamic_dimensions: tuple[
        tuple[str, int, str, int, int, int], ...
    ] = (),
    strict: bool = True,
    explicit_rng: bool = False,
    lifted_states: tuple[str, ...] = (),
    compare_outputs: bool = True,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> CaptureOutcome:
    """Capture one declared region without an implicit eager fallback."""

    import torch

    if not region.pure:
        return _unsupported(
            region,
            "preflight",
            UnsupportedItem(
                "frontend.region_not_pure",
                "TensorRegion declaration is not pure",
            ),
        )
    if len(region.inputs) != len(example_args):
        return _unsupported(
            region,
            "preflight",
            UnsupportedItem(
                "frontend.example_arity",
                f"expected {len(region.inputs)} example args, got {len(example_args)}",
            ),
        )
    try:
        shape_profile.validate_examples(region.inputs, example_args)
    except Exception as exc:
        return _unsupported(
            region,
            "shape_profile",
            UnsupportedItem("frontend.shape_profile", str(exc)),
        )

    module, closure_diagnostics = _as_module(implementation, torch)
    if module is None:
        return _unsupported(
            region,
            "preflight",
            *(
                UnsupportedItem(
                    item.code,
                    item.message,
                    source=item.source,
                    remediation="pass an nn.Module or remove mutable closure state",
                )
                for item in closure_diagnostics
            ),
        )

    try:
        dynamic_shapes = shape_profile.torch_dynamic_shapes(region.inputs)
        export_started = time.perf_counter()
        exported = torch.export.export(
            module,
            example_args,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )
        export_seconds = time.perf_counter() - export_started
    except Exception as exc:
        return _unsupported(
            region,
            "torch.export",
            UnsupportedItem(
                "frontend.export_failed",
                f"{type(exc).__name__}: {exc}",
                source=type(module).__qualname__,
                remediation=(
                    "split the source at an explicit pure TensorRegion boundary "
                    "or record this region as unsupported"
                ),
            ),
        )

    audit = audit_exported_program(
        exported,
        closure_diagnostics=closure_diagnostics,
        explicit_rng=explicit_rng,
        lifted_states=lifted_states,
    )
    if not audit.passed:
        return CaptureOutcome(
            region=region,
            exported_program=None,
            evidence=None,
            report=UnsupportedReport(
                region=region.name,
                stage="effect_audit",
                items=tuple(
                    UnsupportedItem(
                        item.code,
                        item.message,
                        source=item.source,
                        remediation=(
                            "make RNG/state explicit and remove mutable or I/O effects"
                        ),
                    )
                    for item in audit.diagnostics
                ),
            ),
        )

    input_device_map = dict(input_devices or _example_devices(region.inputs, example_args))
    try:
        input_contracts = shape_profile.value_contracts(
            region.inputs,
            devices=input_device_map,
            alignments=input_alignments,
        )
        eager_output = module(*example_args)
        exported_output = exported.module()(*example_args)
        eager_values = _as_output_tuple(eager_output)
        exported_values = _as_output_tuple(exported_output)
        if len(eager_values) != len(region.outputs):
            raise ValueError(
                f"region declares {len(region.outputs)} outputs, implementation "
                f"returned {len(eager_values)}"
            )
        output_values = tuple(
            Value(f"output_{index}", type)
            for index, type in enumerate(region.outputs)
        )
        output_profile = ShapeProfile(
            tuple(
                _dynamic_dimension_from_tuple(item)
                for item in output_dynamic_dimensions
            )
        )
        output_profile.validate_examples(output_values, eager_values)
        output_device_map = dict(
            output_devices
            or _example_devices(output_values, eager_values)
        )
        output_contracts = output_profile.value_contracts(
            output_values,
            devices=output_device_map,
            alignments=output_alignments,
        )
        maximum_absolute_error = _maximum_absolute_error(
            eager_values, exported_values, torch=torch
        )
        if compare_outputs:
            _assert_outputs_close(
                eager_values,
                exported_values,
                torch=torch,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
    except Exception as exc:
        return _unsupported(
            region,
            "contract_validation",
            UnsupportedItem(
                "frontend.contract_mismatch",
                f"{type(exc).__name__}: {exc}",
            ),
        )

    evidence = CaptureEvidence(
        region_name=region.name,
        graph_digest=_graph_digest(exported),
        torch_version=torch.__version__,
        strict_export=strict,
        export_seconds=export_seconds,
        maximum_absolute_error=maximum_absolute_error,
        inputs=input_contracts,
        outputs=output_contracts,
        effect_audit=audit,
    )
    return CaptureOutcome(
        region=region,
        exported_program=exported,
        evidence=evidence,
        report=UnsupportedReport(region.name, "complete", ()),
    )


def capture_annotated_region(
    implementation: Callable[..., object],
    example_args: tuple[object, ...],
    **kwargs: Any,
) -> CaptureOutcome:
    spec = getattr(implementation, "__vlaforge_region__", None)
    if spec is None:
        region = TensorRegion(
            getattr(implementation, "__name__", "unknown"),
            (),
            (),
        )
        return _unsupported(
            region,
            "annotation",
            UnsupportedItem(
                "frontend.missing_annotation",
                "callable has no @tensor_region declaration",
            ),
        )
    return capture_region(spec.as_ir(), implementation, example_args, **kwargs)


def _as_module(
    implementation: Any, torch: Any
) -> tuple[Any | None, tuple[Any, ...]]:
    if isinstance(implementation, torch.nn.Module):
        return implementation.eval(), ()
    if not callable(implementation):
        return None, ()
    closure_diagnostics = audit_callable_closure(implementation)
    if any(
        item.severity.value == "error" for item in closure_diagnostics
    ):
        return None, closure_diagnostics

    class CallableModule(torch.nn.Module):
        def forward(self, *args: object) -> object:
            return implementation(*args)

    return CallableModule().eval(), closure_diagnostics


def _unsupported(
    region: TensorRegion, stage: str, *items: UnsupportedItem
) -> CaptureOutcome:
    if not items:
        items = (
            UnsupportedItem(
                "frontend.unsupported_implementation",
                "implementation is not an nn.Module or callable",
            ),
        )
    return CaptureOutcome(
        region=region,
        exported_program=None,
        evidence=None,
        report=UnsupportedReport(region.name, stage, tuple(items)),
    )


def _example_devices(
    values: tuple[Value, ...], examples: tuple[object, ...]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for value, example in zip(values, examples, strict=True):
        device = getattr(example, "device", None)
        result[value.name] = "cpu" if device is None else str(device)
    return result


def _as_output_tuple(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, tuple | list) else (value,)


def _assert_outputs_close(
    eager: tuple[object, ...],
    exported: tuple[object, ...],
    *,
    torch: Any,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if len(eager) != len(exported):
        raise AssertionError(
            f"eager returned {len(eager)} values, export returned {len(exported)}"
        )
    for expected, actual in zip(eager, exported, strict=True):
        if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
            torch.testing.assert_close(
                expected,
                actual,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
        elif expected != actual:
            raise AssertionError(f"eager output {expected!r} != export {actual!r}")


def _maximum_absolute_error(
    eager: tuple[object, ...],
    exported: tuple[object, ...],
    *,
    torch: Any,
) -> float:
    maximum = 0.0
    for expected, actual in zip(eager, exported, strict=True):
        if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
            if expected.dtype == torch.bool or actual.dtype == torch.bool:
                error = 0.0 if torch.equal(expected, actual) else 1.0
            elif expected.numel() == 0:
                error = 0.0
            else:
                error = float(
                    (expected.detach().to(torch.float64)
                     - actual.detach().to(torch.float64))
                    .abs()
                    .max()
                    .cpu()
                )
            maximum = max(maximum, error)
        elif expected != actual:
            maximum = float("inf")
    return maximum


def _graph_digest(exported: Any) -> str:
    graph = []
    for node in exported.graph_module.graph.nodes:
        graph.append(
            {
                "op": node.op,
                "name": node.name,
                "target": str(node.target),
                "args": repr(node.args),
                "kwargs": repr(node.kwargs),
            }
        )
    signature = {
        "inputs": [str(item) for item in exported.graph_signature.input_specs],
        "outputs": [str(item) for item in exported.graph_signature.output_specs],
    }
    payload = json.dumps(
        {"graph": graph, "signature": signature},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _dynamic_dimension_from_tuple(
    item: tuple[str, int, str, int, int, int]
) -> Any:
    from vlaforge.frontend.shape_profile import DynamicDimension

    value, index, symbol, minimum, optimum, maximum = item
    return DynamicDimension(value, index, symbol, minimum, optimum, maximum)
