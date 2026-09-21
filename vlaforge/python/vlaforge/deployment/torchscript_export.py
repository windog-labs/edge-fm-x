"""Static exported Regions as explicitly unoptimized, device-preserving ATen."""

from __future__ import annotations

import hashlib
import os
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

VARIANT = "torchscript-aten/1"


def trace_boolean_fill_compatible(module, examples, **trace_options):
    """Normalize Python boolean factory scalars for JIT's Scalar overload.

    This opt-in compatibility helper returns an unaccepted trace and a ledger.
    It grants no effect, numerical-policy or deployment certificate. Integer
    zero/one preserve boolean fill values; explicit dtype preserves inference.
    """
    import torch
    from torch.overrides import TorchFunctionMode

    ledger = {"schema": "vlaforge.boolean_fill_trace/1", "normalized_boolean_fills": 0,
              "numerical_certificate": False, "full_model_verified": False}

    class BooleanFillMode(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if function is torch.full:
                fill = args[1] if len(args) > 1 else kwargs.get("fill_value")
                if type(fill) is bool:
                    if len(args) > 1:
                        args = (args[0], int(fill), *args[2:])
                    else:
                        kwargs["fill_value"] = int(fill)
                    if kwargs.get("dtype") is None and kwargs.get("out") is None:
                        kwargs["dtype"] = torch.bool
                    ledger["normalized_boolean_fills"] += 1
            return function(*args, **kwargs)

    with BooleanFillMode():
        traced = torch.jit.trace(module, examples, **trace_options)
    return traced, ledger


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _outputs(value):
    import torch

    values = (value,) if isinstance(value, torch.Tensor) else value
    if not isinstance(values, tuple) or not values or any(
        not isinstance(item, torch.Tensor) for item in values
    ):
        raise ValueError("TorchScript profile requires a tensor or flat tensor tuple")
    return values


def _bytes(value):
    import torch

    packed = torch.empty(value.numel(), dtype=value.dtype, device=value.device)
    return packed.copy_(value.reshape(-1)).view(torch.uint8)


def _validate_inputs(inputs, examples):
    import torch

    if not isinstance(inputs, tuple) or len(inputs) != len(examples):
        raise ValueError("TorchScript profile requires positional tensor inputs")
    for value, example in zip(inputs, examples, strict=True):
        if (not isinstance(value, torch.Tensor) or value.shape != example.shape
                or value.dtype != example.dtype or value.device != example.device
                or not value.is_contiguous()):
            raise ValueError("TorchScript input changed shape/dtype/device/contiguous ABI")


def _validate_effects(program):
    import torch

    from vlaforge.frontend.effect_audit import audit_exported_program

    audits = []
    for name, graph_module in program.graph_module.named_modules():
        if not isinstance(graph_module, torch.fx.GraphModule):
            continue
        audit = audit_exported_program(SimpleNamespace(
            graph_module=graph_module, graph_signature=program.graph_signature,
        ))
        if not audit.passed:
            details = "; ".join(item.message for item in audit.diagnostics)
            raise ValueError(f"TorchScript Region effect audit failed: {details}")
        # The shared alias audit follows positional write arguments. Reject
        # keyword writes conservatively rather than executing an unaudited write.
        for node in graph_module.graph.nodes:
            schema = getattr(node.target, "_schema", None)
            if schema is not None and schema.is_mutable and any(
                argument.alias_info is not None and argument.alias_info.is_write
                and argument.name in node.kwargs
                for argument in schema.arguments
            ):
                raise ValueError("TorchScript Region cannot contain unaudited keyword mutation")
        audits.append({"graph": name, **audit.to_dict()})
    return audits


def export_torchscript_region(program, output: str | Path, *, validation_cases=()):
    """Trace an already exported static Region, then validate the saved archive.

    This is compilation-time evidence for the provided Region cases only. It
    neither verifies full-model quality nor implements a numerical provider.
    Device relocation is intentionally absent: a CPU scalar and a CUDA scalar
    with identical bytes need not have identical mixed-precision arithmetic.
    """
    import torch

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    non_static_constraints = {
        name: constraint
        for name, constraint in program.range_constraints.items()
        if not callable(getattr(constraint, "is_singleton", None))
        or not constraint.is_singleton()
    }
    if non_static_constraints:
        raise ValueError(
            "TorchScript profile requires static exported shapes: "
            f"{non_static_constraints}"
        )
    signature = program.graph_signature
    if signature.buffers_to_mutate or signature.user_inputs_to_mutate:
        raise ValueError("TorchScript Region cannot mutate caller inputs or persistent buffers")
    effects = _validate_effects(program)
    examples, kwargs = program.example_inputs
    if kwargs or not examples:
        raise ValueError("TorchScript profile requires positional tensor inputs")
    _validate_inputs(examples, examples)
    cases = (examples, *tuple(validation_cases))
    for values in cases:
        _validate_inputs(values, examples)
    module = program.module()
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = {"schema": "vlaforge.torchscript_compile/1", "backend_variant": VARIANT,
             "jit_optimization": False, "archive_device_policy": "preserve",
             "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
             "accepted_static_range_constraints": {
                 str(name): str(constraint)
                 for name, constraint in program.range_constraints.items()
             },
             "effect_audits": effects,
             "full_model_verified": False, "numerical_provider_enforcement": False,
             "external_cuda_graph": False, "validation_cases": []}
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".pt", delete=False) as file:
        temporary = Path(file.name)
    try:
        with torch.inference_mode(), torch.jit.optimized_execution(False), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            traced = torch.jit.trace(module, examples, strict=True, check_trace=True)
            if "prim::PythonOp" in str(traced.inlined_graph):
                raise ValueError("TorchScript archive retains Python execution")
            torch.jit.save(traced, str(temporary))
            loaded = torch.jit.load(str(temporary))
            for inputs in cases:
                expected, actual = _outputs(module(*inputs)), _outputs(loaded(*inputs))
                same = len(expected) == len(actual) and all(
                    left.shape == right.shape and left.dtype == right.dtype
                    and left.device == right.device
                    and torch.equal(_bytes(left), _bytes(right))
                    for left, right in zip(expected, actual, strict=True)
                )
                audit["validation_cases"].append({"bitwise_equal": same, "output_tensors": len(actual)})
                if not same:
                    raise ValueError("saved TorchScript Region differs from exported execution")
            audit["warnings"] = [str(item.message) for item in caught]
        # Exclusive publication must not remove another writer's destination.
        os.link(temporary, output)
        audit.update(status="region_cases_passed", artifact_sha256=_digest(output),
                     artifact_size_bytes=output.stat().st_size,
                     implementation_sha256=_digest(Path(__file__)))
        return audit
    finally:
        temporary.unlink(missing_ok=True)


def _program_target(program) -> str:
    import torch
    from torch.utils._pytree import tree_leaves

    devices = {value.device for value in tree_leaves(
        (program.example_inputs, program.state_dict, program.constants)
    ) if isinstance(value, torch.Tensor)}
    if any(device.type not in {"cpu", "cuda"} for device in devices):
        raise ValueError("TorchScript numerical target supports CPU or CUDA tensors")
    cuda_devices = {device for device in devices if device.type == "cuda"}
    if len(cuda_devices) > 1:
        raise ValueError("TorchScript numerical target requires at most one CUDA device")
    if not cuda_devices:
        return "cpu"
    major, minor = torch.cuda.get_device_capability(next(iter(cuda_devices)))
    return f"sm_{major}{minor}"


def compile_torchscript_region(
    exported_program_path: str | Path,
    output: str | Path,
    *,
    reference_context,
    validation_cases=(),
    target: str | None = None,
):
    """Bind a same-precision compilation to measured bytes and current policy.

    The caller supplies an observed reference context; this function does not
    restore flags or authenticate the reference run. The returned compile record
    still requires a runtime provider and independent full-model validation.
    """
    import torch

    from vlaforge.frontend.region_capture import exported_graph_digest
    from vlaforge.deployment.libtorch_numerical import policy_from_context
    from vlaforge.deployment.numerical import NumericalCompileRecord, canonical_json
    from vlaforge.numerical_context import snapshot

    source, output = Path(exported_program_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    reference_policy = policy_from_context(reference_context)
    reference_context.require_current()
    before = snapshot()
    source_sha = _digest(source)
    program = torch.export.load(source)
    actual_target = _program_target(program)
    if target is not None and target != actual_target:
        raise ValueError(f"requested target {target!r} differs from actual {actual_target!r}")
    graph_sha = exported_graph_digest(program)
    reference_context.require_current()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Only publish after every identity and policy check; a rejected compilation
    # must leave no apparently usable destination archive behind.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".torchscript-") as directory:
        staged = Path(directory) / "region.pt"
        audit = export_torchscript_region(program, staged, validation_cases=validation_cases)
        after = snapshot()
        if after != before:
            raise ValueError("numerical context changed during TorchScript compilation")
        reference_context.require_current()
        if _digest(source) != source_sha:
            raise ValueError("exported program changed during TorchScript compilation")
        artifact_sha, artifact_size = _digest(staged), staged.stat().st_size
        record = NumericalCompileRecord(
            backend="torchscript", target=actual_target,
            compiler_version=str(torch.__version__),
            exported_program_sha256=source_sha, graph_sha256=graph_sha,
            artifact_sha256=artifact_sha, artifact_size_bytes=artifact_size,
            reference_policy=reference_policy,
            requested_compile_policy=reference_policy,
            observed_compile_policy=policy_from_context(after),
            configuration_json=canonical_json({
                "backend_variant": VARIANT,
                "jit_optimization": False,
                "archive_device_policy": "preserve",
                "implementation_sha256": _digest(Path(__file__)),
                "region_validation": audit,
            }),
        )
        result = {
            "schema": "vlaforge.torchscript_numerical_compile/1",
            "numerical_compile_record": record.to_dict(),
            "observed_context_before": before.to_dict(),
            "observed_context_after": after.to_dict(),
            "region_validation": audit,
            "full_model_verified": False,
            "numerical_provider_enforcement": False,
        }
        os.link(staged, output)
    return result
