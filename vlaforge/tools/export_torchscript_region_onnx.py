#!/usr/bin/env python3
"""Export a validated TorchScript tensor region to ONNX with parity evidence.

The tool is model-agnostic. It accepts a TorchScript archive and a tuple of
runtime inputs, exports a fixed-shape ONNX graph, checks the ONNX model, and
records TorchScript/ONNX Runtime parity when the installed runtime supports
the emitted operators.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensor_metadata(value) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def external_files(model, root: Path) -> list[Path]:
    """Find external tensors, including those nested in graph attributes."""
    import onnx

    locations: set[Path] = set()

    def visit(message):
        if isinstance(message, onnx.TensorProto):
            if message.data_location == onnx.TensorProto.EXTERNAL:
                entries = {entry.key: entry.value for entry in message.external_data}
                location = Path(entries.get("location", ""))
                if (not entries.get("location") or location.is_absolute()
                        or ".." in location.parts):
                    raise ValueError("external tensor location must stay inside export directory")
                source = root / location
                if not source.resolve(strict=True).is_relative_to(root.resolve()):
                    raise ValueError("external tensor location escapes export directory")
                if any(path.is_symlink() for path in [source, *source.parents]
                       if path != root and root in path.parents):
                    raise ValueError("external tensor location must not use symlinks")
                if not source.is_file():
                    raise ValueError("external tensor location must be a regular file")
                locations.add(location)
            return
        for field, value in message.ListFields():
            if field.message_type is not None:
                if field.is_repeated:
                    for item in value:
                        visit(item)
                else:
                    visit(value)

    visit(model)
    return sorted(locations)


def publish_files(staged: Path, output: Path, sidecars: list[Path]) -> list[Path]:
    """Publish weights before the graph, without overwriting existing files."""
    published: list[Path] = []
    created_directories: list[Path] = []
    try:
        for relative in [*sidecars, Path(staged.name)]:
            destination = output if relative == Path(staged.name) else output.parent / relative
            if destination == output and relative in sidecars:
                raise ValueError("external tensor location aliases the ONNX graph")
            missing = []
            parent = destination.parent
            while parent != output.parent and not parent.exists():
                missing.append(parent)
                parent = parent.parent
            if not parent.resolve().is_relative_to(output.parent.resolve()):
                raise ValueError("external tensor destination escapes output directory")
            for directory in reversed(missing):
                directory.mkdir()
                created_directories.append(directory)
            os.link(staged.parent / relative, destination)
            published.append(destination)
        return published
    except Exception:
        for path in reversed(published):
            path.unlink()
        for directory in reversed(created_directories):
            directory.rmdir()
        raise


def runtime_feeds(names, original_names, examples):
    supplied = dict(zip(original_names, examples, strict=True))
    if len(set(names)) != len(names) or set(names) - supplied.keys():
        raise ValueError("ONNX input names differ from the named TorchScript contract")
    return {name: supplied[name].detach().cpu().numpy() for name in names}


def relocate_device_constants(module, device):
    """Explicitly relocate traced factory devices, not dtype or scalar constants."""
    import torch

    target = torch.device(device)
    changed = []

    def visit(block, path, method):
        for node in block.nodes():
            if node.kind() == "prim::Constant" and node.outputsSize() == 1:
                value = node.output().toIValue()
                if isinstance(value, torch.device) and value != target:
                    node.s_("value", str(target))
                    changed.append({"module": path, "method": method,
                                    "from": str(value), "to": str(target)})
            for child in node.blocks():
                visit(child, path, method)

    for path, child in module.named_modules():
        for method in child._c._method_names():
            visit(child._c._get_method(method).graph, path, method)
    return changed


@contextmanager
def composite_opmath(opset):
    """Retain FLOAT opmath for native 16-bit activations and variance reductions."""
    import onnx
    import torch
    from torch.onnx import symbolic_helper, symbolic_opset9  # noqa: F401
    try:
        from torch.onnx._internal.torchscript_exporter import registration
    except ImportError:
        from torch.onnx._internal import registration

    originals = {}
    for name in ("aten::gelu", "aten::silu", "aten::var", "aten::var_mean", "aten::std"):
        group = registration.registry.get_function_group(name)
        if group is None or group.get(opset) is None:
            raise RuntimeError("native composite symbolic is unavailable: " + name)
        if any(version <= opset and group._functions.overridden(version) for version in group._functions):
            raise ValueError("existing custom composite symbolic must not be overwritten: " + name)
        originals[name] = group.get(opset)
    record = {"policy": "native_16bit_composites_float_opmath/1", "promoted": {},
              "registration_restored": False, "counts_are_static_export_callbacks": True}

    def wrap(name, original):
        def symbolic(graph, value, *args):
            scalar = symbolic_helper._try_get_scalar_type(value)
            if scalar is None:
                raise ValueError("composite input dtype must be explicit: " + name)
            dtype = int(scalar.onnx_type())
            if dtype not in (onnx.TensorProto.BFLOAT16, onnx.TensorProto.FLOAT16):
                return original(graph, value, *args)
            key = name + ":" + onnx.TensorProto.DataType.Name(dtype)
            record["promoted"][key] = record["promoted"].get(key, 0) + 1
            promoted = graph.op("Cast", value, to_i=onnx.TensorProto.FLOAT)
            result = original(graph, promoted, *args)
            if isinstance(result, (tuple, list)):
                return tuple(graph.op("Cast", item, to_i=dtype) for item in result)
            return graph.op("Cast", result, to_i=dtype)
        return symbolic

    registered = []
    try:
        for name, original in originals.items():
            torch.onnx.register_custom_op_symbolic(name, wrap(name, original), opset)
            registered.append(name)
        yield record
    finally:
        for name in reversed(registered):
            torch.onnx.unregister_custom_op_symbolic(name, opset)
        record["registration_restored"] = True


def tensor_outputs(value):
    import torch

    values = (value,) if isinstance(value, torch.Tensor) else tuple(value)
    if not values or any(not isinstance(item, torch.Tensor) for item in values):
        raise ValueError("TorchScript region must return a flat tuple of tensors")
    return values


@contextmanager
def static_grouped_query_attention(opset, observed_sites=None):
    """Lower provable GQA heads and retain native 16-bit SDPA FLOAT opmath."""
    import onnx
    import torch
    from torch.onnx import symbolic_helper, symbolic_opset14  # noqa: F401
    try:
        from torch.onnx._internal.torchscript_exporter import registration
    except ImportError:
        from torch.onnx._internal import registration

    name = "aten::scaled_dot_product_attention"
    group = registration.registry.get_function_group(name)
    if group is None or group.get(opset) is None:
        raise RuntimeError("standard SDPA symbolic is unavailable")
    if any(version <= opset and group._functions.overridden(version) for version in group._functions):
        raise ValueError("existing custom SDPA symbolic must not be overwritten")
    original = group.get(opset)
    record = {"policy": "static_GQA_repeat_key_value_heads/1", "rewritten": [],
              "observed_specializations": [], "float_opmath_sites": [],
              "safe_softmax_sites": [], "registration_restored": False}
    used_sites = set()

    class SafeSoftmaxGraph:
        def __init__(self, graph, compute_dtype):
            self.graph = graph
            self.compute_dtype = compute_dtype

        def __getattr__(self, name):
            return getattr(self.graph, name)

        def op(self, name, *operands, **attributes):
            graph = self.graph
            if name != "Softmax":
                return graph.op(name, *operands, **attributes)
            assert len(operands) == 1
            logits = operands[0]
            axis = attributes.get("axis_i", -1)
            dtype = torch.float64 if self.compute_dtype == onnx.TensorProto.DOUBLE else torch.float32
            zero = graph.op("Constant", value_t=torch.tensor(0.0, dtype=dtype))
            negative_inf = graph.op("Constant", value_t=torch.tensor(float("-inf"), dtype=dtype))
            valid = graph.op("Not", graph.op("Equal", logits, negative_inf))
            count = graph.op("ReduceSum", graph.op("Cast", valid, to_i=onnx.TensorProto.FLOAT),
                             graph.op("Constant", value_t=torch.tensor([axis], dtype=torch.int64)), keepdims_i=1)
            any_valid = graph.op("Greater", count,
                                 graph.op("Constant", value_t=torch.tensor(0.0, dtype=torch.float32)))
            # Native SDPA returns zero weights for an all-negative-infinity row.
            safe = graph.op("Where", any_valid, logits, zero)
            weights = graph.op("Softmax", safe, **attributes)
            record["safe_softmax_sites"].append({"axis": axis, "compute_onnx_dtype": self.compute_dtype})
            return graph.op("Where", any_valid, weights, zero)

    def repeat_heads(graph, value, rank, heads, repeats):
        def constant(values):
            return graph.op("Constant", value_t=torch.tensor(values, dtype=torch.int64))

        shape = graph.op("Shape", value)
        prefix = graph.op("Slice", shape, constant([0]), constant([rank - 2]))
        tail = graph.op("Slice", shape, constant([rank - 2]), constant([rank]))
        expanded = graph.op("Expand", graph.op("Unsqueeze", value, constant([rank - 2])),
                            graph.op("Concat", prefix, constant([repeats]), tail, axis_i=0))
        batch = graph.op("Slice", shape, constant([0]), constant([rank - 3]))
        result = graph.op("Reshape", expanded,
                          graph.op("Concat", batch, constant([heads * repeats]), tail, axis_i=0))
        sizes = symbolic_helper._get_tensor_sizes(value)
        if sizes is not None:
            result.setType(value.type().with_sizes([*sizes[:rank - 3], heads * repeats, *sizes[rank - 2:]]))
        return result

    @symbolic_helper.parse_args("v", "v", "v", "v", "f", "b", "v", "b")
    def symbolic(graph, query, key, value, mask=None, dropout=0.0, causal=False, scale=None, enable_gqa=False):
        if observed_sites is not None:
            site = str(graph.original_node.sourceRange())
            if site not in observed_sites:
                raise ValueError("observed attention profile has no matching source site")
            shapes = observed_sites[site]
            for operand, expected in zip((query, key, value), shapes, strict=True):
                known = symbolic_helper._get_tensor_sizes(operand)
                if known is not None and (len(known) != len(expected)
                        or any(dim is not None and dim != wanted for dim, wanted in zip(known, expected, strict=True))):
                    raise ValueError("observed attention shape conflicts with a known graph dimension")
                operand.setType(operand.type().with_sizes(expected))
            used_sites.add(site)
            record["observed_specializations"].append({"site_sha256": hashlib.sha256(site.encode()).hexdigest(), "shapes": shapes})
        if enable_gqa:
            ranks = [symbolic_helper._get_tensor_rank(item) for item in (query, key, value)]
            heads = [symbolic_helper._get_tensor_dim_size(item, -3) for item in (query, key, value)]
            if (any(rank is None or rank < 3 for rank in ranks) or len(set(ranks)) != 1
                    or any(head is None or head <= 0 for head in heads)
                    or heads[1] != heads[2] or heads[0] % heads[1]):
                raise ValueError("GQA requires equal known ranks and positive divisible static Q/K/V head counts")
            repeats = heads[0] // heads[1]
            if repeats != 1:
                key = repeat_heads(graph, key, ranks[1], heads[1], repeats)
                value = repeat_heads(graph, value, ranks[2], heads[2], repeats)
            record["rewritten"].append({"rank": ranks[0], "query_heads": heads[0], "key_value_heads": heads[1],
                                         "repeat_interleave": repeats})
        scalar = symbolic_helper._try_get_scalar_type(query)
        if scalar is None:
            raise ValueError("SDPA input dtype must be explicit")
        dtype = int(scalar.onnx_type())
        if dtype not in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16):
            raise ValueError("SDPA requires a real floating input dtype")
        promoted = dtype in (onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16)
        compute_dtype = onnx.TensorProto.FLOAT if promoted else dtype
        if promoted:
            query, key, value = (graph.op("Cast", operand, to_i=compute_dtype) for operand in (query, key, value))
            if not symbolic_helper._is_none(mask):
                mask_type = symbolic_helper._try_get_scalar_type(mask)
                if mask_type is not None and int(mask_type.onnx_type()) in (onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16):
                    mask = graph.op("Cast", mask, to_i=compute_dtype)
            record["float_opmath_sites"].append({"input_onnx_dtype": dtype, "output_onnx_dtype": dtype})
        # Explicit ONNX dtype avoids the legacy scalar pass confusing BF16's JIT enum with COMPLEX128.
        if not symbolic_helper._is_none(scale):
            scale = graph.op("Cast", scale, to_i=compute_dtype)
        result = original(SafeSoftmaxGraph(graph, compute_dtype), query, key, value,
                          mask, dropout, causal, scale, False)
        return graph.op("Cast", result, to_i=dtype) if promoted else result

    torch.onnx.register_custom_op_symbolic(name, symbolic, opset)
    try:
        yield record
        if observed_sites is not None and used_sites != set(observed_sites):
            raise ValueError("observed attention profile has unused source sites")
    finally:
        torch.onnx.unregister_custom_op_symbolic(name, opset)
        record["registration_restored"] = True


def load_attention_shape_profile(path, archive, examples):
    """Bind observed specialization to the exact archive and export inputs."""
    profile = json.loads(path.read_text())
    if (profile.get("schema") != "edgefm.frozen_JIT_observed_attention_shapes/1"
            or not profile.get("all_complete_CPU_outputs_byte_exact")):
        raise ValueError("attention profile must retain complete-output observer validation")
    for name, file in (("archive", archive), ("inputs", examples)):
        expected = profile[name]
        if expected["sha256"] != digest(file) or expected["bytes"] != file.stat().st_size:
            raise ValueError("attention profile does not match export " + name)
    sites = {}
    for row in profile["sites"]:
        shapes = row["shapes"]
        if (len(shapes) != 3 or len({len(shape) for shape in shapes}) != 1
                or any(len(shape) < 3 or any(type(dim) is not int or dim <= 0 for dim in shape) for shape in shapes)
                or shapes[1][-3] != shapes[2][-3] or shapes[0][-3] % shapes[1][-3]):
            raise ValueError("observed attention profile has an invalid head/shape contract")
        site = row["source_range"]
        if not isinstance(site, str) or not site or site in sites:
            raise ValueError("observed attention profile requires unique nonempty source sites")
        sites[site] = shapes
    if not sites:
        raise ValueError("observed attention profile is empty")
    return sites


def numpy_outputs(value) -> tuple:
    import numpy as np

    if isinstance(value, np.ndarray):
        return (value,)
    return tuple(value)


def tensor_metrics(reference, actual) -> dict[str, Any]:
    import numpy as np

    reference = np.asarray(reference)
    actual = np.asarray(actual)
    if reference.shape != actual.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
        }
    if reference.dtype == np.bool_ or actual.dtype == np.bool_:
        delta = np.logical_xor(reference, actual)
        return {
            "shape_match": True,
            "exact": bool(np.array_equal(reference, actual)),
            "mismatch_count": int(np.count_nonzero(delta)),
        }
    left = reference.astype(np.float64)
    right = actual.astype(np.float64)
    delta = right - left
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return {
        "shape_match": True,
        "exact": bool(np.array_equal(reference, actual)),
        "maximum_absolute_error": float(np.max(np.abs(delta))),
        "mean_squared_error": float(np.mean(delta * delta)),
        "root_mean_squared_error": float(np.sqrt(np.mean(delta * delta))),
        "cosine_similarity": (
            float(np.vdot(right, left) / denominator)
            if denominator
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchscript", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--relocate-device-constants", action="store_true",
        help="Explicitly rewrite traced factory device constants to --device; record every change.",
    )
    parser.add_argument(
        "--no-constant-folding",
        action="store_true",
        help="Disable the legacy exporter constant-folding pass.",
    )
    parser.add_argument(
        "--skip-runtime-validation",
        action="store_true",
        help="Keep ONNX checker evidence without claiming runtime parity.",
    )
    parser.add_argument("--lower-static-gqa", action="store_true",
                        help="Expand constant-head GQA and retain 16-bit SDPA FLOAT opmath before standard ONNX export.")
    parser.add_argument("--attention-shape-profile", type=Path,
                        help="Optional complete-output-validated observed shapes, bound to this exact archive/input tuple.")
    args = parser.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    torchscript = args.torchscript.resolve(strict=True)
    inputs_path = args.inputs.resolve(strict=True)
    output = args.output.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output.with_suffix(".report.json")
    )
    if output.exists():
        raise FileExistsError(output)
    if report_path.exists():
        raise FileExistsError(report_path)
    if args.opset < 17:
        raise ValueError("opset must be at least 17")
    if args.attention_shape_profile and not args.lower_static_gqa:
        raise ValueError("attention shape specialization requires --lower-static-gqa")
    observed_sites = (load_attention_shape_profile(args.attention_shape_profile, torchscript, inputs_path)
                      if args.attention_shape_profile else None)

    started_ns = time.time_ns()
    try:
        module = torch.jit.load(str(torchscript), map_location=args.device).to(args.device)
        relocations = (relocate_device_constants(module, args.device)
                       if args.relocate_device_constants else [])
        examples = torch.load(inputs_path, map_location=args.device, weights_only=True)
        if not isinstance(examples, tuple) or not examples:
            raise ValueError("runtime inputs must be a non-empty tuple")
        if any(not isinstance(value, torch.Tensor) for value in examples):
            raise ValueError("runtime inputs must contain only tensors")
        examples = tuple(value.to(args.device).contiguous() for value in examples)
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            reference = tensor_outputs(module(*examples))
    except Exception as error:
        write_json(report_path, {
            "schema": "vlaforge.torchscript_onnx_export/1", "status": "failed",
            "phase": "load-or-reference", "torchscript": str(torchscript),
            "torchscript_sha256": digest(torchscript), "inputs": str(inputs_path),
            "inputs_sha256": digest(inputs_path), "device": args.device,
            "device_constant_relocation_requested": args.relocate_device_constants,
            "error_type": type(error).__name__, "message": str(error),
            "elapsed_ns": time.time_ns() - started_ns,
        })
        raise

    input_names = [f"input_{index}" for index in range(len(examples))]
    output_names = [f"output_{index}" for index in range(len(reference))]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".torchscript-onnx-", dir=output.parent)
    )
    staged = temporary_root / output.name
    report: dict[str, Any] = {
        "schema": "vlaforge.torchscript_onnx_export/1",
        "status": "started",
        "torchscript": str(torchscript),
        "torchscript_sha256": digest(torchscript),
        "inputs": str(inputs_path),
        "inputs_sha256": digest(inputs_path),
        "output": str(output),
        "opset": args.opset,
        "dynamo": False,
        "device": args.device,
        "device_constant_relocation_requested": args.relocate_device_constants,
        "device_constant_relocations": relocations,
        "torch_version": str(torch.__version__),
        "onnx_version": str(onnx.__version__),
        "onnxruntime_version": str(ort.__version__),
        "input_names": input_names,
        "output_names": output_names,
        "input_contract": [tensor_metadata(value) for value in examples],
        "output_contract": [tensor_metadata(value) for value in reference],
        "started_ns": started_ns,
        "attention_shape_profile": ({"path": str(args.attention_shape_profile.resolve()),
                                      "sha256": digest(args.attention_shape_profile), "scope": "exact frozen input/profile only"}
                                     if args.attention_shape_profile else None),
    }
    published: list[Path] = []
    try:
        gqa_policy = static_grouped_query_attention(args.opset, observed_sites) if args.lower_static_gqa else nullcontext(None)
        with composite_opmath(args.opset) as opmath, gqa_policy as gqa, torch.inference_mode(), torch.jit.optimized_execution(False):
            report["composite_opmath"] = opmath
            report["static_gqa_lowering"] = gqa
            torch.onnx.export(
                module,
                examples,
                str(staged),
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=not args.no_constant_folding,
                input_names=input_names,
                output_names=output_names,
                training=torch.onnx.TrainingMode.EVAL,
                dynamo=False,
                external_data=False,
            )
        # The legacy exporter automatically externalizes archives above 2 GiB.
        # Keep those weights out of the protobuf and validate by filename.
        model = onnx.load(str(staged), load_external_data=False)
        sidecars = external_files(model, temporary_root)
        onnx.checker.check_model(str(staged))
        onnx_input_names = [value.name for value in model.graph.input]
        onnx_output_names = [value.name for value in model.graph.output]
        report["onnx_checker"] = "passed"
        report["onnx_input_names"] = onnx_input_names
        report["onnx_output_names"] = onnx_output_names
        report["onnx_nodes"] = len(model.graph.node)
        report["onnx_initializers"] = len(model.graph.initializer)
        report["onnx_op_types"] = sorted(
            {node.op_type for node in model.graph.node}
        )

        if args.skip_runtime_validation:
            report["runtime_parity"] = {
                "status": "skipped",
                "reason": "explicit --skip-runtime-validation",
            }
        else:
            session = ort.InferenceSession(
                str(staged),
                providers=["CPUExecutionProvider"],
            )
            feeds = runtime_feeds(onnx_input_names, input_names, examples)
            actual = numpy_outputs(session.run(None, feeds))
            expected = numpy_outputs(
                tuple(value.detach().cpu().numpy() for value in reference)
            )
            parity = [
                tensor_metrics(left, right)
                for left, right in zip(expected, actual, strict=True)
            ]
            report["runtime_parity"] = {
                "status": "passed",
                "outputs": parity,
                "all_shapes_match": all(item["shape_match"] for item in parity),
                "constant_folded_inputs": len(examples) - len(onnx_input_names),
            }

        published = publish_files(staged, output, sidecars)
        report.update(
            {
                "status": "passed",
                "output_sha256": digest(output),
                "output_size": output.stat().st_size,
                "external_data": [
                    {
                        "location": str(path),
                        "size": (output.parent / path).stat().st_size,
                        "sha256": digest(output.parent / path),
                    }
                    for path in sidecars
                ],
                "elapsed_ns": time.time_ns() - started_ns,
            }
        )
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
                "elapsed_ns": time.time_ns() - started_ns,
            }
        )
        write_json(report_path, report)
        for path in reversed(published):
            path.unlink()
        raise
    finally:
        for path in sorted(temporary_root.glob("**/*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        temporary_root.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
