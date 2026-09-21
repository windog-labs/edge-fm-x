"""Frozen, model-independent tensor Session benchmark contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from vlaforge.ir.program import Module

SCHEMA = "vlaforge.session_latency_protocol/1"
MULTI_SCHEMA = "vlaforge.session_latency_protocol/2"
POLICIES = ("off", "batch-only", "required")
BOUNDARY = "model-tensor: resident CUDA inputs through Session run and completion"
HOST_BOUNDARY = "host-model-tensor: H2D, input binding, Session run, completion and complete D2H"
HOST_TIMING_SEGMENTS = ("h2d_ns", "bind_ns", "model_ns", "d2h_ns")
DTYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "f64": 8,
               "i64": 8, "i32": 4, "u64": 8, "u8": 1, "bool": 1}
OUTPUT_DTYPES = ("f32", "f16", "bf16")
MULTI_FLOAT_DTYPES = (*OUTPUT_DTYPES, "f64")
EXACT_DTYPES = ("i64", "i32", "u64", "u8", "bool")
MULTI_PROTOCOL_KEYS = {
    "schema", "boundary", "warmup", "measured", "processes", "policies", "bundles",
    "quality_gate", "eager_validation", "samples", "evidence", "gpu_ordinal",
    "monitor_gpu", "cuda_visible_devices", "cuda_arch", "cmake_prefix_path",
    "bundle_metadata_mode", "paper_gates", "outputs", "full_paper_acceptance",
    "boundary_includes", "boundary_excludes", "separate_wall_boundary", "model",
    "quality_reason", "checkpoint_sha256", "sample_order", "policy_order",
    "telemetry_interval_seconds", "clock_power_policy", "other_compute_owner_policy",
    "owner_identity_mode", "allocator_observation", "reference_identity",
    "reference_limitations", "numerical_worker_bootstrap", "aoti_package_extraction_root",
}


def includes_host_io(protocol):
    boundary = protocol.get("boundary")
    if boundary not in (BOUNDARY, HOST_BOUNDARY):
        raise ValueError("unsupported Session latency boundary")
    return boundary == HOST_BOUNDARY


def validate_host_timing(timings, rows):
    """Check complete per-call transfer evidence against the main timing CSV."""
    if not timings or len(timings) != len(rows):
        raise ValueError("host timing missing or incomplete")
    keys = {"run", "host_call_ns", *HOST_TIMING_SEGMENTS}
    for index, (timing, row) in enumerate(zip(timings, rows, strict=True)):
        if set(timing) != keys:
            raise ValueError("host timing columns differ from the explicit schema")
        if any(not isinstance(value, (int, str)) or isinstance(value, bool)
               or not str(value).isdecimal() for value in timing.values()):
            raise ValueError("host timing must contain nonnegative integer nanoseconds")
        values = {key: int(value) for key, value in timing.items()}
        if values["run"] != index or int(row["run"]) != index:
            raise ValueError("host timing call order differs")
        if (values["model_ns"] <= 0 or values["host_call_ns"] <= 0
                or sum(values[key] for key in HOST_TIMING_SEGMENTS) != values["host_call_ns"]):
            raise ValueError("host timing segments do not conserve complete call time")
        if values["host_call_ns"] != int(row["latency_ns"]):
            raise ValueError("latency CSV does not match the host timing boundary")


def numerical_worker_bootstrap(protocol):
    """Require an explicit process/thread ownership declaration before setters."""
    if "numerical_worker_bootstrap" not in protocol:
        return None
    value = protocol["numerical_worker_bootstrap"]
    if (not isinstance(value, dict)
            or set(value) != {"mode", "acknowledge_exclusive_process", "acknowledge_calling_thread"}
            or value["mode"] != "libtorch-explicit-exclusive-process/1"
            or value["acknowledge_exclusive_process"] is not True
            or value["acknowledge_calling_thread"] is not True):
        raise ValueError("numerical worker bootstrap requires explicit mode and process/thread ownership")
    return dict(value)


def tensor_bytes(dtype: str, shape) -> int:
    if dtype not in DTYPE_BYTES or any(type(dim) is not int or dim < 1 for dim in shape):
        raise ValueError("unsupported dtype or nonpositive static Tensor shape")
    return math.prod(shape) * DTYPE_BYTES[dtype]


def decode_tensor(raw: bytes, dtype: str, shape):
    """Decode explicit little-endian storage; no numeric dtype guessing."""
    import numpy as np

    if len(raw) != tensor_bytes(dtype, shape):
        raise ValueError("tensor byte count differs from dtype and complete shape")
    formats = {"f32": "<f4", "f16": "<f2", "bf16": "<u2", "f64": "<f8",
               "i64": "<i8", "i32": "<i4", "u64": "<u8", "u8": "u1", "bool": "u1"}
    value = np.frombuffer(raw, dtype=formats[dtype])
    if dtype == "bf16":
        value = (value.astype("<u4") << 16).view("<f4")
    if not np.isfinite(value).all():
        raise ValueError("tensor must be finite")
    if dtype == "bool" and not np.isin(value, [0, 1]).all():
        raise ValueError("bool storage must contain only 0 or 1")
    return value.reshape(shape)


def encode_reference(value, dtype: str, shape) -> bytes:
    """Accept FP32 representations of BF16 only after a bit-preserving roundtrip."""
    import numpy as np

    tensor_bytes(dtype, shape)
    if dtype not in OUTPUT_DTYPES or value.shape != tuple(shape) or not np.isfinite(value).all():
        raise ValueError("reference must be a finite complete supported floating output")
    if dtype == "bf16":
        if value.dtype != np.dtype("float32"):
            raise ValueError("BF16 reference requires explicit FP32 lossless representation")
        storage = np.ascontiguousarray(value, dtype="<f4").view("<u4")
        if np.any(storage & 0xFFFF):
            raise ValueError("FP32 reference is not losslessly representable as BF16")
        raw = (storage >> 16).astype("<u2").tobytes()
        if decode_tensor(raw, dtype, shape).tobytes() != value.tobytes():
            raise ValueError("BF16 reference roundtrip changed storage bits")
        return raw
    expected = np.dtype("float32" if dtype == "f32" else "float16")
    if value.dtype != expected:
        raise ValueError("reference dtype differs from declared output dtype")
    return np.ascontiguousarray(value, dtype="<f4" if dtype == "f32" else "<f2").tobytes()


def load_reference_array(path, output_name: str, dtype: str, shape) -> Any:
    """Load an ndarray reference from .npy or an unambiguous complete-output .npz."""
    import numpy as np

    packed = np.load(path, allow_pickle=False)
    if isinstance(packed, np.ndarray):
        return packed
    if not hasattr(packed, "files"):
        raise ValueError("output reference must be a .npy array or .npz archive")
    formats = {"f32": "float32", "f16": "float16", "bf16": "float32", "f64": "float64",
               "i64": "int64", "i32": "int32", "u64": "uint64", "u8": "uint8", "bool": "bool"}
    expected = np.dtype(formats[dtype])
    with packed:
        keys = list(packed.files)
        matches = []
        for key in packed.files:
            candidate = packed[key]
            if (candidate.shape == tuple(shape) and candidate.dtype == expected
                    and np.isfinite(candidate).all()):
                matches.append((key, np.asarray(candidate)))
    if len(matches) != 1:
        raise ValueError(
            f"npz reference for {output_name} must contain exactly one complete "
            f"{dtype} {tuple(shape)} array; found {sorted(keys)}"
        )
    return matches[0][1]


def active_indices(shape, dimensions=None) -> list[int]:
    """Flatten an explicit last-axis selection without discarding storage evidence."""
    tensor_bytes("f32", shape)
    if dimensions is None:
        return list(range(math.prod(shape)))
    if (not shape or not isinstance(dimensions, (list, tuple)) or not dimensions
            or any(type(index) is not int or not 0 <= index < shape[-1] for index in dimensions)
            or len(set(dimensions)) != len(dimensions)):
        raise ValueError("active dimensions must be unique valid nonempty last-axis indices")
    return [row * shape[-1] + dim for row in range(math.prod(shape[:-1])) for dim in dimensions]


def numeric_metrics(expected, actual) -> dict[str, Any]:
    import numpy as np

    left, right = np.asarray(expected, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if left.shape != right.shape or not left.size or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("metrics require matching nonempty finite tensors")
    left, right = left.reshape(-1), right.reshape(-1)
    delta = right - left
    mse, maximum = float(np.mean(delta * delta)), float(np.max(np.abs(delta)))
    norms = np.linalg.norm(left), np.linalg.norm(right)
    cosine = float(np.clip(np.dot(left, right) / (norms[0] * norms[1]), -1, 1)) if min(norms) > 1e-12 else None
    if not all(math.isfinite(x) for x in (mse, maximum, *norms)) or (cosine is not None and not math.isfinite(cosine)):
        raise ValueError("metric overflow")
    return {"count": left.size, "mse": mse, "rmse": math.sqrt(mse), "max_abs": maximum,
            "cosine": cosine, "reference_norm": float(norms[0]), "candidate_norm": float(norms[1])}


def output_spec(module: Module, dimensions=None) -> dict[str, Any]:
    if len(module.outputs) != 1 or module.outputs[0].payload.dtype not in OUTPUT_DTYPES:
        raise ValueError("benchmark requires one complete FP32, FP16 or BF16 Tensor output")
    port = module.outputs[0]
    dtype, shape = port.payload.dtype, port.payload.shape
    return {"dtype": dtype, "shape": list(shape), "size_bytes": tensor_bytes(dtype, shape),
            "count": math.prod(shape), "byte_order": "little",
            "raw_file": "outputs." + dtype, "active_indices": active_indices(shape, dimensions),
            "primary_metric_space": "active-dimensions" if dimensions is not None else "complete-output",
            "active_dimensions": dimensions}


def multi_output_declarations(protocol):
    declarations = protocol.get("outputs")
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("v2 requires a nonempty complete outputs declaration")
    names, primaries = [], 0
    for item in declarations:
        if (not isinstance(item, dict) or set(item) - {"name", "role", "active_dimensions"}
                or not isinstance(item.get("name"), str) or not item["name"]
                or item.get("role") not in ("primary-action", "float", "exact")):
            raise ValueError("invalid or unknown output declaration fields")
        names.append(item["name"])
        primaries += item["role"] == "primary-action"
        if "active_dimensions" in item and item["role"] != "primary-action":
            raise ValueError("only primary action permits active dimensions")
    if len(set(names)) != len(names) or primaries != 1:
        raise ValueError("outputs require unique names and exactly one primary action")
    return declarations


def benchmark_output_contract(module, protocol):
    """Keep legacy storage unchanged; v2 binds every ABI output explicitly."""
    if protocol["schema"] == SCHEMA:
        return output_spec(module, protocol.get("active_dimensions"))
    declarations = multi_output_declarations(protocol)
    if [item["name"] for item in declarations] != [port.name for port in module.outputs]:
        raise ValueError("outputs must cover all Module outputs in ABI order")
    result, device = [], None
    for index, (port, item) in enumerate(zip(module.outputs, declarations, strict=True)):
        dtype, shape = port.payload.dtype, port.payload.shape
        floating = item["role"] != "exact"
        if dtype not in (MULTI_FLOAT_DTYPES if floating else EXACT_DTYPES):
            raise ValueError("output role differs from its actual Tensor dtype")
        if (not isinstance(port.device, str) or not port.device.startswith("cuda:")
                or not port.device[5:].isdigit()
                or (device is not None and port.device != device)):
            raise ValueError("outputs require one explicit CUDA device")
        device = port.device
        primary = item["role"] == "primary-action"
        dimensions = item.get("active_dimensions")
        result.append({"name": port.name, "index": index, "role": item["role"],
                       "dtype": dtype, "shape": list(shape), "device": device,
                       "size_bytes": tensor_bytes(dtype, shape), "count": math.prod(shape),
                       "byte_order": "little",
                       "raw_file": "outputs." + dtype if primary else f"output-{index}.{dtype}",
                       "direct_file": "direct.bin" if primary else f"direct-{index}.bin",
                       "eager_file": "eager.bin" if primary else f"eager-{index}.bin",
                       "active_indices": active_indices(shape, dimensions) if primary else None,
                       "active_dimensions": dimensions,
                       "primary_metric_space": "active-dimensions" if dimensions is not None else "complete-output"})
    primary = next(item for item in result if item["role"] == "primary-action")
    return dict(primary, schema="vlaforge.session_output_contract/2", outputs=result,
                primary_output=primary["name"])


def encode_output_reference(value, spec):
    """Integer evidence never travels through float, including values above 2**53."""
    import numpy as np

    dtype, shape = spec["dtype"], spec["shape"]
    if dtype in OUTPUT_DTYPES:
        return encode_reference(value, dtype, shape)
    formats = {"f64": "<f8", "i32": "<i4", "i64": "<i8", "u64": "<u8", "u8": "u1", "bool": "?"}
    if (dtype not in formats or value.shape != tuple(shape)
            or value.dtype != np.dtype(formats[dtype])):
        raise ValueError("reference dtype and complete shape must match output contract")
    raw = np.ascontiguousarray(value, dtype=formats[dtype]).tobytes()
    decode_tensor(raw, dtype, shape)
    return raw


def compare_output_bytes(actual, direct, eager, spec):
    """Complete-byte gate plus per-output metrics, never cross-output reduction."""
    value = decode_tensor(actual, spec["dtype"], spec["shape"])
    decode_tensor(direct, spec["dtype"], spec["shape"])
    reference = decode_tensor(eager, spec["dtype"], spec["shape"])
    if actual != direct:
        raise ValueError("complete same-artifact output bytes differ")
    result = {"same_artifact_bitwise_equal": True, "eager_bitwise_equal": actual == eager}
    if spec["role"] == "exact":
        if actual != eager:
            raise ValueError("exact output differs from eager reference")
        result["comparison"] = "exact-bytes-only"
    else:
        result["comparison"] = "finite-floating-output"
        result["complete_storage_metrics"] = numeric_metrics(reference, value)
        if spec["role"] == "primary-action":
            active = spec["active_indices"]
            result["primary_metrics"] = numeric_metrics(reference.reshape(-1)[active], value.reshape(-1)[active])
    return result


def protocol_policies(value: Mapping[str, Any]) -> tuple[str, ...]:
    policies = value.get("policies", POLICIES)
    if (not isinstance(policies, (list, tuple)) or not policies
            or any(item not in POLICIES for item in policies)
            or len(policies) != len(set(policies))):
        raise ValueError("policies must be a nonempty unique supported subset")
    return tuple(policies)


def validate_protocol(value: Mapping[str, Any]) -> None:
    if value.get("schema") not in (SCHEMA, MULTI_SCHEMA):
        raise ValueError("unsupported Session latency protocol or boundary")
    includes_host_io(value)
    numerical_worker_bootstrap(value)
    if "aoti_package_extraction_root" in value:
        root = value["aoti_package_extraction_root"]
        if not isinstance(root, str) or not root:
            raise ValueError("AOTI extraction root must be a canonical absolute literal path")
        path = PurePosixPath(root)
        if (not path.is_absolute() or str(path) != root or ".." in path.parts
                or any(ord(ch) < 32 or ord(ch) == 127 or ch in '\\";$' for ch in root)):
            raise ValueError("AOTI extraction root must be a canonical absolute literal path")
    if value["schema"] == MULTI_SCHEMA:
        if set(value) - MULTI_PROTOCOL_KEYS:
            raise ValueError("unknown v2 protocol fields")
        declarations = multi_output_declarations(value)
        names = {item["name"] for item in declarations}
        if not isinstance(value.get("samples"), list):
            raise ValueError("v2 samples must be an explicit list")
        gates = value.get("paper_gates")
        if (not isinstance(gates, dict) or set(gates) != {"mse_max", "cosine_min"}
                or any(type(item) not in (int, float) or not math.isfinite(item) for item in gates.values())
                or gates["mse_max"] < 0 or not -1 <= gates["cosine_min"] <= 1):
            raise ValueError("v2 paper gates require explicit finite supported bounds")
        for sample in value.get("samples", []):
            if (not isinstance(sample, dict) or set(sample) - {"sample_id", "inputs", "outputs"}
                    or not isinstance(sample.get("inputs"), dict)
                    or not isinstance(sample.get("outputs"), dict)
                    or set(sample["outputs"]) != names):
                raise ValueError("v2 sample must contain every output reference")
            for reference in sample["outputs"].values():
                if (not isinstance(reference, dict) or set(reference) != {"direct", "eager"}
                        or any(not isinstance(path, str) or not path for path in reference.values())):
                    raise ValueError("output reference requires exact direct/eager paths")
    elif "outputs" in value:
        raise ValueError("multi-output declarations require explicit protocol v2")
    for name in ("warmup", "measured", "processes"):
        if type(value.get(name)) is not int or value[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if value["warmup"] != 128 or value["measured"] < 1024 or value["processes"] != 5:
        raise ValueError(
            "formal protocol requires 128 warmup, >=1024 measured and five processes"
        )
    if set(value.get("bundles", {})) != set(protocol_policies(value)):
        raise ValueError("protocol bundles must match the selected policies")
    if value.get("quality_gate") not in ("passed", "failed"):
        raise ValueError("quality gate must be explicitly recorded")
    if value.get("eager_validation", "paper-gates") not in ("paper-gates", "bitwise"):
        raise ValueError("unsupported eager validation gate")
    if value.get("owner_identity_mode", "process-pid") not in ("process-pid", "cuda-registration-handshake"):
        raise ValueError("unsupported GPU owner identity mode")
    if value.get("allocator_observation", "off") not in ("off", "libtorch-native/1"):
        raise ValueError("unsupported allocator observation mode")
    if not value.get("samples") or not value.get("evidence"):
        raise ValueError("protocol requires paired samples and provenance evidence")
    if "cuda_visible_devices" in value or "monitor_gpu" in value:
        identity = value.get("monitor_gpu")
        if (not isinstance(identity, str) or not identity.startswith("GPU-")
                or value.get("cuda_visible_devices") != identity or value.get("gpu_ordinal") != 0):
            raise ValueError("isolated CUDA mapping requires matching monitored GPU UUID and logical ordinal zero")
    if value["warmup"] % len(value["samples"]) or value["measured"] % len(
        value["samples"]
    ):
        raise ValueError("every process must use balanced complete sample cycles")


def validate_loop_policy(selected, plan, policy, mode="selection-manifest") -> None:
    loops = [task for task in plan["tasks"] if task["opcode"] == "vla.for"]
    if mode == "ordinary-source":
        if selected is not None or policy != "off":
            raise ValueError("ordinary-source metadata only supports unselected off bundles")
    elif mode != "selection-manifest" or selected is None or selected.get("requested") != policy:
        raise ValueError("bundle compile policy differs from benchmark label")
    if any(task["attributes"].get("replay", "off") != policy for task in loops):
        raise ValueError("actual scheduled loop policy differs from benchmark label")


def process_order(processes: int = 5, policies=POLICIES) -> list[tuple[int, str]]:
    policies = protocol_policies({"policies": policies})
    return [
        (repeat, policies[(repeat + offset) % len(policies)])
        for repeat in range(processes)
        for offset in range(len(policies))
    ]


def validate_fidelity(protocol, rows):
    if not rows:
        raise ValueError("missing complete-output fidelity evidence")
    if protocol["quality_gate"] == "passed" and not all(row["paper_numeric_gate"] for row in rows):
        raise ValueError("observed output fails declared paper numeric gate")
    if protocol.get("eager_validation") == "bitwise" and not all(row["eager_bitwise_equal"] for row in rows):
        raise ValueError("observed output fails eager bitwise gate")


def input_specs(module: Module, *, contract=None) -> tuple[str, int, int]:
    if contract is None:
        output_spec(module)
    output = module.outputs[contract["index"] if contract and "outputs" in contract else 0]
    device = output.device
    if not device or not device.startswith("cuda:"):
        raise ValueError("benchmark requires explicit CUDA tensor ports")
    declarations = []
    for port in module.inputs:
        if port.device != device or port.payload.dtype not in DTYPE_BYTES:
            raise ValueError(
                "benchmark requires one CUDA device and supported Tensor inputs"
            )
        shape = port.payload.shape
        if any(type(dim) is not int or dim < 1 for dim in shape):
            raise ValueError("benchmark requires positive static Tensor shapes")
        dtype = "VLAFORGE_DTYPE_" + port.payload.dtype.upper()
        dimensions = ",".join(map(str, shape))
        size = tensor_bytes(port.payload.dtype, shape)
        declarations.append(f"{{{dtype}, {{{dimensions}}}, {size}u}}")
    if any(type(dim) is not int or dim < 1 for dim in output.payload.shape):
        raise ValueError("benchmark requires static output shape")
    return (
        ",\n".join(declarations),
        int(device.removeprefix("cuda:")),
        math.prod(output.payload.shape),
    )


def replay_checks(rows: list[Mapping[str, Any]], *, per_call: bool = False) -> tuple[str, str]:
    checks, failures = [], []
    for row in rows:
        task, policy, steps = row["task_id"], row["policy"], row["steps"]
        if policy not in ("batch-only", "required"):
            raise ValueError(
                "formal benchmark requires explicit off/batch-only/required"
            )
        state = "READY" if policy == "required" else "UNPREPARED"
        captured = steps if policy == "required" else 0
        replays = "run + 1u" if policy == "required" else "0u"
        ordinary = "run + 1u" if policy == "batch-only" else "0u"
        telemetry = f'''      std::printf("REPLAY,%zu,{task},{policy},%u,%u,%llu,%llu\\n", run,
          static_cast<unsigned>(info.state), info.captured_steps,
          static_cast<unsigned long long>(info.replay_count),
          static_cast<unsigned long long>(info.ordinary_count));
''' if per_call else ""
        checks.append(f"""
    {{
      VLAForgeBoundedReplayInfo info{{}}; info.struct_size = sizeof(info);
      if (vlaforge_model_session_get_replay_info(session, {task}u, &info).code != VLAFORGE_STATUS_OK ||
          info.state != VLAFORGE_REPLAY_{state} || info.captured_steps != {captured}u ||
          info.replay_count != {replays} || info.ordinary_count != {ordinary}) return 17;
{telemetry}      if (run + 1u == warmup + measured) std::fprintf(stderr, "REPLAY_FINAL,{task},%u,%u,%llu,%llu\\n",
          static_cast<unsigned>(info.state), info.captured_steps,
          static_cast<unsigned long long>(info.replay_count), static_cast<unsigned long long>(info.ordinary_count));
    }}
""")
        failures.append(f"""
      {{
        VLAForgeBoundedReplayInfo info{{}}; info.struct_size = sizeof(info);
        if (vlaforge_model_session_get_replay_info(session, {task}u, &info).code == VLAFORGE_STATUS_OK &&
            info.state == VLAFORGE_REPLAY_POISONED) {{
          std::fprintf(stderr, "fatal replay/context: %s\\n", info.reason);
          api->destroy(session); std::fflush(stderr); std::_Exit(10);
        }}
      }}
""")
    return "".join(checks), "".join(failures)


def validate_rows(
    rows: list[Mapping[str, str]], *, warmup: int, measured: int, samples: int
) -> None:
    if len(rows) != warmup + measured:
        raise ValueError("missing or duplicated benchmark calls")
    for index, row in enumerate(rows):
        if int(row["run"]) != index or int(row["sample"]) != index % samples:
            raise ValueError("benchmark call/sample order changed")
        if (
            int(row["measured"]) != int(index >= warmup)
            or int(row["revision"]) != index + 1
        ):
            raise ValueError("warmup boundary or fresh input revision changed")
        if (
            int(row["latency_ns"]) <= 0
            or row["finite"] != "1"
            or row["direct_exact"] != "1"
        ):
            raise ValueError("latency or same-artifact fidelity failure")
        for prefix in ("direct", "eager"):
            for name in ("mse", "max_abs"):
                if not math.isfinite(float(row[f"{prefix}_{name}"])):
                    raise ValueError("nonfinite benchmark metric")
            if row[f"{prefix}_cosine_defined"] not in ("0", "1"):
                raise ValueError("cosine status missing")
            if row[f"{prefix}_cosine_defined"] == "1" and not math.isfinite(
                float(row[f"{prefix}_cosine"])
            ):
                raise ValueError("nonfinite defined cosine")
        if float(row["direct_mse"]) or float(row["direct_max_abs"]):
            raise ValueError("same-artifact output changed")
