"""Probe a generated native Session for contracts, commits and lifecycle.

This is a host-side validation harness, not deployment evidence. It exercises
the generated C ABI directly because the normal Python binding intentionally
validates shapes before the native boundary.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import sys


class Status(C.Structure):
    _fields_ = [
        ("code", C.c_int),
        ("message", C.c_void_p),
        ("message_size", C.c_size_t),
    ]


class Device(C.Structure):
    _fields_ = [("kind", C.c_int), ("ordinal", C.c_int32)]


class Tensor(C.Structure):
    _fields_ = [
        ("data", C.c_void_p),
        ("size_bytes", C.c_uint64),
        ("dimensions", C.POINTER(C.c_int64)),
        ("rank", C.c_uint32),
        ("dtype", C.c_int),
        ("device", Device),
    ]


class BoundTensor(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        ("tensor", Tensor),
        ("layout", C.c_int),
        ("alignment", C.c_uint64),
    ]


class Stamp(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        ("has_revision", C.c_uint8),
        ("has_timestamp", C.c_uint8),
        ("reserved", C.c_uint8 * 6),
        ("revision", C.c_uint64),
        ("timestamp_ns", C.c_uint64),
    ]


BindFn = C.CFUNCTYPE(
    Status, C.c_void_p, C.c_uint32, C.POINTER(BoundTensor), C.POINTER(Stamp)
)
RunFn = C.CFUNCTYPE(Status, C.c_void_p)
ReadFn = C.CFUNCTYPE(
    Status, C.c_void_p, C.c_uint32, C.POINTER(BoundTensor)
)
DestroyFn = C.CFUNCTYPE(None, C.c_void_p)
ResetFn = C.CFUNCTYPE(Status, C.c_void_p, C.c_uint64)


class Api(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        ("abi_version", C.c_uint32),
        ("schema_digest", C.c_void_p),
        ("schema_digest_size", C.c_size_t),
        ("bind_tensor", BindFn),
        ("bind_scalar", C.c_void_p),
        ("run", RunFn),
        ("read_output_tensor", ReadFn),
        ("read_output_scalar", C.c_void_p),
        ("reset_episode", ResetFn),
        ("destroy", DestroyFn),
    ]


DTYPE_CODES = {
    "bool": 1,
    "i32": 2,
    "i64": 3,
    "f16": 4,
    "bf16": 5,
    "f32": 6,
    "f64": 7,
    "u64": 8,
    "u8": 9,
}

TORCH_DTYPES = {
    "bool": "bool",
    "i32": "int32",
    "i64": "int64",
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
    "u64": "uint64",
    "u8": "uint8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(spec: dict) -> int:
    import torch

    dtype = getattr(torch, TORCH_DTYPES[spec["payload"]["dtype"]])
    count = 1
    for dimension in spec["payload"]["shape"]:
        count *= int(dimension)
    return count * torch.empty((), dtype=dtype).element_size()


def status_message(status: Status) -> str:
    if not status.message:
        return ""
    return C.string_at(status.message, min(int(status.message_size), 4096)).decode(
        "utf-8", errors="replace"
    )


def device_tuple(name: str) -> tuple[int, int]:
    if name == "cpu":
        return 0, 0
    match = re.fullmatch(r"cuda:([0-9]+)", name)
    if match is None:
        raise ValueError(f"unsupported tensor device: {name}")
    return 1, int(match.group(1))


def nonempty_status(status: Status) -> dict[str, object]:
    return {
        "rejected": bool(status.code),
        "code": int(status.code),
        "message": status_message(status),
    }


def load_inputs(
    sample_dir: Path,
    input_schema: dict,
    *,
    device: str,
) -> dict[int, object]:
    import torch

    result = {}
    for spec in input_schema["inputs"]:
        if spec["payload"]["kind"] != "tensor":
            raise ValueError("contract probe supports tensor inputs only")
        path = sample_dir / f"{spec['input_id']}.bin"
        payload = path.read_bytes()
        expected = tensor_bytes(spec)
        if len(payload) != expected:
            raise ValueError(
                f"input payload size differs for {spec['name']}: "
                f"{len(payload)} != {expected}"
            )
        dtype = getattr(torch, TORCH_DTYPES[spec["payload"]["dtype"]])
        values = torch.frombuffer(bytearray(payload), dtype=dtype)
        result[spec["input_id"]] = values.reshape(spec["payload"]["shape"]).to(device)
    return result


def sample_input_hashes(sample_dir: Path, input_schema: dict) -> dict[str, str]:
    return {
        spec["name"]: sha256(sample_dir / f"{spec['input_id']}.bin")
        for spec in input_schema["inputs"]
        if spec["payload"]["kind"] == "tensor"
    }


def create_session(library: C.CDLL, bundle: Path) -> C.c_void_p:
    create = library.vlaforge_model_session_create_from_bundle
    create.argtypes = [C.c_char_p, C.c_size_t, C.POINTER(C.c_void_p)]
    create.restype = Status
    session = C.c_void_p()
    encoded = os.fsencode(bundle)
    status = create(encoded, len(encoded), C.byref(session))
    if status.code or not session:
        raise RuntimeError(
            f"native Session creation failed: {status_message(status)}"
        )
    return session


def run_values(
    api: Api,
    session: C.c_void_p,
    input_schema: dict,
    output_schema: dict,
    values: dict[int, object],
    *,
    revision: int,
    device: str,
    keepalive: list[object],
) -> dict[str, str]:
    statuses = bind_inputs(
        api,
        session,
        input_schema,
        values,
        revision=revision,
        keepalive=keepalive,
    )
    if any(item.code for item in statuses):
        raise RuntimeError("valid lifecycle input binding was rejected")
    status = api.run(session)
    if status.code:
        raise RuntimeError(
            f"valid lifecycle run failed: {status_message(status)}"
        )
    hashes, errors = output_hashes(
        api, session, output_schema, device=device
    )
    if errors:
        raise RuntimeError(
            "lifecycle output read failed: " + "; ".join(errors)
        )
    return hashes


def probe_lifecycle(
    library: C.CDLL,
    api: Api,
    bundle: Path,
    input_schema: dict,
    output_schema: dict,
    values: dict[int, object],
    alternate: dict[int, object],
    *,
    sample_dir: Path,
    alternate_sample_dir: Path,
    device: str,
) -> dict[str, object]:
    primary_input_hashes = sample_input_hashes(sample_dir, input_schema)
    alternate_input_hashes = sample_input_hashes(
        alternate_sample_dir, input_schema
    )
    if primary_input_hashes == alternate_input_hashes:
        raise ValueError(
            "lifecycle probe requires distinct primary and alternate inputs"
        )

    sessions: list[C.c_void_p] = []
    keepalive_a: list[object] = []
    keepalive_b: list[object] = []
    keepalive_fresh: list[object] = []
    try:
        session_a = create_session(library, bundle)
        sessions.append(session_a)
        session_b = create_session(library, bundle)
        sessions.append(session_b)

        baseline = run_values(
            api,
            session_a,
            input_schema,
            output_schema,
            values,
            revision=1,
            device=device,
            keepalive=keepalive_a,
        )
        isolated_alternate = run_values(
            api,
            session_b,
            input_schema,
            output_schema,
            alternate,
            revision=1,
            device=device,
            keepalive=keepalive_b,
        )
        alternate_same_session = run_values(
            api,
            session_a,
            input_schema,
            output_schema,
            alternate,
            revision=2,
            device=device,
            keepalive=keepalive_a,
        )
        isolated_primary = run_values(
            api,
            session_b,
            input_schema,
            output_schema,
            values,
            revision=2,
            device=device,
            keepalive=keepalive_b,
        )
        repeated_primary = run_values(
            api,
            session_a,
            input_schema,
            output_schema,
            values,
            revision=3,
            device=device,
            keepalive=keepalive_a,
        )
        repeated_alternate = run_values(
            api,
            session_b,
            input_schema,
            output_schema,
            alternate,
            revision=3,
            device=device,
            keepalive=keepalive_b,
        )

        rejected_reset = api.reset_episode(session_a, 0)
        outputs_after_rejected_reset, rejected_reset_read_errors = output_hashes(
            api, session_a, output_schema, device=device
        )
        if rejected_reset_read_errors:
            raise RuntimeError(
                "outputs became unavailable after a rejected reset: "
                + "; ".join(rejected_reset_read_errors)
            )

        accepted_reset = api.reset_episode(session_a, 1)
        if accepted_reset.code:
            raise RuntimeError(
                "valid reset failed: " + status_message(accepted_reset)
            )
        outputs_after_accepted_reset, accepted_reset_read_errors = output_hashes(
            api, session_a, output_schema, device=device
        )
        if not accepted_reset_read_errors:
            raise RuntimeError(
                "valid reset did not invalidate the prior committed outputs"
            )

        reset_primary = run_values(
            api,
            session_a,
            input_schema,
            output_schema,
            values,
            revision=4,
            device=device,
            keepalive=keepalive_a,
        )
        session_b_after_a_reset, session_b_after_a_reset_errors = output_hashes(
            api, session_b, output_schema, device=device
        )
        if session_b_after_a_reset_errors:
            raise RuntimeError(
                "resetting Session A affected Session B: "
                + "; ".join(session_b_after_a_reset_errors)
            )

        reset_b = api.reset_episode(session_b, 1)
        if reset_b.code:
            raise RuntimeError(
                "valid second-Session reset failed: " + status_message(reset_b)
            )
        reset_alternate = run_values(
            api,
            session_b,
            input_schema,
            output_schema,
            alternate,
            revision=4,
            device=device,
            keepalive=keepalive_b,
        )

        api.destroy(session_a)
        sessions.remove(session_a)
        fresh_session = create_session(library, bundle)
        sessions.append(fresh_session)
        fresh_primary = run_values(
            api,
            fresh_session,
            input_schema,
            output_schema,
            values,
            revision=1,
            device=device,
            keepalive=keepalive_fresh,
        )

        comparisons = {
            "same_session_repeat_is_stable": [
                repeated_primary,
                baseline,
            ],
            "second_session_primary_matches": [
                isolated_primary,
                baseline,
            ],
            "second_session_alternate_matches": [
                isolated_alternate,
                alternate_same_session,
            ],
            "second_session_repeat_is_stable": [
                repeated_alternate,
                isolated_alternate,
            ],
            "same_input_after_rejected_reset": [
                outputs_after_rejected_reset,
                baseline,
            ],
            "same_input_after_accepted_reset": [
                reset_primary,
                baseline,
            ],
            "second_session_survives_first_reset": [
                session_b_after_a_reset,
                isolated_alternate,
            ],
            "same_alternate_after_second_reset": [
                reset_alternate,
                isolated_alternate,
            ],
            "fresh_session_is_stable": [
                fresh_primary,
                baseline,
            ],
        }
        if baseline == alternate_same_session:
            raise RuntimeError(
                "primary and alternate inputs produced identical outputs"
            )
        for name, (actual, expected) in comparisons.items():
            if actual != expected:
                raise RuntimeError(f"lifecycle comparison failed: {name}")

        return {
            "schema": "vlaforge.native_session_lifecycle/1",
            "status": "passed",
            "primary_input_sha256": primary_input_hashes,
            "alternate_input_sha256": alternate_input_hashes,
            "baseline_outputs_sha256": baseline,
            "alternate_outputs_sha256": alternate_same_session,
            "repeated_primary_outputs_sha256": repeated_primary,
            "repeated_alternate_outputs_sha256": repeated_alternate,
            "isolated_primary_outputs_sha256": isolated_primary,
            "isolated_alternate_outputs_sha256": isolated_alternate,
            "outputs_after_rejected_reset": outputs_after_rejected_reset,
            "outputs_after_accepted_reset": outputs_after_accepted_reset,
            "accepted_reset_invalidated_outputs": True,
            "reset_primary_outputs_sha256": reset_primary,
            "session_b_outputs_after_session_a_reset": session_b_after_a_reset,
            "reset_alternate_outputs_sha256": reset_alternate,
            "fresh_session_primary_outputs_sha256": fresh_primary,
            "rejected_reset": nonempty_status(rejected_reset),
            "accepted_reset": nonempty_status(accepted_reset),
            "second_session_reset": nonempty_status(reset_b),
            "same_session_repeat_stable": True,
            "two_session_interleaving_isolated": True,
            "destroy_recreate_stable": True,
        }
    finally:
        for session in sessions:
            api.destroy(session)


def output_hashes(
    api: Api,
    session: C.c_void_p,
    output_schema: dict,
    *,
    device: str,
) -> tuple[dict[str, str], list[str]]:
    import torch

    hashes = {}
    errors = []
    for spec in output_schema["outputs"]:
        bound = BoundTensor()
        status = api.read_output_tensor(
            session, int(spec["output_id"]), C.byref(bound)
        )
        if status.code:
            errors.append(
                f"{spec['name']}: {status_message(status)}"
                if status_message(status)
                else spec["name"]
            )
            continue
        expected = tensor_bytes(spec)
        if (
            bound.struct_size != C.sizeof(BoundTensor)
            or bound.layout != 0
            or not bound.tensor.data
            or bound.tensor.size_bytes != expected
            or bound.tensor.rank != len(spec["payload"]["shape"])
            or (
                bound.tensor.rank
                and tuple(
                    int(bound.tensor.dimensions[index])
                    for index in range(bound.tensor.rank)
                )
                != tuple(spec["payload"]["shape"])
            )
        ):
            errors.append(f"{spec['name']}: output metadata differs")
            continue
        dtype = getattr(torch, TORCH_DTYPES[spec["payload"]["dtype"]])
        host = torch.empty(spec["payload"]["shape"], dtype=dtype, device="cpu")
        cuda = C.CDLL("libcudart.so")
        cuda.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
        cuda.cudaMemcpy.restype = C.c_int
        if cuda.cudaMemcpy(host.data_ptr(), bound.tensor.data, expected, 2):
            errors.append(f"{spec['name']}: complete D2H failed")
            continue
        torch.cuda.synchronize(device)
        hashes[spec["name"]] = hashlib.sha256(
            host.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    return hashes, errors


def bind_inputs(
    api: Api,
    session: C.c_void_p,
    input_schema: dict,
    values: dict[int, object],
    *,
    revision: int,
    keepalive: list[object] | None = None,
) -> list[Status]:
    statuses = []
    for spec in input_schema["inputs"]:
        value = values[int(spec["input_id"])]
        dimensions = (C.c_int64 * value.ndim)(*value.shape)
        tensor = BoundTensor(
            C.sizeof(BoundTensor),
            Tensor(
                value.data_ptr(),
                value.numel() * value.element_size(),
                dimensions,
                value.ndim,
                DTYPE_CODES[spec["payload"]["dtype"]],
                Device(*device_tuple(spec["device"])),
            ),
            0,
            int(spec["alignment"]),
        )
        stamp = Stamp()
        stamp.struct_size = C.sizeof(Stamp)
        stamp.has_revision = 1
        stamp.revision = revision
        if keepalive is not None:
            keepalive.append((dimensions, tensor, stamp))
        statuses.append(
            api.bind_tensor(
                session,
                int(spec["input_id"]),
                C.byref(tensor),
                C.byref(stamp),
            )
        )
    return statuses


def bind_with_shape_mutation(
    api: Api,
    session: C.c_void_p,
    spec: dict,
    value,
    shape: list[int],
    *,
    revision: int,
) -> dict[str, object]:
    dimensions = (C.c_int64 * len(shape))(*shape)
    tensor = BoundTensor(
        C.sizeof(BoundTensor),
        Tensor(
            value.data_ptr(),
            value.numel() * value.element_size(),
            dimensions,
            len(shape),
            DTYPE_CODES[spec["payload"]["dtype"]],
            Device(*device_tuple(spec["device"])),
        ),
        0,
        int(spec["alignment"]),
    )
    stamp = Stamp()
    stamp.struct_size = C.sizeof(Stamp)
    stamp.has_revision = 1
    stamp.revision = revision
    status = api.bind_tensor(
        session, int(spec["input_id"]), C.byref(tensor), C.byref(stamp)
    )
    return {
        "input": spec["name"],
        "shape": shape,
        "size_bytes": int(value.numel() * value.element_size()),
        **nonempty_status(status),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--alternate-sample-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--lifecycle",
        action="store_true",
        help=(
            "also probe repeated calls, two-Session interleaving, reset and "
            "destroy/recreate behavior"
        ),
    )
    args = parser.parse_args()

    if args.output.exists():
        raise ValueError("output already exists")
    if sys.byteorder != "little" or C.sizeof(C.c_void_p) != 8:
        raise ValueError("probe requires little-endian 64-bit Python")
    bundle = args.bundle.resolve(strict=True)
    library = args.library.resolve(strict=True)
    input_schema = json.loads(
        (bundle / "metadata/input_schema.json").read_text()
    )
    output_schema = json.loads(
        (bundle / "metadata/output_schema.json").read_text()
    )

    import torch

    torch.cuda.set_device(args.device)
    values = load_inputs(args.sample_dir, input_schema, device=args.device)
    alternate_dir = args.alternate_sample_dir or args.sample_dir
    alternate = load_inputs(alternate_dir, input_schema, device=args.device)
    lib = C.CDLL(str(library), mode=C.RTLD_LOCAL)
    get_api = lib.vlaforge_model_session_api
    get_api.argtypes = []
    get_api.restype = C.POINTER(Api)
    api_pointer = get_api()
    if not api_pointer:
        raise RuntimeError("generated library returned a null Session ABI")
    api = api_pointer.contents
    if api.struct_size != C.sizeof(Api) or api.abi_version != 2:
        raise RuntimeError("generated library returned an incompatible Session ABI")

    session = create_session(lib, bundle)

    try:
        keepalive: list[object] = []
        initial_statuses = bind_inputs(
            api,
            session,
            input_schema,
            values,
            revision=1,
            keepalive=keepalive,
        )
        if any(item.code for item in initial_statuses):
            raise RuntimeError("valid input binding was rejected")
        initial_run = api.run(session)
        if initial_run.code:
            raise RuntimeError(
                f"valid native run failed: {status_message(initial_run)}"
            )
        baseline_hashes, baseline_errors = output_hashes(
            api, session, output_schema, device=args.device
        )
        if baseline_errors:
            raise RuntimeError("baseline output read failed: " + "; ".join(baseline_errors))

        shape_rejections = []
        revision = 2
        for spec in input_schema["inputs"]:
            if spec["payload"]["kind"] != "tensor":
                continue
            value = values[int(spec["input_id"])]
            shape = list(spec["payload"]["shape"])
            for dimension, extent in enumerate(shape):
                if len(shape) == 1 and extent == 1:
                    replacement = 2
                else:
                    replacement = extent + 1 if dimension == 0 else extent - 1
                if replacement == extent or replacement < 1:
                    continue
                changed = list(shape)
                changed[dimension] = replacement
                shape_rejections.append(
                    bind_with_shape_mutation(
                        api,
                        session,
                        spec,
                        value,
                        changed,
                        revision=revision,
                    )
                )
                revision += 1

        accepted = next(
            (
                spec
                for spec in input_schema["inputs"]
                if spec["name"] == "accepted"
            ),
            None,
        )
        accepted_rejection = None
        isolation_hashes = None
        if accepted is not None:
            alternate_values = dict(alternate)
            alternate_values[int(accepted["input_id"])] = torch.zeros(
                accepted["payload"]["shape"],
                dtype=torch.bool,
                device=args.device,
            )
            rejected_bind = bind_inputs(
                api,
                session,
                input_schema,
                alternate_values,
                revision=revision,
                keepalive=keepalive,
            )
            if any(item.code for item in rejected_bind):
                raise RuntimeError("accepted=false input could not be bound")
            rejected_run = api.run(session)
            accepted_rejection = nonempty_status(rejected_run)
            if rejected_run.code == 0:
                raise RuntimeError("accepted=false native run unexpectedly committed")
            isolation_hashes, read_errors = output_hashes(
                api, session, output_schema, device=args.device
            )
            if read_errors:
                isolation_hashes = None
            elif isolation_hashes != baseline_hashes:
                raise RuntimeError(
                    "rejected accepted=false call polluted committed outputs"
                )
            revision += 1

        final_statuses = bind_inputs(
            api,
            session,
            input_schema,
            values,
            revision=revision,
            keepalive=keepalive,
        )
        if any(item.code for item in final_statuses):
            raise RuntimeError("valid input rebinding after rejection failed")
        final_run = api.run(session)
        if final_run.code:
            raise RuntimeError(
                f"valid run after rejection failed: {status_message(final_run)}"
            )
        final_hashes, final_errors = output_hashes(
            api, session, output_schema, device=args.device
        )
        if final_errors or final_hashes != baseline_hashes:
            raise RuntimeError("valid output changed after rejected calls")

        if not shape_rejections or not all(
            item["rejected"] for item in shape_rejections
        ):
            raise RuntimeError("one or more malformed shapes were not rejected")
        if accepted_rejection is None or not accepted_rejection["rejected"]:
            raise RuntimeError("accepted=false rejection was not exercised")
        lifecycle = None
        if args.lifecycle:
            if args.alternate_sample_dir is None:
                raise ValueError(
                    "--lifecycle requires --alternate-sample-dir"
                )
            lifecycle = probe_lifecycle(
                lib,
                api,
                bundle,
                input_schema,
                output_schema,
                values,
                alternate,
                sample_dir=args.sample_dir.resolve(strict=True),
                alternate_sample_dir=args.alternate_sample_dir.resolve(
                    strict=True
                ),
                device=args.device,
            )
        report = {
            "schema": (
                "vlaforge.native_session_contract_probe/2"
                if lifecycle is not None
                else "vlaforge.native_session_contract_probe/1"
            ),
            "status": "passed",
            "scope": (
                "host-side generated C ABI rejection and Session lifecycle "
                "probe; not deployment timing or no-Python evidence"
                if lifecycle is not None
                else
                "host-side generated C ABI rejection probe; not deployment "
                "timing or no-Python evidence"
            ),
            "bundle": str(bundle),
            "bundle_manifest_sha256": sha256(bundle / "bundle.json"),
            "library": str(library),
            "library_sha256": sha256(library),
            "input_schema_sha256": sha256(
                bundle / "metadata/input_schema.json"
            ),
            "output_schema_sha256": sha256(
                bundle / "metadata/output_schema.json"
            ),
            "supported_profile": {
                "inputs": [
                    {
                        "name": spec["name"],
                        "dtype": spec["payload"]["dtype"],
                        "shape": spec["payload"]["shape"],
                        "device": spec["device"],
                    }
                    for spec in input_schema["inputs"]
                ],
                "outputs": [
                    {
                        "name": spec["name"],
                        "dtype": spec["payload"]["dtype"],
                        "shape": spec["payload"]["shape"],
                        "device": spec["device"],
                    }
                    for spec in output_schema["outputs"]
                ],
            },
            "baseline_outputs_sha256": baseline_hashes,
            "shape_rejections": shape_rejections,
            "accepted_false_rejection": accepted_rejection,
            "committed_outputs_after_rejection": isolation_hashes,
            "final_outputs_sha256": final_hashes,
            "valid_outputs_unchanged_after_rejection": True,
        }
        if lifecycle is not None:
            report["lifecycle"] = lifecycle
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
        print(json.dumps({
            "status": report["status"],
            "shape_rejections": len(shape_rejections),
            "accepted_false_rejected": accepted_rejection["rejected"],
            "outputs_unchanged": True,
            "lifecycle_passed": (
                lifecycle is not None and lifecycle["status"] == "passed"
            ),
        }, indent=2))
    finally:
        if session:
            api.destroy(session)


if __name__ == "__main__":
    main()
