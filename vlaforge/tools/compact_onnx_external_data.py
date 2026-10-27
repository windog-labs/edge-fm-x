#!/usr/bin/env python3
"""Publish an extracted ONNX model with only its referenced external byte unions.

No tensor payload is decoded and no ONNX external-data saver is called. All
source descriptors are read-only; publication never uses links or overwrites.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from export_torchscript_region_onnx import digest, external_files


_BITS = {
    "FLOAT": 32, "UINT8": 8, "INT8": 8, "UINT16": 16, "INT16": 16,
    "INT32": 32, "INT64": 64, "BOOL": 8, "FLOAT16": 16, "DOUBLE": 64,
    "UINT32": 32, "UINT64": 64, "COMPLEX64": 64, "COMPLEX128": 128,
    "BFLOAT16": 16, "FLOAT8E4M3FN": 8, "FLOAT8E4M3FNUZ": 8,
    "FLOAT8E5M2": 8, "FLOAT8E5M2FNUZ": 8, "FLOAT8E8M0": 8,
    "UINT4": 4, "INT4": 4, "FLOAT4E2M1": 4, "UINT2": 2, "INT2": 2,
}
_MAX_EXTENT = (1 << 63) - 1
_CHUNK_BYTES = 8 << 20


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensors(message, path="model"):
    import onnx

    if isinstance(message, onnx.TensorProto):
        yield path, message
        return
    for field, value in message.ListFields():
        if field.message_type is not None:
            if field.is_repeated:
                for index, child in enumerate(value):
                    yield from _tensors(child, f"{path}.{field.name}[{index}]")
            else:
                yield from _tensors(value, f"{path}.{field.name}")


def _metadata_digest(model, *, without_external=False):
    if without_external:
        import onnx

        copy = onnx.ModelProto()
        copy.CopyFrom(model)
        for _, tensor in _tensors(copy):
            tensor.ClearField("external_data")
        model = copy
    return hashlib.sha256(model.SerializeToString()).hexdigest()


def _no_symlinks(path):
    for component in [*reversed(path.parents), path]:
        if component.is_symlink():
            raise ValueError("source/output paths must not contain symlinks: " + str(component))


def _number(text, name):
    _require(isinstance(text, str) and re.fullmatch(r"[0-9]+", text) is not None,
             "external " + name + " must be a nonnegative decimal integer")
    result = int(text)
    _require(result <= _MAX_EXTENT, "external " + name + " exceeds signed 64-bit file extent")
    return result


def _tensor_bytes(tensor):
    import onnx

    try:
        dtype = onnx.TensorProto.DataType.Name(tensor.data_type)
    except ValueError as error:
        raise ValueError("unknown external tensor dtype") from error
    _require(dtype in _BITS, "external tensor dtype is not a supported fixed-width type: " + dtype)
    _require(not tensor.HasField("segment"), "segmented external tensor storage is unsupported")
    count = 1
    for dimension in tensor.dims:
        _require(dimension >= 0, "external tensor has a negative dimension")
        count *= dimension
    size = (count * _BITS[dtype] + 7) // 8
    _require(size <= _MAX_EXTENT, "external tensor byte extent exceeds signed 64-bit range")
    return dtype, size


def _stat(info):
    return {"device": info.st_dev, "inode": info.st_ino, "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
            "links": info.st_nlink}


def _check_source(path, stream, before):
    _no_symlinks(path)
    current = path.stat(follow_symlinks=False)
    _require(stat.S_ISREG(current.st_mode) and current.st_nlink == 1,
             "source external file must remain regular and single-link")
    _require(_stat(current) == before == _stat(os.fstat(stream.fileno())),
             "source external file identity/stat changed during publication")


def _check_output(path, before):
    _require(not path.is_symlink(), "published sidecar became a symlink")
    current = path.stat(follow_symlinks=False)
    _require(stat.S_ISREG(current.st_mode) and current.st_nlink == 1 and _stat(current) == before,
             "published sidecar identity/stat changed after copy")


def _scan(stream, start, length, members, *, output=None, chunk_bytes=_CHUNK_BYTES):
    """Hash a union and distinct member ranges in one bounded-buffer sweep."""
    whole = hashlib.sha256()
    hashes = {(a, n): hashlib.sha256() for a, n in members}
    pending = sorted(members)
    active = []
    next_member = 0
    position = start
    stream.seek(start)
    while position < start + length:
        count = min(chunk_bytes, start + length - position)
        block = stream.read(count)
        _require(len(block) == count, "source/output external range is truncated")
        if output is not None:
            written = output.write(block)
            _require(written == len(block), "short external sidecar write")
        whole.update(block)
        end = position + len(block)
        while next_member < len(pending) and pending[next_member][0] < end:
            active.append(pending[next_member])
            next_member += 1
        remaining = []
        view = memoryview(block)
        for begin, size in active:
            left, right = max(begin, position), min(begin + size, end)
            if left < right:
                hashes[(begin, size)].update(view[left - position:right - position])
            if begin + size > end:
                remaining.append((begin, size))
        active = remaining
        position = end
    return whole.hexdigest(), {key: value.hexdigest() for key, value in hashes.items()}


def _fresh_file(path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    return os.fdopen(os.open(path, flags, 0o600), "wb")


def _write_json(path, value):
    with _fresh_file(path) as output:
        output.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
        output.flush()
        os.fsync(output.fileno())


def publish(model, source_dir, output_dir, *, model_name="model.onnx",
            data_name="model.onnx.data", chunk_bytes=_CHUNK_BYTES):
    """Return ``(copied_model, report)`` after publishing a self-contained part.

    Input must be metadata loaded with ``load_external_data=False``. The output
    directory must be fresh, its parent must exist, and it must be outside the
    source tree. Source pins cover only referenced unions plus stable file
    identity/stats; no unused source bytes or neural tensor values are loaded.
    Failures preserve a fresh failed directory/receipt and publish no owned
    graph. The caller must never treat a raised publication as a valid part.
    """
    import onnx

    source = Path(source_dir).absolute()
    output = Path(output_dir).absolute()
    _no_symlinks(source)
    _no_symlinks(output)
    source = source.resolve(strict=True)
    _require(source.is_dir(), "external source root must be a directory")
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    _require(output != source and source not in output.parents,
             "fresh output directory must be outside the source tree")
    for name in (model_name, data_name):
        _require(isinstance(name, str) and name not in ("", ".", "..") and
                 Path(name).name == name and "\\" not in name and "\0" not in name,
                 "published graph/sidecar names must be plain basenames")
    _require(len({model_name, data_name, "compaction.json", "failure.json"}) == 4,
             "published filenames must not alias one another")
    _require(type(chunk_bytes) is int and 0 < chunk_bytes <= _CHUNK_BYTES,
             "stream chunk size must be between 1 and 8 MiB")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    references = []
    for path, tensor in _tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            _require(not tensor.external_data, "non-external tensor carries ambiguous external metadata")
            continue
        entries = {}
        for entry in tensor.external_data:
            _require(entry.key not in entries, "duplicate external metadata key")
            _require(entry.key in {"location", "offset", "length"},
                     "unsupported external metadata key (checksum/basepath require a separate policy): " + entry.key)
            entries[entry.key] = entry.value
        location = entries.get("location", "")
        relative = Path(location)
        _require(location and "\\" not in location and "\0" not in location and
                 not relative.is_absolute() and ".." not in relative.parts,
                 "external tensor location must stay inside the source directory")
        _require(not any(field.name.endswith("_data") and field.name != "external_data"
                         for field, _ in tensor.ListFields()),
                 "external tensor has materialized or conflicting inline payload")
        dtype, expected = _tensor_bytes(tensor)
        offset = _number(entries.get("offset", "0"), "offset")
        length = _number(entries["length"], "length") if "length" in entries else None
        references.append({"tensor_path": path, "tensor_name": tensor.name, "dtype": dtype,
                           "shape": list(tensor.dims), "location": relative.as_posix(),
                           "source_offset": offset, "length": length, "expected_bytes": expected})

    # Reuse the exporter's recursive discovery/path contract, without its
    # hard-link publisher. Validation above also rejects duplicate key tricks.
    locations = external_files(model, source)
    original_sha = _metadata_digest(model)
    body_sha = _metadata_digest(model, without_external=True)
    copied = onnx.ModelProto()
    copied.CopyFrom(model)
    copied_tensors = dict(_tensors(copied))
    created_graph = None
    output_created = False
    report = {"schema": "vlaforge.compact_onnx_external_data/1", "status": "validating",
              "source_dir": str(source), "output_dir": str(output),
              "source_metadata_sha256": original_sha, "source_metadata_body_sha256": body_sha,
              "metadata_only": True, "tensor_values_decoded": False, "source_files_written": False,
              "full_source_payload_rehashed": False, "source_pin_scope": "referenced-range SHA256 plus file identity/stat before and after",
              "source_files": [], "unions": [], "ranges": [], "tensors": references}
    try:
        with ExitStack() as stack:
            files = {}
            for relative in locations:
                path = source / relative
                _no_symlinks(path)
                info = path.stat(follow_symlinks=False)
                _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                         "external source must be a regular single-link file; hardlinks are forbidden")
                stream = stack.enter_context(os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb", buffering=0))
                before = _stat(info)
                _check_source(path, stream, before)
                files[relative.as_posix()] = {"path": path, "stream": stream, "before": before}
            distinct = {}
            for ref in references:
                f = files[ref["location"]]
                available = f["before"]["bytes"] - ref["source_offset"]
                length = available if ref["length"] is None else ref["length"]
                _require(available >= 0 and 0 <= length <= available and
                         ref["source_offset"] + length <= _MAX_EXTENT,
                         "external offset/length exceeds source file extent or is truncated")
                _require(length == ref["expected_bytes"], "external length differs from fixed-width dtype/shape extent")
                ref["length"] = length
                key = (ref["location"], ref["source_offset"], length)
                distinct.setdefault(key, []).append(ref)
            plans = []
            destination_offset = 0
            for location in sorted(files):
                spans = sorted((start, size) for loc, start, size in distinct if loc == location and size)
                unions = []
                for begin, size in spans:
                    end = begin + size
                    if unions and begin <= unions[-1][1]:
                        unions[-1][1] = max(unions[-1][1], end)
                    else:
                        unions.append([begin, end])
                for begin, end in unions:
                    members = [(a, n) for a, n in spans if begin <= a and a + n <= end]
                    before_sha, member_sha = _scan(files[location]["stream"], begin, end - begin, members, chunk_bytes=chunk_bytes)
                    plans.append({"location": location, "start": begin, "length": end - begin,
                                  "output_offset": destination_offset, "members": members,
                                  "before_sha": before_sha, "member_sha": member_sha})
                    destination_offset += end - begin
            for f in files.values():
                _check_source(f["path"], f["stream"], f["before"])
            output.mkdir(mode=0o700)
            output_created = True
            output_stat = output.stat()
            sidecar = output / data_name
            with _fresh_file(sidecar) as target:
                for plan in plans:
                    whole, members = _scan(files[plan["location"]]["stream"], plan["start"], plan["length"],
                                           plan["members"], output=target, chunk_bytes=chunk_bytes)
                    _require(whole == plan["before_sha"] and members == plan["member_sha"],
                             "source referenced bytes changed between pin and copy")
                target.flush()
                os.fsync(target.fileno())
            _require(sidecar.stat().st_nlink == 1 and sidecar.stat().st_size == destination_offset,
                     "published sidecar is not complete/single-link")
            sidecar_pin = _stat(sidecar.stat(follow_symlinks=False))
            for plan in plans:
                source_after, member_after = _scan(files[plan["location"]]["stream"], plan["start"], plan["length"],
                                                  plan["members"], chunk_bytes=chunk_bytes)
                translated = [(plan["output_offset"] + a - plan["start"], n) for a, n in plan["members"]]
                _check_output(sidecar, sidecar_pin)
                with sidecar.open("rb", buffering=0) as target:
                    target_sha, target_members = _scan(target, plan["output_offset"], plan["length"], translated, chunk_bytes=chunk_bytes)
                _require(source_after == plan["before_sha"] == target_sha and member_after == plan["member_sha"],
                         "source/output referenced range SHA256 changed")
                union_id = len(report["unions"])
                report["unions"].append({"id": union_id, "source_location": plan["location"], "source_offset": plan["start"],
                    "bytes": plan["length"], "output_offset": plan["output_offset"],
                    "source_sha256_before": plan["before_sha"], "source_sha256_after": source_after,
                    "output_sha256": target_sha})
                for start, size in plan["members"]:
                    new_offset = plan["output_offset"] + start - plan["start"]
                    expected = plan["member_sha"][(start, size)]
                    _require(target_members[(new_offset, size)] == expected, "independent member range SHA256 mismatch")
                    aliases = distinct[(plan["location"], start, size)]
                    range_id = len(report["ranges"])
                    report["ranges"].append({"id": range_id, "union_id": union_id, "source_location": plan["location"],
                        "source_offset": start, "bytes": size, "output_offset": new_offset,
                        "source_sha256_before": expected, "source_sha256_after": member_after[(start, size)],
                        "output_sha256": target_members[(new_offset, size)], "tensor_reference_count": len(aliases)})
                    for ref in aliases:
                        ref.update(output_offset=new_offset, range_id=range_id)
            for (location, start, size), aliases in distinct.items():
                if size == 0:
                    range_id = len(report["ranges"])
                    empty = hashlib.sha256(b"").hexdigest()
                    report["ranges"].append({"id": range_id, "union_id": None, "source_location": location,
                        "source_offset": start, "bytes": 0, "output_offset": destination_offset,
                        "source_sha256_before": empty, "source_sha256_after": empty, "output_sha256": empty,
                        "tensor_reference_count": len(aliases)})
                    for ref in aliases:
                        ref.update(output_offset=destination_offset, range_id=range_id)
            for ref in references:
                tensor = copied_tensors[ref["tensor_path"]]
                tensor.ClearField("external_data")
                for key, value in (("location", data_name), ("offset", str(ref["output_offset"])), ("length", str(ref["length"]))):
                    tensor.external_data.add(key=key, value=value)
            _require(_metadata_digest(copied, without_external=True) == body_sha,
                     "tensor values/dtypes/shapes or non-address metadata changed")
            _require(_metadata_digest(model) == original_sha, "caller source model metadata mutated")
            for location, f in files.items():
                _check_source(f["path"], f["stream"], f["before"])
                report["source_files"].append({"location": location, "before": f["before"], "after": _stat(os.fstat(f["stream"].fileno()))})
            logical = sum(ref["length"] for ref in references)
            unique = sum(size for _, _, size in distinct)
            report.update(status="published-byte-verified-not-runtime-validated", external_tensors=len(references),
                          unique_ranges=len(distinct), source_used_bytes=logical, source_unique_range_bytes=unique,
                          source_union_bytes=destination_offset, copied_bytes=destination_offset,
                          exact_duplicate_bytes=logical - unique, overlap_duplicate_bytes=unique - destination_offset,
                          total_duplicate_bytes=logical - destination_offset,
                          source_unreferenced_bytes=sum(f["before"]["bytes"] for f in files.values()) - destination_offset,
                          source_read_bytes=3 * destination_offset, source_metadata_unchanged=True,
                          non_address_metadata_unchanged=True,
                          sidecar={"location": data_name, "bytes": destination_offset, "sha256": digest(sidecar), "links": 1})
            _check_output(sidecar, sidecar_pin)
            payload = copied.SerializeToString()
            model_path = output / model_name
            with _fresh_file(model_path) as target:
                created_graph = _stat(os.fstat(target.fileno()))
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            _require(model_path.stat().st_nlink == 1 and digest(model_path) == hashlib.sha256(payload).hexdigest(),
                     "published model metadata bytes changed")
            _require(output.stat().st_ino == output_stat.st_ino and output.stat().st_dev == output_stat.st_dev,
                     "publication output directory changed")
            for f in files.values():
                _check_source(f["path"], f["stream"], f["before"])
            _check_output(sidecar, sidecar_pin)
            _require(_metadata_digest(model) == original_sha, "caller source model changed during publication")
            report["model"] = {"location": model_name, "bytes": len(payload), "sha256": digest(model_path), "links": 1}
            _write_json(output / "compaction.json", report)
            return copied, report
    except Exception as error:
        if output_created:
            graph = output / model_name
            if created_graph is not None and graph.exists() and not graph.is_symlink():
                current = graph.stat(follow_symlinks=False)
                if (current.st_dev, current.st_ino) == (created_graph["device"], created_graph["inode"]):
                    graph.unlink()
            failure = {**report, "status": "failed", "error": str(error), "valid_published_model": False}
            if not (output / "failure.json").exists():
                _write_json(output / "failure.json", failure)
        raise


def main():
    import onnx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="model.onnx")
    parser.add_argument("--data-name", default="model.onnx.data")
    args = parser.parse_args()
    before = digest(args.input)
    model = onnx.load(args.input, load_external_data=False)
    _, report = publish(model, args.input.absolute().parent, args.output_dir,
                        model_name=args.model_name, data_name=args.data_name)
    _require(digest(args.input) == before, "CLI source metadata file changed")
    print(json.dumps({"status": report["status"], "model": report["model"], "sidecar": report["sidecar"],
                      "source_used_bytes": report["source_used_bytes"], "source_union_bytes": report["source_union_bytes"],
                      "total_duplicate_bytes": report["total_duplicate_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
