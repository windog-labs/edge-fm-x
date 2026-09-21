"""Version-pinned translation of export enums to the AOTI proxy's native ABI."""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

_MARKER = "extra/vlaforge-proxy-enums.json"


def _read_json(payload):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate package JSON key: {name}")
            result[name] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite package JSON scalar: {value}")

    return json.loads(payload, object_pairs_hook=unique, parse_constant=invalid)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def native_enum_maps():
    import torch
    from torch._export.serde import serialize

    if torch.__version__.split("+")[0] != "2.10.0":
        raise ValueError("proxy enum translation is verified only for Torch 2.10.0")
    library_path = Path(torch.__file__).parent / "lib/libtorch_cpu.so"
    library = ctypes.CDLL(str(library_path))
    result = {}
    for key, table, category in (
        ("as_scalar_type", serialize._SERIALIZE_TO_TORCH_DTYPE, "dtype"),
        ("as_layout", serialize._SERIALIZE_TO_TORCH_LAYOUT, "layout"),
        (
            "as_memory_format",
            serialize._SERIALIZE_TO_TORCH_MEMORY_FORMAT,
            "memory_format",
        ),
    ):
        mapping = {}
        for encoded, value in table.items():
            symbol = f"aoti_torch_{category}_{str(value).removeprefix('torch.')}"
            function = getattr(library, symbol)
            function.argtypes, function.restype = [], ctypes.c_int32
            mapping[int(encoded)] = {"value": function(), "symbol": symbol}
        result[key] = mapping
    return result, {"torch": torch.__version__, "abi_library": str(library_path)}


def translate_proxy_arguments(document, mapping):
    """Translate typed enum arguments only; preserve ordinary integers and tensors."""
    result = copy.deepcopy(document)
    rewrites = []

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in mapping:
                    if (
                        len(value) != 1
                        or type(child) is not int
                        or child not in mapping[key]
                    ):
                        raise ValueError(f"unsupported serialized proxy enum at {path}")
                    native = mapping[key][child]
                    value[key] = native["value"]
                    rewrites.append(
                        {
                            "path": path,
                            "kind": key,
                            "before": child,
                            "after": native["value"],
                            "abi_symbol": native["symbol"],
                        }
                    )
                else:
                    visit(child, f"{path}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}/{index}")

    if set(result) != {"nodes"} or not isinstance(result["nodes"], list):
        raise ValueError("unknown AOTI proxy document schema")
    for index, node in enumerate(result["nodes"]):
        for argument in node["node"]["inputs"]:
            visit(argument["arg"], f"nodes/{index}/inputs/{argument['name']}/arg")
    return result, rewrites


def normalize_proxy_enums(source, destination):
    """Create a distinct package, keeping original code/weights and all assertions.

    Torch 2.10 export enums are not c10 enum values. OSSProxyExecutor reads the
    JSON values directly as c10 enums. Resolve the translation from that exact
    installation's serializer maps and exported AOTI C getters, not a model table.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("proxy normalization requires a new destination")
    mapping, environment = native_enum_maps()
    report = {
        "schema": "vlaforge.aoti_proxy_enum_translation/1",
        "status": "started",
        **environment,
        "source_sha256": digest(source),
        "rewrites": [],
        "unchanged_members": [],
        "numeric_parity_verified": False,
    }
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if not names or len(names) != len(set(names)):
            raise ValueError("empty package or duplicate archive members")
        if any(name.endswith(_MARKER) for name in names):
            raise ValueError("package proxy enums were already normalized")
        replacements = {}
        for name in names:
            if name.endswith(".wrapper.json"):
                document, rewrites = translate_proxy_arguments(
                    _read_json(archive.read(name)), mapping
                )
                if rewrites:
                    replacements[name] = json.dumps(
                        document, separators=(",", ":")
                    ).encode()
                    report["rewrites"].extend(
                        {"member": name, **item} for item in rewrites
                    )
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
        ) as temporary:
            with zipfile.ZipFile(temporary, "w") as target:
                for info in archive.infolist():
                    if info.filename in replacements:
                        target.writestr(copy.copy(info), replacements[info.filename])
                        continue
                    member_hash = hashlib.sha256()
                    with (
                        archive.open(info) as incoming,
                        target.open(copy.copy(info), "w", force_zip64=True) as outgoing,
                    ):
                        for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                            member_hash.update(block)
                            outgoing.write(block)
                    report["unchanged_members"].append(
                        {"path": info.filename, "sha256": member_hash.hexdigest()}
                    )
                marker = names[0].split("/")[0] + "/" + _MARKER
                target.writestr(
                    marker,
                    json.dumps(
                        {
                            "schema": report["schema"],
                            "source_sha256": report["source_sha256"],
                        }
                    ),
                )
            temporary.flush()
            # Publish atomically without replacing an independently created file.
            # On failure only our private temporary is removed, never destination.
            os.link(temporary.name, destination)
    report.update(
        status="translated_not_numerically_validated",
        artifact_sha256=digest(destination),
    )
    return report


def verify_package_audit(path, configs, audit, *, artifact_sha256=None):
    """Check completed package translation evidence before reuse or deployment.

    This verifies byte/provenance consistency, not model numerical parity. A
    legacy non-translation profile may omit package audit metadata entirely.
    """
    expected = package_pass_records(configs)
    if not isinstance(audit, dict) or audit.get("passes", []) != expected:
        raise ValueError("backend package passes differ")
    translation = audit.get("translation")
    if not expected:
        if translation is not None:
            raise ValueError("unexpected package translation for this profile")
        return
    if (
        not isinstance(translation, dict)
        or translation.get("schema") != "vlaforge.aoti_proxy_enum_translation/1"
        or translation.get("status") != "translated_not_numerically_validated"
        or translation.get("numeric_parity_verified") is not False
        or not isinstance(translation.get("torch"), str)
        or translation["torch"].split("+")[0] != "2.10.0"
        or not isinstance(translation.get("rewrites"), list)
        or not isinstance(translation.get("unchanged_members"), list)
    ):
        raise ValueError("package translation is missing, incomplete, or unsupported")
    actual = artifact_sha256 if artifact_sha256 is not None else digest(path)
    if translation.get("artifact_sha256") != actual:
        raise ValueError("package translation final artifact digest mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        markers = [name for name in names if name.endswith(_MARKER)]
        if len(names) != len(set(names)) or len(markers) != 1:
            raise ValueError("normalized package marker or member set is invalid")
        marker = _read_json(archive.read(markers[0]))
        if marker != {
            "schema": translation["schema"],
            "source_sha256": translation.get("source_sha256"),
        }:
            raise ValueError("package translation source marker mismatch")
        source_sha = translation.get("source_sha256")
        if (
            not isinstance(source_sha, str)
            or len(source_sha) != 64
            or any(char not in "0123456789abcdef" for char in source_sha)
        ):
            raise ValueError("package translation source digest is invalid")
        rewritten = set()
        for rewrite in translation["rewrites"]:
            if not isinstance(rewrite, dict) or not isinstance(
                rewrite.get("member"), str
            ):
                raise ValueError("package translation rewrite record is invalid")  # noqa: TRY004
            name = rewrite["member"]
            if name not in names or not name.endswith(".wrapper.json"):
                raise ValueError("package translation may rewrite wrapper JSON only")
            rewritten.add(name)
        unchanged = {}
        for record in translation["unchanged_members"]:
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise ValueError("unchanged package member record is invalid")
            name = record["path"]
            if not isinstance(name, str) or name in unchanged:
                raise ValueError("duplicate or invalid unchanged package member")
            unchanged[name] = record["sha256"]
        if set(unchanged) != set(names) - rewritten - set(markers):
            raise ValueError("package translation member coverage differs")
        for name, expected_sha in unchanged.items():
            with archive.open(name) as stream:
                actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(f"unchanged package member digest differs: {name}")


def package_pass_records(configs):
    if not configs.get("fallback_by_default"):
        return []
    return [
        {
            "name": "aoti_proxy_export_enum_to_native_abi",
            "source_sha256": digest(Path(__file__)),
            "torch_version": "2.10.0",
            "stage": "post_package_before_artifact_identity",
        }
    ]


def finalize_aoti_package(path, configs):
    """Finalize a newly compiled candidate, retaining its original package."""
    passes = package_pass_records(configs)
    if not passes:
        return {"passes": [], "translation": None}
    path = Path(path)
    original = path.with_suffix(".unfinalized.pt2")
    if original.exists():
        raise ValueError("unfinalized package already exists; use a new attempt")
    os.link(path, original)
    path.unlink()
    return {"passes": passes, "translation": normalize_proxy_enums(original, path)}
