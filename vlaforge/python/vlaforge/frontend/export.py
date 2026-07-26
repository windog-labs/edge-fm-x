"""Persistence helpers for verified torch.export captures."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import zipfile
from pathlib import Path

from vlaforge.frontend.region_capture import CaptureOutcome


_LEGACY_EXPORT_MEMBERS = (
    "serialized_exported_program.json",
    "serialized_state_dict.pt",
    "serialized_constants.pt",
    "serialized_example_inputs.pt",
    "version",
)


def save_exported_region(
    capture: CaptureOutcome,
    *,
    program_path: str | Path,
    evidence_path: str | Path,
) -> None:
    capture.require_supported()
    assert capture.exported_program is not None
    assert capture.evidence is not None
    import torch

    program_output = Path(program_path)
    evidence_output = Path(evidence_path)
    program_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(capture.exported_program, program_output)
    evidence_output.write_text(
        json.dumps(
            capture.evidence.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_exported_region(
    path: str | Path,
    *,
    mmap_cache: str | Path | None = None,
) -> object:
    """Load a verified export, optionally mmap-loading legacy tensor storage.

    PyTorch's compatibility loader reads every legacy ``.pt2e`` member into
    Python bytes before deserializing it. A multi-billion-parameter state dict
    is consequently resident twice. ``mmap_cache`` extracts the trusted tensor
    members into a content-addressed local cache and lets ``torch.load`` map
    them instead. Current PT2 archives continue to use PyTorch's native loader.
    """

    import torch

    source = Path(path)
    if mmap_cache is None or not _is_legacy_export(source):
        return torch.export.load(source)
    return _load_legacy_export_mmap(
        torch,
        source,
        Path(mmap_cache),
    )


def _is_legacy_export(path: Path) -> bool:
    with zipfile.ZipFile(path) as archive:
        members = frozenset(archive.namelist())
    return frozenset(_LEGACY_EXPORT_MEMBERS).issubset(members)


def _load_legacy_export_mmap(
    torch: object,
    path: Path,
    cache_root: Path,
) -> object:
    from torch._export.serde.schema import ExportedProgram
    from torch._export.serde import serialize

    graph_decoder = getattr(serialize, "_bytes_to_dataclass", None)
    if graph_decoder is None:
        dict_decoder = serialize._dict_to_dataclass

        def graph_decoder(type_: object, value: bytes) -> object:
            return dict_decoder(type_, json.loads(value.decode("utf-8")))

    cache_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        graph = graph_decoder(
            ExportedProgram,
            archive.read("serialized_exported_program.json"),
        )
        state_path = _cached_member(
            archive,
            "serialized_state_dict.pt",
            cache_root,
        )
        constants_path = _cached_member(
            archive,
            "serialized_constants.pt",
            cache_root,
        )
        example_inputs_path = _cached_member(
            archive,
            "serialized_example_inputs.pt",
            cache_root,
        )

    load_options = {"mmap": True, "weights_only": False}
    state_dict = torch.load(state_path, **load_options)
    constants = torch.load(constants_path, **load_options)
    example_inputs = torch.load(example_inputs_path, **load_options)
    deserializer = serialize.ExportedProgramDeserializer()
    deserialize_options: dict[str, object] = {}
    if "_unsafe_skip_version_check" in inspect.signature(
        deserializer.deserialize
    ).parameters:
        # This path is opt-in, accepts only the fixed legacy archive members
        # above, and is followed by an exact active-version re-export in the
        # AOTI tool.  Newer PyTorch releases otherwise reject older schemas
        # before their compatibility deserializer can process the graph.
        deserialize_options["_unsafe_skip_version_check"] = True
    return deserializer.deserialize(
        graph,
        state_dict,
        constants,
        example_inputs,
        **deserialize_options,
    )


def _cached_member(
    archive: zipfile.ZipFile,
    name: str,
    cache_root: Path,
) -> Path:
    info = archive.getinfo(name)
    identity = hashlib.sha256(
        (
            f"{name}\0{info.CRC:08x}\0{info.file_size}\0"
            f"{info.compress_size}\0{info.compress_type}"
        ).encode()
    ).hexdigest()
    output = cache_root / f"{identity}-{Path(name).name}"
    if output.is_file() and output.stat().st_size == info.file_size:
        return output

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with archive.open(info) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if temporary.stat().st_size != info.file_size:
            raise OSError(
                f"legacy export cache size mismatch for {name}: "
                f"{temporary.stat().st_size} != {info.file_size}"
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
