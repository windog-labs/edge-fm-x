"""Opt-in owned-directory loading for compiled Linux Torch 2.10 AOTI packages.

This module does not patch Torch or change environment variables. The ordinary
Torch package loader remains the default for existing callers elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import warnings
import zipfile
from pathlib import Path, PurePosixPath

_CHUNK = 1024 * 1024
_MODEL = "data/aotinductor/model/"
_CONSTANTS = "data/constants/"


class _OwnedDirectory:
    def __init__(self, root):
        self.name = tempfile.mkdtemp(prefix="vlaforge-aoti-", dir=root)

    def cleanup(self):
        if self.name is not None:
            shutil.rmtree(self.name)
            self.name = None


def _verify(stream, sha256: str, size_bytes: int) -> None:
    stream.seek(0)
    if os.fstat(stream.fileno()).st_size != size_bytes:
        raise ValueError("AOTI package size mismatch")
    if hashlib.file_digest(stream, "sha256").hexdigest() != sha256:
        raise ValueError("AOTI package SHA-256 mismatch")
    stream.seek(0)


def _name(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or len(value) > 4096 or path.is_absolute() or str(path) != value
            or ".." in path.parts or value.endswith("/")
            or any(ord(ch) < 32 or ord(ch) == 127 or ch in "\\:" for ch in value)):
        raise ValueError("unsafe AOTI package record path")
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate AOTI metadata field")
        result[key] = value
    return result


def _extract(stream, root: Path) -> tuple[_OwnedDirectory, dict]:
    with zipfile.ZipFile(stream) as archive:
        records = archive.infolist()
        if not records or len(records) > 100000:
            raise ValueError("AOTI package record count exceeds limit")
        prefix = records[0].filename.split("/", 1)[0] + "/"
        seen, selected = set(), {}
        for record in records:
            full = _name(record.filename)
            if not full.startswith(prefix):
                raise ValueError("AOTI package records have inconsistent prefix")
            name = _name(full[len(prefix):])
            if name in seen:
                raise ValueError("duplicate AOTI package record")
            seen.add(name)
            mode = stat.S_IFMT(record.external_attr >> 16)
            if mode not in (0, stat.S_IFREG) or record.flag_bits & 1:
                raise ValueError("AOTI archive links, special files or encryption unsupported")
            if name.startswith(_MODEL):
                output = _name(name[len(_MODEL):])
            elif name.startswith(_CONSTANTS):
                output = PurePosixPath(name).name
            else:
                continue
            if output in selected:
                raise ValueError("AOTI package flattened record collision")
            selected[output] = record
        for name, expected in (("archive_format", b"pt2"), ("archive_version", b"0")):
            record = archive.getinfo(prefix + name)
            if record.file_size > 128 or archive.read(record) != expected:
                raise ValueError("unsupported AOTI archive format/version")
        libraries = [name for name in selected if name.endswith(".so")]
        blobs = [name for name in selected if name.endswith(".blob")]
        if len(libraries) != 1 or len(PurePosixPath(libraries[0]).parts) != 1:
            raise ValueError("AOTI explicit extraction requires exactly one compiled library")
        if len(blobs) > 1:
            raise ValueError("AOTI package contains multiple weight blobs")
        metadata_name = str(PurePosixPath(libraries[0]).with_suffix("")) + "_metadata.json"
        if metadata_name not in selected or selected[metadata_name].file_size > _CHUNK:
            raise ValueError("AOTI package requires bounded shared-library metadata")
        metadata = json.loads(archive.read(selected[metadata_name]), object_pairs_hook=_object)
        if not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values()):
            raise ValueError("invalid AOTI package metadata")
        total = sum(record.file_size for record in selected.values())
        if total > shutil.disk_usage(root).free:
            raise ValueError("insufficient AOTI extraction space")
        directory = _OwnedDirectory(root)
        ledger = []
        try:
            for name, record in sorted(selected.items()):
                destination = Path(directory.name) / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with archive.open(record) as source, destination.open("xb") as output:
                    while payload := source.read(_CHUNK):
                        output.write(payload)
                        digest.update(payload)
                        size += len(payload)
                if size != record.file_size:
                    raise ValueError("AOTI extracted member size mismatch")
                ledger.append({"record": record.filename, "file": name, "size_bytes": size,
                               "sha256": digest.hexdigest()})
            return directory, {"library": libraries[0], "weight_blob": blobs[0] if blobs else None,
                               "metadata": metadata, "members": ledger}
        except BaseException:
            directory.cleanup()
            raise


class AotiPackageCallable:
    """Owned ordinary runner; destroy any externally captured graph before close.

    Calls and close are serialized. CUDA close drains before releasing the
    runner, then cleans only its owned directory. This does not enforce any
    process-global numerical policy or prevent external Torch setters.
    """

    def __init__(self, runner, directory, device, evidence):
        import torch.utils._pytree as pytree

        self._runner = runner
        self._directory = directory
        self._device = device
        self._lock = threading.RLock()
        self.extraction = evidence
        call_spec = runner.get_call_spec()
        self._input_spec = pytree.treespec_loads(call_spec[0])
        self._output_spec = pytree.treespec_loads(call_spec[1])

    def __call__(self, *args, **kwargs):
        import torch
        import torch.utils._pytree as pytree
        from torch.export._tree_utils import reorder_kwargs

        with self._lock:
            if self._runner is None:
                raise RuntimeError("AOTI package callable is closed")
            flat = pytree.tree_flatten((args, reorder_kwargs(kwargs, self._input_spec)))[0]
            # Match Torch 2.10 AOTICompiledModel's specialized non-Tensor ABI.
            flat = [value for value in flat if isinstance(value, torch.Tensor)]
            return pytree.tree_unflatten(self._runner.run(flat), self._output_spec)

    def close(self):
        import torch

        with self._lock:
            if self._runner is not None:
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                self._runner = None
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception as error:  # noqa: BLE001
            # A failed CUDA drain must not remove live backing files. There is
            # no autonomous temporary-directory finalizer that could do so.
            warnings.warn(f"AOTI owner cleanup failed; backing directory retained: {error}", RuntimeWarning, stacklevel=1)


def load_aoti_package(
    path: str | Path, *, extraction_root: str | Path,
    sha256: str, size_bytes: int, device: str,
) -> AotiPackageCallable:
    """Load a verified compiled package without the upstream /tmp loader.

    Explicitly supports Linux Torch 2.10, CPU and CUDA. CUDA uses its public
    single-threaded raw runner constructor. The CPU Python binding exposes only
    the ordinary runner, so this helper makes no CPU single-threaded-mode claim.
    Use as a context manager, or call close() before discarding the owner.
    """
    import torch

    if sys.platform != "linux" or not re.fullmatch(r"2\.10\.\d+(?:\+[A-Za-z0-9_.-]+)?", str(torch.__version__)):
        raise ValueError("explicit AOTI extraction requires audited Linux Torch 2.10")
    if not isinstance(sha256, str) or not re.fullmatch("[0-9a-f]{64}", sha256):
        raise ValueError("AOTI extraction requires lowercase SHA-256")
    if type(size_bytes) is not int or size_bytes <= 0:
        raise ValueError("AOTI extraction requires a positive exact package size")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", device):
        raise ValueError("AOTI extraction requires explicit cpu or cuda:ordinal device")
    target = torch.device(device)
    value = os.fspath(extraction_root)
    root = Path(value)
    info = root.lstat()
    if (not root.is_absolute() or str(root) != value or root.resolve() != root
            or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("AOTI extraction root must be canonical private owned directory (0700)")
    with Path(path).open("rb") as stream:
        _verify(stream, sha256, size_bytes)
        directory, evidence = _extract(stream, root)
        runner = None
        try:
            _verify(stream, sha256, size_bytes)
            if evidence["metadata"].get("AOTI_DEVICE_KEY") != target.type:
                raise ValueError("AOTI package device metadata mismatch")
            library = str(Path(directory.name) / evidence["library"])
            if target.type == "cuda":
                runner = torch._C._aoti.AOTIModelContainerRunnerCuda(library, 1, device, directory.name, True)
            else:
                runner = torch._C._aoti.AOTIModelContainerRunnerCpu(library, 1)
            if evidence["weight_blob"]:
                runner.update_constant_buffer_from_blob(str(Path(directory.name) / evidence["weight_blob"]))
            evidence.update({"schema": "vlaforge.aoti_package_extraction/1", "package": str(Path(path).resolve()),
                             "package_sha256": sha256, "package_size_bytes": size_bytes,
                             "root": str(root), "directory": directory.name, "device": device,
                             "cuda_single_threaded_runner": target.type == "cuda",
                             "environment_modified": False, "shared_cache": False})
            return AotiPackageCallable(runner, directory, target, evidence)
        except BaseException:
            runner = None
            directory.cleanup()
            raise
