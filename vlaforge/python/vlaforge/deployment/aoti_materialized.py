"""Verified build-time AOTI payloads with deployment-asset file lifetime.

No loaded library is unlinked on close. OS-retained code mappings, including
GNU-unique DSOs, remain distinct from runner-owned model resources.
"""

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from vlaforge.deployment.aoti_load import _extract, _object, _verify

MAGIC = "VLAFORGE_AOTI_MATERIALIZED"
SUFFIX = ".vfaoti"


def _name(value):
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.\-/]{1,4096}", value)
            or value in (".", "-") or PurePosixPath(value).is_absolute()
            or str(PurePosixPath(value)) != value or ".." in PurePosixPath(value).parts):
        raise ValueError("unsafe materialized AOTI relative path")
    return value


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("materialized AOTI SHA-256 must be explicit lowercase hex")
    return value


def _size(value, *, positive=False):
    if type(value) is not int or not (int(positive) <= value <= 2**63 - 1):
        raise ValueError("invalid materialized AOTI byte count")
    return value


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class MaterializedFile:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self):
        _name(self.path)
        _digest(self.sha256)
        _size(self.size_bytes)


@dataclass(frozen=True)
class MaterializedAotiPackage:
    package_sha256: str
    package_size_bytes: int
    device_type: str
    library: str
    weight_blob: str | None
    files: tuple[MaterializedFile, ...]

    def __post_init__(self):
        _digest(self.package_sha256)
        _size(self.package_size_bytes, positive=True)
        if self.device_type not in ("cpu", "cuda"):
            raise ValueError("unsupported materialized AOTI device")
        _name(self.library)
        if self.weight_blob is not None:
            _name(self.weight_blob)
        if not 1 <= len(self.files) <= 100000 or any(not isinstance(item, MaterializedFile) for item in self.files):
            raise ValueError("invalid materialized AOTI member set")
        names = [item.path for item in self.files]
        if names != sorted(set(names)) or self.library not in names or not self.library.endswith(".so"):
            raise ValueError("materialized AOTI files must be sorted, unique and contain their library")
        if [name for name in names if name.endswith(".so")] != [self.library]:
            raise ValueError("materialized AOTI needs exactly one library")
        blobs = [name for name in names if name.endswith(".blob")]
        if blobs != ([] if self.weight_blob is None else [self.weight_blob]):
            raise ValueError("materialized AOTI weight blob declaration differs")
        if str(PurePosixPath(self.library).with_suffix("")) + "_metadata.json" not in names:
            raise ValueError("materialized AOTI library metadata missing")

    def canonical_text(self):
        lines = [MAGIC + " 1", f"package {self.package_sha256} {self.package_size_bytes}",
            f"model {self.device_type} {self.library} {self.weight_blob or '-'}", f"files {len(self.files)}"]
        lines.extend(f"file {item.path} {item.sha256} {item.size_bytes}" for item in self.files)
        return "\n".join([*lines, "end", ""])

    @classmethod
    def parse(cls, text):
        if not isinstance(text, str) or len(text) > 16 * 1024**2:
            raise ValueError("materialized AOTI manifest exceeds limit")
        lines = text.splitlines()
        try:
            if lines[0] != MAGIC + " 1":
                raise ValueError("unsupported materialized AOTI manifest")
            package, model, count = [line.split() for line in lines[1:4]]
            if (len(package) != 3 or package[0] != "package"
                    or len(model) != 4 or model[0] != "model"
                    or len(count) != 2 or count[0] != "files"
                    or len(lines) != int(count[1]) + 5 or lines[-1] != "end"):
                raise ValueError("invalid canonical materialized AOTI manifest")
            files = []
            for line in lines[4:-1]:
                fields = line.split()
                if len(fields) != 4 or fields[0] != "file":
                    raise ValueError("invalid materialized AOTI member declaration")
                files.append(MaterializedFile(fields[1], fields[2], int(fields[3])))
            result = cls(package[1], int(package[2]), model[1], model[2], None if model[3] == "-" else model[3], tuple(files))
            if result.canonical_text() != text:
                raise ValueError("noncanonical materialized AOTI manifest")
            return result
        except (IndexError, TypeError) as error:
            raise ValueError("invalid canonical materialized AOTI manifest") from error

    def verify(self, root):
        root = Path(root).resolve()
        for item in self.files:
            path = root / item.path
            if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root)
                    or any(parent.is_symlink() for parent in path.parents if parent != root and parent.is_relative_to(root))
                    or path.stat().st_size != item.size_bytes or file_digest(path) != item.sha256):
                raise ValueError("materialized AOTI member missing or changed: " + item.path)
        metadata = json.loads((root / (str(PurePosixPath(self.library).with_suffix("")) + "_metadata.json")).read_text(), object_pairs_hook=_object)
        if not isinstance(metadata, dict) or metadata.get("AOTI_DEVICE_KEY") != self.device_type:
            raise ValueError("materialized AOTI signed metadata device mismatch")
        return metadata


def materialize_aoti_package(package, output, *, sha256, size_bytes):
    """Publish checked payloads without loading code or changing archive bytes."""
    _digest(sha256)
    _size(size_bytes, positive=True)
    output = Path(output)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    with Path(package).open("rb") as stream:
        _verify(stream, sha256, size_bytes)
        owner, ledger = _extract(stream, output)
        try:
            _verify(stream, sha256, size_bytes)
            files = tuple(MaterializedFile("payload/" + item["file"], item["sha256"], item["size_bytes"]) for item in ledger["members"])
            result = MaterializedAotiPackage(sha256, size_bytes, ledger["metadata"].get("AOTI_DEVICE_KEY"),
                "payload/" + ledger["library"], None if ledger["weight_blob"] is None else "payload/" + ledger["weight_blob"], files)
            Path(owner.name).rename(output / "payload")
            owner.name = None
            result.verify(output)
            manifest = output / ("model" + SUFFIX)
            with manifest.open("x") as target:
                target.write(result.canonical_text())
            return manifest
        finally:
            owner.cleanup()


def load_materialized_aoti(path, *, sha256, size_bytes, device):
    """Read a verified immutable payload; closing the runner never deletes it."""
    import sys

    import torch

    from vlaforge.deployment.aoti_load import AotiPackageCallable

    if sys.platform != "linux" or str(torch.__version__).split("+")[0] != "2.10.0":
        raise ValueError("materialized AOTI loader requires audited Linux Torch 2.10.0")
    if not re.fullmatch(r"cpu|cuda:[0-9]+", device):
        raise ValueError("materialized AOTI loader requires explicit device")
    path = Path(path)
    with path.open("rb") as stream:
        _verify(stream, _digest(sha256), _size(size_bytes, positive=True))
    package = MaterializedAotiPackage.parse(path.read_text())
    package.verify(path.parent)
    target = torch.device(device)
    if target.type != package.device_type:
        raise ValueError("materialized AOTI target device mismatch")
    library = path.parent / package.library
    runner = (torch._C._aoti.AOTIModelContainerRunnerCuda(str(library), 1, device, str(library.parent), True)
              if target.type == "cuda" else torch._C._aoti.AOTIModelContainerRunnerCpu(str(library), 1))
    if package.weight_blob is not None:
        runner.update_constant_buffer_from_blob(str(path.parent / package.weight_blob))
    evidence = {"schema": "vlaforge.aoti_materialized_load/1", "manifest": str(path.resolve()), "manifest_sha256": sha256,
        "source_package_sha256": package.package_sha256, "file_lifetime": "deployment-asset", "runtime_extraction": False,
        "runtime_file_deletion": False, "device": device, "files": [vars(item) for item in package.files]}
    return AotiPackageCallable(runner, None, target, evidence)


def materialized_region_contract(original, manifest, *, artifact_path):
    """Bind a byte-preserving package transformation to a new artifact identity."""
    from vlaforge.deployment.contract import ArtifactKind
    from vlaforge.deployment.numerical import canonical_json, strict_json

    if original.artifact_kind is not ArtifactKind.AOTI_PACKAGE or original.capability.backend != "aoti":
        raise ValueError("materialization requires an original AOTI package contract")
    manifest = Path(manifest)
    package = MaterializedAotiPackage.parse(manifest.read_text())
    package.verify(manifest.parent)
    if package.package_sha256 != original.artifact_sha256 or package.package_size_bytes != original.artifact_size_bytes:
        raise ValueError("materialization does not match its original compiled artifact")
    digest, size = file_digest(manifest), manifest.stat().st_size
    binding = original.numerical_binding
    if binding is not None:
        configuration = strict_json(binding.compile_record.configuration_json)
        configuration["byte_preserving_materialization"] = {"source_artifact_sha256": original.artifact_sha256,
            "source_compile_record_sha256": binding.compile_record.digest(), "manifest_sha256": digest}
        record = replace(binding.compile_record, artifact_sha256=digest, artifact_size_bytes=size, configuration_json=canonical_json(configuration))
        binding = replace(binding, compile_record=record, requirement=replace(binding.requirement, compile_record_sha256=record.digest()))
    return replace(original, artifact_kind=ArtifactKind.AOTI_MATERIALIZED, artifact_path=artifact_path,
                   artifact_sha256=digest, artifact_size_bytes=size, numerical_binding=binding)
