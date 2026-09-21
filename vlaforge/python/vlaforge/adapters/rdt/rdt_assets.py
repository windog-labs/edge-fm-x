"""Byte-verified official RDT, T5 and SigLIP assets, without model imports.

Mirrors can transport files only after an official, revision-pinned inventory
has fixed their identities. Existing mismatched files are never overwritten.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


RDT_SOURCE_REVISION = "cd79363a1387e8f81c7724d070ef7e45fd23150f"
RDT_ASSETS = {
    "policy": {
        "repo": "robotics-diffusion-transformer/rdt-1b",
        "revision": "eb09036cc64ca4945051acbd1bd581d30a1d7711",
        "license": "mit",
        "files": ("README.md", "config.json", "pytorch_model.bin"),
    },
    "text": {
        "repo": "google/t5-v1_1-xxl",
        "revision": "3db67ab1af984cf10548a73467f0e5bca2aaaeb2",
        "license": "apache-2.0",
        "files": (
            "README.md", "config.json", "tokenizer_config.json",
            "special_tokens_map.json", "spiece.model", "pytorch_model.bin",
        ),
    },
    "vision": {
        "repo": "google/siglip-so400m-patch14-384",
        "revision": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        "license": "apache-2.0",
        "files": (
            "README.md", "config.json", "preprocessor_config.json", "model.safetensors",
        ),
    },
}
TRANSPORT_ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")


def file_identity(path: Path) -> dict:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            sha256.update(block)
            git_blob.update(block)
    return {"size": size, "sha256": sha256.hexdigest(), "blobId": git_blob.hexdigest()}


def selected_records(metadata: dict, component: str) -> tuple[dict, ...]:
    spec = RDT_ASSETS[component]
    if metadata.get("id") != spec["repo"] or metadata.get("sha") != spec["revision"]:
        raise ValueError("repository/revision differs from the pinned official profile")
    if metadata.get("cardData", {}).get("license") != spec["license"]:
        raise ValueError("license differs from the selected official profile")
    siblings = metadata.get("siblings", [])
    names = [item["rfilename"] for item in siblings]
    if len(names) != len(set(names)):
        raise ValueError("duplicate Hub file names")
    by_name = {item["rfilename"]: item for item in siblings}
    records = []
    for name in spec["files"]:
        if name not in by_name:
            raise ValueError(f"required file absent from full Hub inventory: {name}")
        item = by_name[name]
        if type(item.get("size")) is not int or item["size"] < 1:
            raise ValueError(f"invalid file size: {name}")
        if not re.fullmatch(r"[0-9a-f]{40}", item.get("blobId", "")):
            raise ValueError(f"missing Git blob identity: {name}")
        if "lfs" in item and (
            item["lfs"].get("size") != item["size"]
            or not re.fullmatch(r"[0-9a-f]{64}", item["lfs"].get("sha256", ""))
        ):
            raise ValueError(f"invalid LFS file identity: {name}")
        if name.endswith((".bin", ".safetensors")) and "lfs" not in item:
            raise ValueError("model weights require the complete LFS SHA256 identity")
        records.append(dict(item))
    return tuple(records)


def verify_file(path: Path, record: dict) -> dict:
    if not path.is_file() or path.stat().st_size != record["size"]:
        raise ValueError(f"asset missing or size mismatch: {path}")
    actual = file_identity(path)
    if "lfs" in record:
        valid = actual["sha256"] == record["lfs"]["sha256"]
    else:
        valid = actual["blobId"] == record["blobId"]
    if not valid:
        raise ValueError(f"asset content digest mismatch: {path}")
    return actual


def _target(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
        raise ValueError("unsafe asset path")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("asset path escapes component root")
    return path


def verify_assets(root: Path) -> dict:
    components = {}
    for component, spec in RDT_ASSETS.items():
        metadata_path = root / "metadata" / f"{component}.json"
        metadata = json.loads(metadata_path.read_text())
        records = selected_records(metadata, component)
        components[component] = {
            "repo": spec["repo"], "revision": spec["revision"],
            "license": spec["license"], "metadata": file_identity(metadata_path),
            "files": {
                item["rfilename"]: verify_file(
                    _target(root / component, item["rfilename"]), item
                )
                for item in records
            },
        }
    return {
        "schema": "vlaforge.rdt_assets/1", "all_required_assets_verified": True,
        "components": components, "real_model_execution_verified": False,
        "full_pipeline_weights_include_unused_t5_decoder_and_siglip_text": True,
    }


def verify_source(root: Path) -> dict:
    # Older vendor Git versions only accept safe.directory in global/system
    # config. Select a private per-process file, never change the user's config.
    with tempfile.TemporaryDirectory(prefix="rdt-source-git-") as directory:
        config = Path(directory) / "config"
        subprocess.check_call(["git", "config", "--file", str(config), "--add", "safe.directory", str(root.resolve())])
        environment = {**os.environ, "GIT_CONFIG_GLOBAL": str(config)}
        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, env=environment,
            ).strip()
        if git("rev-parse", "HEAD") != RDT_SOURCE_REVISION:
            raise ValueError("RDT source revision mismatch")
        if git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError("RDT tracked source modifications are not an official baseline")
        paths = git("ls-files", "-z").split("\0")
    return {
        "revision": RDT_SOURCE_REVISION,
        "files": {name: file_identity(root / name) for name in paths if name},
        "tracked_source_clean": True,
    }


def freeze_metadata(root: Path) -> dict:
    """Fetch only small metadata from the official Hub at fixed commits."""
    output = root / "metadata"
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for component, spec in RDT_ASSETS.items():
        path = output / f"{component}.json"
        url = f"https://huggingface.co/api/models/{spec['repo']}/revision/{spec['revision']}?blobs=true"
        if path.exists():
            metadata = json.loads(path.read_text())
        else:
            with urlopen(url, timeout=45) as response:
                raw = response.read(2 * 1024**2 + 1)
            if len(raw) > 2 * 1024**2:
                raise ValueError("Hub metadata exceeds expected small inventory bound")
            metadata = json.loads(raw)
            selected_records(metadata, component)
            with path.open("xb") as stream:
                stream.write(raw)
        records = selected_records(metadata, component)
        summary[component] = {
            "repo": spec["repo"], "revision": spec["revision"],
            "license": spec["license"], "official_api": url,
            "metadata": file_identity(path), "records": records,
            "required_bytes": sum(item["size"] for item in records),
        }
    return summary


def download_file(root: Path, component: str, record: dict, *, endpoint: str) -> dict:
    """Resume owned partial bytes and publish only after the complete digest gate."""
    if endpoint not in TRANSPORT_ENDPOINTS:
        raise ValueError("unsupported transport endpoint")
    spec = RDT_ASSETS[component]
    target = _target(root / component, record["rfilename"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return {"reused_verified": True, "file": verify_file(target, record)}
    partial = target.with_name(target.name + ".vf-part")
    owner = target.with_name(target.name + ".vf-part.json")
    identity = {"repo": spec["repo"], "revision": spec["revision"], "record": record}
    if partial.is_symlink() or owner.is_symlink():
        raise ValueError("partial/owner symbolic links are not accepted")
    if owner.exists() and json.loads(owner.read_text()) != identity:
        raise ValueError("partial-file provenance differs from pinned asset")
    if partial.exists() and not owner.exists():
        raise ValueError("unowned partial file will not be overwritten")
    if not owner.exists():
        with owner.open("x") as stream:
            json.dump(identity, stream, sort_keys=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > record["size"]:
        raise ValueError("partial file exceeds the pinned asset size")
    started = time.monotonic()
    if offset < record["size"]:
        url = f"{endpoint}/{spec['repo']}/resolve/{spec['revision']}/{quote(record['rfilename'], safe='/')}"
        # A unique query avoids stale signed redirects, not content version pinning.
        url += f"?download=true&vf_offset={offset}&vf_time={time.time_ns()}"
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with urlopen(Request(url, headers=headers), timeout=90) as response:
            if offset:
                expected = f"bytes {offset}-{record['size'] - 1}/{record['size']}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("resume requires the exact remaining Content-Range")
            elif response.status != 200:
                raise ValueError("initial download requires a complete HTTP 200 response")
            if int(response.headers.get("Content-Length", -1)) != record["size"] - offset:
                raise ValueError("HTTP content length differs from pinned remaining size")
            with partial.open("ab" if offset else "xb") as stream:
                downloaded = offset
                last_progress = time.monotonic()
                while chunk := response.read(8 * 1024**2):
                    downloaded += len(chunk)
                    if downloaded > record["size"]:
                        raise ValueError("download exceeds expected size")
                    stream.write(chunk)
                    if time.monotonic() - last_progress >= 15:
                        print(json.dumps({
                            "event": "progress", "component": component,
                            "file": record["rfilename"], "bytes": downloaded,
                            "size": record["size"], "elapsed_s": time.monotonic() - started,
                        }), flush=True)
                        last_progress = time.monotonic()
                stream.flush()
                os.fsync(stream.fileno())
    actual = verify_file(partial, record)
    os.link(partial, target)
    partial.unlink()
    return {
        "reused_verified": False, "file": actual, "resumed_bytes": offset,
        "transport_endpoint": endpoint, "elapsed_s": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("freeze", "small", "complete", "verify"), required=True)
    parser.add_argument("--endpoint", choices=TRANSPORT_ENDPOINTS, default=TRANSPORT_ENDPOINTS[0])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers not in (1, 2, 3):
        parser.error("workers must be 1, 2 or 3")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.mode == "freeze":
        report = freeze_metadata(args.root)
    elif args.mode == "verify":
        report = verify_assets(args.root)
    else:
        # Download commands are offline with respect to identity selection.
        work = []
        for component in RDT_ASSETS:
            metadata = json.loads((args.root / "metadata" / f"{component}.json").read_text())
            work.extend((component, record) for record in selected_records(metadata, component)
                        if args.mode == "complete" or record["size"] < 2 * 1024**2)
        needed = sum(record["size"] for _, record in work)
        if shutil.disk_usage(args.root).free < needed + 1024**3:
            raise RuntimeError("insufficient free space for the full selected assets")
        print(json.dumps({"event": "start", "pid": os.getpid(), "bytes": needed}), flush=True)
        def download(item):
            component, record = item
            result = download_file(args.root, component, record, endpoint=args.endpoint)
            print(json.dumps({"event": "verified", "component": component,
                              "name": record["rfilename"], **result}), flush=True)
            return {"component": component, "name": record["rfilename"], **result}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            report = list(pool.map(download, work))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
