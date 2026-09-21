"""Streaming, generation-pinned GCS checkpoint download with verified resume."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from http.client import HTTPException
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from vlaforge.adapters.openpi.openpi_checkpoint import (
    file_digest,
    gcs_checksum_type,
    verify_gcs_checkpoint,
    verify_gcs_object,
)

GCS_ENDPOINTS = ("storage.googleapis.com", "storage-download.googleapis.com")


def inventory_objects(inventory: dict, checkpoint_name: str):
    if inventory.get("nextPageToken") or not inventory.get("items"):
        raise ValueError("inventory must contain every checkpoint object")
    prefix = f"checkpoints/{checkpoint_name}/"
    seen = set()
    for item in inventory["items"]:
        name = item["name"]
        if not name.startswith(prefix):
            raise ValueError(f"object outside selected checkpoint: {name}")
        relative = name.removeprefix(prefix)
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != relative
            or not path.parts
        ):
            raise ValueError(f"unsafe object name: {name}")
        if relative in seen:
            raise ValueError(f"duplicate object name: {name}")
        seen.add(relative)
        gcs_checksum_type(item)
        yield path, item


def _matches(path: Path, item: dict) -> bool:
    if not path.is_file() or path.stat().st_size != int(item["size"]):
        return False
    try:
        verify_gcs_object(path, item)
    except ValueError:
        return False
    return True


def download_checkpoint(
    *,
    inventory_path: str | Path,
    destination: str | Path,
    checkpoint_name: str,
    log_file: str | Path,
    attempts: int = 8,
    endpoint: str = GCS_ENDPOINTS[0],
    api: str = "xml",
) -> dict[str, object]:
    if attempts < 1:
        raise ValueError("download attempts must be positive")
    if endpoint not in GCS_ENDPOINTS:
        raise ValueError("download endpoint must be an official GCS endpoint")
    if api not in ("xml", "json") or (api == "json" and endpoint != GCS_ENDPOINTS[0]):
        raise ValueError(
            "select XML or the official storage.googleapis.com JSON media API"
        )
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    inventory_path = Path(inventory_path)
    inventory = json.loads(inventory_path.read_text())
    objects = tuple(inventory_objects(inventory, checkpoint_name))
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        (root / ".vlaforge-download.lock").open("a") as lock,
        log_path.open("a", buffering=1) as log,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def emit(event, **fields):
            record = {
                "time": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "event": event,
                **fields,
            }
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            log.write(line + "\n")

        total = sum(int(item["size"]) for _, item in objects)
        if shutil.disk_usage(root).free < total + 2 * 1024**3:
            raise RuntimeError("insufficient disk for checkpoint plus safety reserve")
        emit(
            "start",
            destination=str(root),
            inventory=file_digest(inventory_path),
            objects=len(objects),
            bytes=total,
            endpoint=endpoint,
            api=api,
        )
        try:
            for relative, item in objects:
                target = root / str(relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.resolve().is_relative_to(root):
                    raise ValueError(f"object escapes destination: {target}")
                if target.exists():
                    if not _matches(target, item):
                        raise FileExistsError(
                            f"existing file failed verification; refusing overwrite: {target}"
                        )
                    emit(
                        "verified_existing",
                        object=item["name"],
                        bytes=int(item["size"]),
                        checksum_type=gcs_checksum_type(item),
                        md5_verified=gcs_checksum_type(item) == "md5",
                    )
                    continue
                partial = target.with_name(target.name + ".vf-part")
                owner = target.with_name(target.name + ".vf-part.json")
                if partial.is_symlink() or owner.is_symlink():
                    raise FileExistsError("refusing symbolic-link partial or owner")
                if partial.exists() and (
                    not owner.is_file() or json.loads(owner.read_text()) != item
                ):
                    raise FileExistsError(
                        f"unowned partial file; refusing overwrite: {partial}"
                    )
                if owner.exists() and json.loads(owner.read_text()) != item:
                    raise FileExistsError(f"partial file provenance mismatch: {owner}")
                if not owner.exists():
                    with owner.open("x") as stream:
                        json.dump(item, stream, sort_keys=True)
                if api == "json":
                    url = (
                        f"https://{endpoint}/download/storage/v1/b/openpi-assets/o/"
                        + quote(item["name"], safe="")
                    )
                    query = {"generation": item["generation"], "alt": "media"}
                else:
                    url = f"https://{endpoint}/openpi-assets/" + quote(
                        item["name"], safe="/"
                    )
                    query = {"generation": item["generation"]}
                url += "?" + urlencode(query)
                for attempt in range(1, attempts + 1):
                    try:
                        offset = partial.stat().st_size if partial.exists() else 0
                        if offset > int(item["size"]):
                            raise ValueError(
                                f"partial file exceeds expected size: {partial}"
                            )
                        if offset == int(item["size"]) and _matches(partial, item):
                            break
                        headers = {"Range": f"bytes={offset}-"} if offset else {}
                        emit(
                            "object_start",
                            object=item["name"],
                            attempt=attempt,
                            offset=offset,
                            size=int(item["size"]),
                        )
                        with urlopen(
                            Request(url, headers=headers), timeout=90
                        ) as response:
                            generation = response.headers.get("x-goog-generation")
                            if generation != str(item["generation"]):
                                raise ValueError(
                                    "GCS returned another object generation"
                                )
                            if offset and response.status == 206:
                                if not response.headers.get(
                                    "Content-Range", ""
                                ).startswith(f"bytes {offset}-"):
                                    raise ValueError(
                                        "HTTP partial response has an unexpected range"
                                    )
                            elif response.status == 200:
                                offset = 0
                            else:
                                raise ValueError(
                                    f"unexpected download response: {response.status}"
                                )
                            mode = "ab" if offset else "wb"
                            with partial.open(mode) as stream:
                                progress = offset
                                last_report = time.monotonic()
                                while chunk := response.read(4 * 1024**2):
                                    stream.write(chunk)
                                    progress += len(chunk)
                                    if progress > int(item["size"]):
                                        raise ValueError(
                                            "download exceeded frozen object size"
                                        )
                                    if time.monotonic() - last_report >= 30:
                                        stream.flush()
                                        emit(
                                            "progress",
                                            object=item["name"],
                                            bytes=progress,
                                            size=int(item["size"]),
                                        )
                                        last_report = time.monotonic()
                                stream.flush()
                                os.fsync(stream.fileno())
                        if not _matches(partial, item):
                            raise ValueError(
                                f"download checksum mismatch: {item['name']}"
                            )
                        break
                    except (HTTPError, HTTPException, OSError, ValueError) as error:
                        emit(
                            "object_retry",
                            object=item["name"],
                            attempt=attempt,
                            error=str(error),
                        )
                        if isinstance(error, HTTPError) and error.code in (
                            401,
                            403,
                            404,
                        ):
                            raise
                        if attempt == attempts:
                            raise
                        # A complete but corrupt owned partial must restart, not resume
                        # from EOF; preserve the bad bytes for failure inspection.
                        if partial.exists() and partial.stat().st_size >= int(
                            item["size"]
                        ):
                            rejected = partial.with_name(
                                partial.name + f".rejected-{time.time_ns()}"
                            )
                            partial.rename(rejected)
                        time.sleep(min(2**attempt, 30))
                os.link(partial, target)
                partial.unlink()
                owner.unlink()
                verification = verify_gcs_object(target, item)
                emit(
                    "object_complete",
                    object=item["name"],
                    file=verification["local"],
                    checksum_type=verification["checksum_type"],
                    md5_verified=verification["md5_verified"],
                    crc32c_verified=verification["crc32c_verified"],
                )
            result = verify_gcs_checkpoint(
                root, inventory_path, checkpoint_name=checkpoint_name
            )
            report_path = root / "vlaforge_download.json"
            report_path.write_text(json.dumps(result, indent=2) + "\n")
            emit("complete", report=str(report_path), objects=len(objects), bytes=total)
            return result
        except Exception as error:
            emit("failed", error=f"{type(error).__name__}: {error}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-path", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument(
        "--checkpoint-name", choices=("pi0_base", "pi05_base"), required=True
    )
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--endpoint", choices=GCS_ENDPOINTS, default=GCS_ENDPOINTS[0])
    parser.add_argument("--api", choices=("xml", "json"), default="xml")
    download_checkpoint(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
