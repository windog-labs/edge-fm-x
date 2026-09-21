"""Pinned offline processor assets for the shared OpenPI Adapter."""

from __future__ import annotations

import argparse
from http.client import HTTPException
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.adapters.openpi.openpi_download import GCS_ENDPOINTS


PALIGEMMA_TOKENIZER_OBJECT = {
    "name": "paligemma_tokenizer.model",
    "generation": "1711547605575873",
    "size": "4264023",
    "md5Hash": "FCCtyYVnIKVZ6KhyhLGV4g==",
    "crc32c": "cKDEzw==",
}
PALIGEMMA_TOKENIZER_SHA256 = (
    "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
)


def _tokenizer_path(cache_root: str | Path | None) -> Path:
    root = cache_root or os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")
    root = Path(root).expanduser().resolve()
    path = root / "big_vision/paligemma_tokenizer.model"
    if not path.resolve().is_relative_to(root):
        raise ValueError("tokenizer path escapes the selected cache directory")
    return path


def verify_openpi_tokenizer(cache_root: str | Path | None = None) -> dict:
    """Fail before upstream tokenization can download unpinned processor bytes."""
    path = _tokenizer_path(cache_root)
    actual = file_digest(path)
    if (
        actual["size_bytes"] != int(PALIGEMMA_TOKENIZER_OBJECT["size"])
        or actual["md5_base64"] != PALIGEMMA_TOKENIZER_OBJECT["md5Hash"]
        or actual["sha256"] != PALIGEMMA_TOKENIZER_SHA256
    ):
        raise ValueError(f"OpenPI tokenizer digest mismatch: {path}")
    return {
        "schema": "vlaforge.openpi_processor/1",
        "source": "gs://big_vision/paligemma_tokenizer.model",
        "object": dict(PALIGEMMA_TOKENIZER_OBJECT),
        "path": str(path),
        "file": actual,
    }


def download_openpi_tokenizer(
    *,
    inventory_path: str | Path,
    cache_root: str | Path,
    endpoint: str = GCS_ENDPOINTS[0],
    attempts: int = 8,
) -> dict:
    """Download the small shared tokenizer, preserving every unknown existing file."""
    if endpoint not in GCS_ENDPOINTS or attempts < 1:
        raise ValueError("an official GCS endpoint and positive attempts are required")
    inventory = Path(inventory_path)
    if json.loads(inventory.read_text()) != PALIGEMMA_TOKENIZER_OBJECT:
        raise ValueError("tokenizer inventory differs from the frozen official object")
    target = _tokenizer_path(cache_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        report = verify_openpi_tokenizer(cache_root)
        return {**report, "inventory": file_digest(inventory), "reused_verified": True}
    query = urlencode({"generation": PALIGEMMA_TOKENIZER_OBJECT["generation"]})
    url = f"https://{endpoint}/big_vision/paligemma_tokenizer.model?{query}"
    for attempt in range(1, attempts + 1):
        partial = None
        try:
            with urlopen(url, timeout=90) as response:
                if (
                    response.status != 200
                    or response.headers.get("x-goog-generation")
                    != PALIGEMMA_TOKENIZER_OBJECT["generation"]
                ):
                    raise ValueError(
                        "tokenizer HTTP status or object generation mismatch"
                    )
                with tempfile.NamedTemporaryFile(
                    prefix="paligemma_tokenizer.vf-part-",
                    dir=target.parent,
                    delete=False,
                ) as stream:
                    partial = Path(stream.name)
                    size = 0
                    while chunk := response.read(1024**2):
                        stream.write(chunk)
                        size += len(chunk)
                        if size > int(PALIGEMMA_TOKENIZER_OBJECT["size"]):
                            raise ValueError("tokenizer download exceeded frozen size")
                    stream.flush()
                    os.fsync(stream.fileno())
            actual = file_digest(partial)
            if (
                actual["size_bytes"] != int(PALIGEMMA_TOKENIZER_OBJECT["size"])
                or actual["md5_base64"] != PALIGEMMA_TOKENIZER_OBJECT["md5Hash"]
                or actual["sha256"] != PALIGEMMA_TOKENIZER_SHA256
            ):
                raise ValueError("tokenizer download checksum mismatch")
            os.link(partial, target)
            partial.unlink()
            report = verify_openpi_tokenizer(cache_root)
            return {
                **report,
                "inventory": file_digest(inventory),
                "reused_verified": False,
            }
        except FileExistsError:
            raise
        except (HTTPException, OSError, ValueError) as error:
            print(
                json.dumps(
                    {
                        "event": "tokenizer_retry",
                        "pid": os.getpid(),
                        "attempt": attempt,
                        "error": str(error),
                        "preserved_partial": str(partial) if partial else None,
                    }
                ),
                flush=True,
            )
            if (
                attempt == attempts
                or isinstance(error, HTTPError)
                and error.code in (401, 403, 404)
            ):
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-path", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--endpoint", choices=GCS_ENDPOINTS, default=GCS_ENDPOINTS[0])
    parser.add_argument("--attempts", type=int, default=8)
    print(json.dumps(download_openpi_tokenizer(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
