"""Downloader transport tests use bytes, not a substitute model/checkpoint."""

import base64
import hashlib
import io
import json
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from vlaforge.adapters.openpi import openpi_download as download


def _fixture(tmp_path):
    payloads = {
        "params/part": b"deterministic bytes for transport test",
        "assets/trossen/norm_stats.json": b'{"mean": [0]}',
    }
    items = [
        {
            "name": f"checkpoints/pi0_base/{name}",
            "size": str(len(value)),
            "generation": "123456",
            "md5Hash": base64.b64encode(hashlib.md5(value).digest()).decode(),
        }
        for name, value in payloads.items()
    ]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"items": items}))
    return (
        payloads,
        items,
        dict(
            inventory_path=inventory,
            destination=tmp_path / "weights",
            checkpoint_name="pi0_base",
            log_file=tmp_path / "download.jsonl",
        ),
    )


class Response(io.BytesIO):
    def __init__(self, data, *, offset=0, original_size=None):
        super().__init__(data)
        self.status = 206 if offset else 200
        self.headers = {"x-goog-generation": "123456"}
        if offset:
            self.headers["Content-Range"] = (
                f"bytes {offset}-{original_size - 1}/{original_size}"
            )


def _transport(payloads, calls, endpoint="storage.googleapis.com", api="xml"):
    def fetch(request, timeout):
        url = urlsplit(request.full_url)
        assert url.scheme == "https" and url.hostname == endpoint
        expected = {"generation": ["123456"]}
        prefix = "/openpi-assets/checkpoints/pi0_base/"
        if api == "json":
            expected["alt"] = ["media"]
            prefix = "/download/storage/v1/b/openpi-assets/o/checkpoints/pi0_base/"
        assert parse_qs(url.query) == expected
        name = unquote(url.path).removeprefix(prefix)
        offset = (
            int(request.get_header("Range")[6:-1]) if request.has_header("Range") else 0
        )
        calls.append((name, offset))
        body = payloads[name]
        return Response(body[offset:], offset=offset, original_size=len(body))

    return fetch


def test_download_verifies_every_object_and_reuses_only_verified_files(
    tmp_path, monkeypatch
):
    payloads, _, config = _fixture(tmp_path)
    calls = []
    monkeypatch.setattr(download, "urlopen", _transport(payloads, calls))
    result = download.download_checkpoint(**config)
    assert len(result["objects"]) == 2
    assert len(calls) == 2
    again = download.download_checkpoint(**config)
    assert again == result
    assert len(calls) == 2
    assert (config["destination"] / "vlaforge_download.json").is_file()


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("api", ["xml", "json"])
def test_owned_partial_resumes_with_frozen_generation_and_byte_range(
    tmp_path, monkeypatch, composite, api
):
    payloads, items, config = (_composite_fixture if composite else _fixture)(tmp_path)
    partial = config["destination"] / "params/part.vf-part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payloads["params/part"][:5])
    partial.with_name("part.vf-part.json").write_text(json.dumps(items[0]))
    calls = []
    monkeypatch.setattr(download, "urlopen", _transport(payloads, calls, api=api))
    download.download_checkpoint(**config, api=api)
    assert calls[0] == ("params/part", 5)
    assert not partial.exists()
    assert (config["destination"] / "params/part").read_bytes() == payloads[
        "params/part"
    ]


@pytest.mark.parametrize("partial", [False, True])
def test_unknown_existing_data_is_preserved_without_network_or_overwrite(
    tmp_path, monkeypatch, partial
):
    _, _, config = _fixture(tmp_path)
    target = config["destination"] / (
        "params/part.vf-part" if partial else "params/part"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"unverified existing user data")
    monkeypatch.setattr(
        download,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("network must not be used"),
    )
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        download.download_checkpoint(**config)
    assert target.read_bytes() == b"unverified existing user data"


def test_checksum_failure_preserves_rejected_bytes_then_retries(tmp_path, monkeypatch):
    payloads, _, config = _fixture(tmp_path)
    calls = []
    normal = _transport(payloads, calls)
    failed = False

    def fetch(request, timeout):
        nonlocal failed
        if not failed:
            failed = True
            return Response(b"X" * len(payloads["params/part"]))
        return normal(request, timeout)

    monkeypatch.setattr(download, "urlopen", fetch)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    download.download_checkpoint(**config)
    rejected = tuple((config["destination"] / "params").glob("*.rejected-*"))
    assert len(rejected) == 1
    assert rejected[0].read_bytes() == b"X" * len(payloads["params/part"])


def test_generation_mismatch_is_not_accepted(tmp_path, monkeypatch):
    payloads, _, config = _fixture(tmp_path)

    def fetch(*_, **__):
        response = Response(payloads["params/part"])
        response.headers["x-goog-generation"] = "654321"
        return response

    monkeypatch.setattr(download, "urlopen", fetch)
    with pytest.raises(ValueError, match="generation"):
        download.download_checkpoint(**config, attempts=1)
    assert not (config["destination"] / "params/part").exists()


def test_official_alternate_endpoint_keeps_generation_and_checksum(
    tmp_path, monkeypatch
):
    payloads, _, config = _fixture(tmp_path)
    endpoint = "storage-download.googleapis.com"
    monkeypatch.setattr(download, "urlopen", _transport(payloads, [], endpoint))
    assert (
        len(download.download_checkpoint(**config, endpoint=endpoint)["objects"]) == 2
    )


def test_non_official_endpoint_is_rejected_before_io(tmp_path):
    _, _, config = _fixture(tmp_path)
    with pytest.raises(ValueError, match="official GCS"):
        download.download_checkpoint(**config, endpoint="mirror.example.com")
    assert not config["destination"].exists()


@pytest.mark.parametrize(
    "api,endpoint",
    [
        ("unknown", "storage.googleapis.com"),
        ("json", "storage-download.googleapis.com"),
    ],
)
def test_invalid_media_route_is_rejected_before_io(tmp_path, api, endpoint):
    _, _, config = _fixture(tmp_path)
    with pytest.raises(ValueError, match="media API"):
        download.download_checkpoint(**config, api=api, endpoint=endpoint)
    assert not config["destination"].exists()


def test_symbolic_link_partial_is_preserved(tmp_path):
    _, items, config = _fixture(tmp_path)
    outside = tmp_path / "existing-data"
    outside.write_bytes(b"preserve")
    partial = config["destination"] / "params/part.vf-part"
    partial.parent.mkdir(parents=True)
    partial.symlink_to(outside)
    partial.with_name("part.vf-part.json").write_text(json.dumps(items[0]))
    with pytest.raises(FileExistsError, match="symbolic-link"):
        download.download_checkpoint(**config)
    assert outside.read_bytes() == b"preserve"


def _composite_fixture(tmp_path):
    crc32c = pytest.importorskip("google_crc32c")
    payloads, items, config = _fixture(tmp_path)
    items[0].pop("md5Hash")
    items[0]["componentCount"] = 32
    items[0]["crc32c"] = base64.b64encode(
        crc32c.Checksum(payloads["params/part"]).digest()
    ).decode()
    config["inventory_path"].write_text(json.dumps({"items": items}))
    return payloads, items, config


def test_composite_uses_official_crc_and_never_reports_md5_verified(
    tmp_path, monkeypatch
):
    payloads, _, config = _composite_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(download, "urlopen", _transport(payloads, calls))
    report = download.download_checkpoint(**config)
    composite = report["objects"][0]
    assert composite["checksum_type"] == "crc32c"
    assert composite["md5_verified"] is False
    assert composite["crc32c_verified"] is True
    assert (
        composite["local"]["sha256"]
        == hashlib.sha256(payloads["params/part"]).hexdigest()
    )
    assert download.download_checkpoint(**config) == report
    assert len(calls) == 2


@pytest.mark.parametrize(
    "failure",
    [
        "missing_count",
        "bad_count",
        "mixed_md5",
        "missing_crc",
        "bad_crc",
        "wrong_size",
        "wrong_generation",
        "missing_generation_header",
    ],
)
def test_composite_negative_gates_preserve_incomplete_status(
    tmp_path, monkeypatch, failure
):
    payloads, items, config = _composite_fixture(tmp_path)
    item = items[0]
    if failure == "missing_count":
        item.pop("componentCount")
    elif failure == "bad_count":
        item["componentCount"] = None
    elif failure == "mixed_md5":
        item["md5Hash"] = base64.b64encode(
            hashlib.md5(payloads["params/part"]).digest()
        ).decode()
    elif failure == "missing_crc":
        item.pop("crc32c")
    elif failure == "bad_crc":
        item["crc32c"] = base64.b64encode(b"\0" * 4).decode()
    elif failure == "wrong_size":
        item["size"] = str(int(item["size"]) + 1)
    config["inventory_path"].write_text(json.dumps({"items": items}))

    def fetch(*_, **__):
        response = Response(payloads["params/part"])
        if failure == "wrong_generation":
            response.headers["x-goog-generation"] = "654321"
        elif failure == "missing_generation_header":
            response.headers.pop("x-goog-generation")
        return response

    monkeypatch.setattr(download, "urlopen", fetch)
    with pytest.raises(ValueError):
        download.download_checkpoint(**config, attempts=1)
    assert not (config["destination"] / "params/part").exists()
    assert not (config["destination"] / "vlaforge_download.json").exists()
