"""Processor transport/gate tests, not pretrained action-model evidence."""

import base64
import hashlib
import io
import json

import pytest

from vlaforge.adapters.openpi import openpi_assets as assets


@pytest.fixture
def transport(tmp_path, monkeypatch):
    body = b"processor transport test bytes"
    item = {
        "name": "paligemma_tokenizer.model",
        "generation": "123",
        "size": str(len(body)),
        "md5Hash": base64.b64encode(hashlib.md5(body).digest()).decode(),
        "crc32c": "not-used-by-MD5-gate",
    }
    monkeypatch.setattr(assets, "PALIGEMMA_TOKENIZER_OBJECT", item)
    monkeypatch.setattr(
        assets, "PALIGEMMA_TOKENIZER_SHA256", hashlib.sha256(body).hexdigest()
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(item))
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        response = io.BytesIO(body)
        response.status = 200
        response.headers = {"x-goog-generation": "123"}
        return response

    monkeypatch.setattr(assets, "urlopen", fetch)
    return body, calls, dict(inventory_path=inventory, cache_root=tmp_path / "cache")


def test_processor_download_is_generation_pinned_and_reuses_only_verified(transport):
    body, calls, config = transport
    report = assets.download_openpi_tokenizer(**config)
    assert report["file"]["size_bytes"] == len(body)
    assert report["reused_verified"] is False
    assert calls == [
        "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model?generation=123"
    ]
    assert assets.download_openpi_tokenizer(**config)["reused_verified"] is True
    assert len(calls) == 1


def test_processor_download_preserves_unverified_existing_file(transport):
    _, calls, config = transport
    target = config["cache_root"] / "big_vision/paligemma_tokenizer.model"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing user bytes")
    with pytest.raises(ValueError, match="digest mismatch"):
        assets.download_openpi_tokenizer(**config)
    assert not calls
    assert target.read_bytes() == b"existing user bytes"


def test_processor_offline_gate_uses_process_cache_root(transport, monkeypatch):
    _, _, config = transport
    monkeypatch.setenv("OPENPI_DATA_HOME", str(config["cache_root"]))
    with pytest.raises(FileNotFoundError):
        assets.verify_openpi_tokenizer()
    downloaded = assets.download_openpi_tokenizer(**config)
    assert assets.verify_openpi_tokenizer()["file"] == downloaded["file"]


def test_processor_changed_inventory_is_rejected_before_network(transport):
    _, calls, config = transport
    config["inventory_path"].write_text("{}")
    with pytest.raises(ValueError, match="frozen official"):
        assets.download_openpi_tokenizer(**config)
    assert not calls


def test_processor_cache_cannot_write_through_an_external_symlink(transport, tmp_path):
    _, calls, config = transport
    outside = tmp_path / "outside"
    outside.mkdir()
    config["cache_root"].mkdir()
    (config["cache_root"] / "big_vision").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        assets.download_openpi_tokenizer(**config)
    assert not calls
    assert not tuple(outside.iterdir())
