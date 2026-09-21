"""CPU contract negatives only; real-weight evidence lives outside unit tests."""

import hashlib

import pytest
from vlaforge.adapters.cogact.cogact_real import (
    PUBLIC_LLM_FILES,
    PUBLIC_LLM_REVISION,
    REQUIRED_GROUPS,
    file_sha256,
    require_complete_groups,
    validate_public_llm_lock,
)


@pytest.mark.parametrize("name", REQUIRED_GROUPS)
def test_missing_weight_group_rejects_random_fallback(name):
    state = {group: {"weight": object()} for group in REQUIRED_GROUPS}
    del state[name]
    with pytest.raises(ValueError, match="random fallback forbidden"):
        require_complete_groups(state)


@pytest.mark.parametrize("bad", ({}, None, [], "missing"))
def test_invalid_action_group_rejects_random_fallback(bad):
    state = {group: {"weight": object()} for group in REQUIRED_GROUPS}
    state["action_model"] = bad
    with pytest.raises(ValueError, match="action_model"):
        require_complete_groups(state)


def test_checkpoint_root_must_be_dictionary():
    with pytest.raises(TypeError, match="state dictionary"):
        require_complete_groups([])


def test_complete_structure_is_not_a_weight_verification_claim():
    state = {group: {"unverified-weight": object()} for group in REQUIRED_GROUPS}
    assert require_complete_groups(state) is None


def test_file_identity_covers_all_bytes(tmp_path):
    path = tmp_path / "asset"
    path.write_bytes(b"full asset bytes")
    assert file_sha256(path) == hashlib.sha256(b"full asset bytes").hexdigest()
    before = file_sha256(path)
    path.write_bytes(b"full asset byteS")
    assert file_sha256(path) != before


def public_lock():
    return {"repo": "openvla/openvla-7b", "revision": PUBLIC_LLM_REVISION,
            "status": "verified", "gated": False, "official_metadata": True,
            "files": [{"path": name, "sha256": digest, "revision": PUBLIC_LLM_REVISION, "verified": True}
                      for name, digest in PUBLIC_LLM_FILES.items()]}


def test_complete_fixed_public_dependency_lock():
    validate_public_llm_lock(public_lock())


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "digest", "revision", "unverified", "repo_revision"))
def test_public_dependency_lock_cannot_be_downgraded(mutation):
    lock = public_lock()
    if mutation == "missing":
        lock["files"].pop()
    elif mutation == "duplicate":
        lock["files"][-1] = lock["files"][0]
    elif mutation == "digest":
        lock["files"][0]["sha256"] = "0" * 64
    elif mutation == "revision":
        lock["files"][0]["revision"] = "0" * 40
    elif mutation == "unverified":
        lock["files"][0]["verified"] = False
    else:
        lock["revision"] = "main"
    with pytest.raises(ValueError):
        validate_public_llm_lock(lock)
