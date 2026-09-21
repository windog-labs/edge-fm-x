"""Synthetic evidence-boundary negatives, not real model deployment claims."""

import pytest

from vlaforge.adapters.openpi.openpi_native_join import (
    SCHEMA, checked, record, validate_complete_bytes, validate_typed_native_join,
)


def test_complete_tensor_bytes_preserve_every_call_and_signed_zero():
    value = b"\0\0\0\x80" + b"\0\0\0\0"
    validate_complete_bytes(value * 4, value, value, size=8, calls=4)
    with pytest.raises(ValueError, match="bytes differ"):
        validate_complete_bytes(b"\0" * 32, value, value, size=8, calls=4)


@pytest.mark.parametrize("size,calls", [(True, 2), (2, True), (0, 1), (1, 0), (-1, 1), (1, -1), (1.0, 1)])
def test_rejects_ambiguous_counts(size, calls):
    with pytest.raises(ValueError, match="positive integers"):
        validate_complete_bytes(b"aa", b"a", b"a", size=size, calls=calls)


@pytest.mark.parametrize("raw,direct,eager", [(b"aaa", b"a", b"a"), (b"a", b"a", b"a"),
    (b"aa", b"aa", b"a"), (b"aa", b"a", b"aa"), (b"ab", b"a", b"a"), (b"aa", b"b", b"a")])
def test_rejects_missing_extra_or_changed_complete_outputs(raw, direct, eager):
    with pytest.raises(ValueError):
        validate_complete_bytes(raw, direct, eager, size=1, calls=2)


@pytest.mark.parametrize("value", [{}, {"schema": SCHEMA}, {"schema": "vlaforge.openpi_typed_native_join/99"}])
def test_rejects_unknown_or_incomplete_join(value):
    with pytest.raises(ValueError, match="schema"):
        validate_typed_native_join(value)


def test_source_identity_has_no_silent_path_or_hash_rebinding(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"status":"built-unexecuted"}')
    reference = record(path)
    assert checked(reference)[1]["status"] == "built-unexecuted"
    path.write_text('{"status":"passed"}')
    with pytest.raises(ValueError, match="identity changed"):
        checked(reference)


def test_rejects_ambiguous_extra_record_metadata(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="identity changed"):
        checked({**record(path), "validated": True})
