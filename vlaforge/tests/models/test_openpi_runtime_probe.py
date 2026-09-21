"""Diagnostic observer tests only, not checkpoint or CUDA execution evidence."""

from types import SimpleNamespace

import pytest

from vlaforge.adapters.openpi.openpi_runtime_probe import record_matmul_setters


def test_matmul_observer_retains_call_order_and_restores_callable():
    received = []
    original = received.append
    fake = SimpleNamespace(set_float32_matmul_precision=original)
    events = []
    with record_matmul_setters(fake, events):
        fake.set_float32_matmul_precision("high")
        fake.set_float32_matmul_precision("highest")
    assert received == ["high", "highest"]
    assert fake.set_float32_matmul_precision is original
    assert [item["value"] for item in events] == received
    assert all(item["returned"] for item in events)


def test_matmul_observer_retains_failure_and_restores_callable():
    def original(value):
        raise ValueError(value)

    fake = SimpleNamespace(set_float32_matmul_precision=original)
    events = []
    with pytest.raises(ValueError, match="bad"):
        with record_matmul_setters(fake, events):
            fake.set_float32_matmul_precision("bad")
    assert fake.set_float32_matmul_precision is original
    assert events == [{"setter": "torch.set_float32_matmul_precision", "value": "bad"}]
