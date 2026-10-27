import copy

import pytest

from vlaforge.validation.tensorrt_build import require_successful_trt_build


def receipt():
    return {"schema": "edgefm.orin_trt_build/1", "status": "built-not-validated", "exit_code": 0,
            "source_sha256": "1" * 64, "engine": {"path": "/task/model.engine", "bytes": 123, "sha256": "2" * 64}}


def test_receipt_validation_does_not_claim_execution_or_require_host_engine_file():
    value = receipt()
    before = copy.deepcopy(value)
    require_successful_trt_build(value, source_sha256="1" * 64)
    assert value == before


def test_successful_collector_exit_cannot_hide_failed_child_build():
    value = receipt()
    value.update(status="compile-failed", exit_code=1, collector_exit_code=0)
    value.pop("engine")
    with pytest.raises(ValueError, match="child build"):
        require_successful_trt_build(value)


@pytest.mark.parametrize("change", ["status", "child-exit", "bool-exit", "missing-engine", "empty-engine", "bool-size", "hash", "source"])
def test_partial_or_inconsistent_build_metadata_is_not_success(change):
    value = receipt()
    if change == "status":
        value["status"] = "running"
    elif change == "child-exit":
        value["exit_code"] = 1
    elif change == "bool-exit":
        value["exit_code"] = False
    elif change == "missing-engine":
        value.pop("engine")
    elif change == "empty-engine":
        value["engine"]["bytes"] = 0
    elif change == "bool-size":
        value["engine"]["bytes"] = True
    elif change == "hash":
        value["engine"]["sha256"] = "bad"
    else:
        value["source_sha256"] = "3" * 64
    with pytest.raises(ValueError):
        require_successful_trt_build(value, source_sha256="1" * 64)
