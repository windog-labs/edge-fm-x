"""Identity transport fixtures; these bytes are not executable libraries."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def tool():
    spec = importlib.util.spec_from_file_location("runtime_library_benchmark", Path(__file__).resolve().parents[2] / "tools/benchmark_session.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared(tmp_path):
    benchmark = tool()
    build, bundle, source, run = [tmp_path / name for name in ("build", "bundle", "source", "run")]
    for path in (build, bundle, source, run):
        path.mkdir()
    library = build / "libvlaforge_libtorch_numerical_backend.so"
    library.write_bytes(b"fixture-current-build")
    delivered = bundle / library.name
    delivered.write_bytes(b"fixture-delivered-build")
    records = benchmark.runtime_library_records(build, bundle, SimpleNamespace(binaries=(SimpleNamespace(path=library.name, role="runtime_shared_library"),)))
    (source / "runtime-libraries.json").write_text(json.dumps({"off": records}))
    (tmp_path / "prepared.json").write_text(json.dumps({"numerical_worker_initialization": {"off": {"configured": True}}}))
    (run / "process-maps.txt").write_text(f"1000-2000 r-xp 0000 00:01 1 {library}\n")
    return benchmark, library, run, records


def test_collects_both_actual_build_and_bundle_copy(tmp_path):
    benchmark, library, run, records = prepared(tmp_path)
    assert len(records) == 2 and len({item["sha256"] for item in records}) == 2
    result = benchmark.verify_runtime_library_execution(tmp_path, run, "off")
    assert result["verified"] and result["actual_libraries"] == [next(item for item in records if item["path"] == str(library))]


@pytest.mark.parametrize("change", ["bytes", "unbound", "missing"])
def test_actual_mapping_or_content_drift_fails_closed(tmp_path, change):
    benchmark, library, run, _ = prepared(tmp_path)
    if change == "bytes":
        library.write_bytes(b"substituted")
    elif change == "unbound":
        (run / "process-maps.txt").write_text("1000-2000 r-xp 0000 00:01 1 /unbound/libvlaforge_libtorch_numerical_backend.so\n")
    else:
        (run / "process-maps.txt").write_text("")
    with pytest.raises(ValueError):
        benchmark.verify_runtime_library_execution(tmp_path, run, "off")


def test_legacy_missing_inventory_is_not_promoted(tmp_path):
    result = tool().verify_runtime_library_execution(tmp_path, tmp_path / "run", "off")
    assert result["status"] == "not-recorded-legacy" and not result["verified"]
