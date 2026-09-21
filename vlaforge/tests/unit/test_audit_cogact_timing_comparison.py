import csv
import hashlib
import importlib.util
import json
from pathlib import Path


PATH = Path(__file__).resolve().parents[2] / "tools/audit_cogact_timing_comparison.py"
SPEC = importlib.util.spec_from_file_location("audit_cogact_timing_comparison", PATH)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n")


def cdf(path: Path, values: list[int]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("latency_ns", "cdf"))
        writer.writerows((value, (index + 1) / len(values))
                         for index, value in enumerate(sorted(values)))


def campaign(tmp_path: Path, name: str, boundary: str, values: list[int]) -> Path:
    root = tmp_path / name
    root.mkdir()
    protocol = {
        "processes": 5, "warmup": 128, "measured": 1024,
        "seed_schedule": [42, 43, 42],
        "timing_boundary": boundary,
        "gpus": ["0", "1", "2", "3", "4"],
        "gpu_uuids": ["GPU-0", "GPU-1", "GPU-2", "GPU-3", "GPU-4"],
        "input_root": str(tmp_path / "inputs"),
        "expected_root": str(tmp_path / "expected"),
        "runner_sha256": "a" * 64,
        "benchmark_script_sha256": "b" * 64,
    }
    if name == "official":
        protocol["data"] = protocol.pop("input_root")
        protocol["expected"] = protocol.pop("expected_root")
        write(root / "protocol.json", protocol)
        report = {
            "protocol_sha256": sha(root / "protocol.json"),
            "boundary": boundary,
            "all_outputs_exact": True,
            "workers": [{"mean_ns": float(value + 2)}
                        for value in range(5)],
        }
    else:
        report = {
            "protocol": protocol,
            "all_outputs_exact": True,
            "autonomous_cpp_rng": True,
            "processes": [{"latency_ns": {"mean": float(value + 1)}}
                          for value in range(5)],
        }
    write(root / "report.json", report)
    cdf(root / "latency-cdf.csv", values)
    return root


def test_timing_comparison_requires_identical_data_and_seed_contract(tmp_path, monkeypatch):
    native_root = campaign(tmp_path, "native", tool.NATIVE_BOUNDARY, list(range(100, 5220)))
    official_root = campaign(tmp_path, "official", tool.OFFICIAL_BOUNDARY, list(range(200, 5320)))
    native_audit = tmp_path / "native-audit.json"
    official_audit = tmp_path / "official-audit.json"
    write(native_audit, {
        "status": "passed",
        "campaign_report_sha256": sha(native_root / "report.json"),
    })
    write(official_audit, {
        "status": "passed",
        "campaign_report_sha256": sha(official_root / "report.json"),
    })
    output = tmp_path / "comparison.json"
    import sys

    monkeypatch.setattr(tool, "bootstrap_interval", lambda *_: (1.5, 1.6))
    sys.argv = [
        "audit", "--native-campaign", str(native_root), "--official-campaign", str(official_root),
        "--native-audit", str(native_audit), "--official-audit", str(official_audit),
        "--output", str(output),
    ]
    tool.main()
    result = json.loads(output.read_text())
    assert result["status"] == "passed"
    assert result["same_inputs"] is True
    assert result["same_seed_schedule"] == [42, 43, 42]
    assert result["same_boundary"] is True
