import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2] / "tools" / "benchmark_horizon_hbm.py"
SPEC = importlib.util.spec_from_file_location("benchmark_horizon_hbm", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HorizonCampaignTest(unittest.TestCase):
    def test_registered_other_client_blocks_even_when_ratio_is_zero(self):
        def sysfs(path):
            if str(path).endswith("/ratio"):
                return "0"
            return "*User via BPU Bus*\nuser\t\tratio\n1160177\t\t0\n"

        no_hrt = MODULE.subprocess.CompletedProcess(["pgrep"], 1)
        with patch.object(MODULE, "read_text", side_effect=sysfs), \
                patch.object(MODULE.subprocess, "run", return_value=no_hrt):
            with self.assertRaisesRegex(RuntimeError, "BPU clients"):
                MODULE.require_idle()

    def test_rejects_mismatched_model_and_missing_input(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.hbm"
            model.write_bytes(b"fixture")
            stage = {"name": "stage", "hbm": str(model), "sha256": "wrong",
                     "inputs": [str(Path(directory) / "missing.bin")]}
            manifest = {"schema": "vlaforge.hbm_performance_campaign/1", "stages": [stage]}
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                MODULE.validate_manifest(manifest)
            stage["sha256"] = MODULE.sha256(model)
            with self.assertRaisesRegex(ValueError, "missing or empty input"):
                MODULE.validate_manifest(manifest)

    def test_child_evidence_retains_peak_rss_and_actual_records(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(MODULE, "snapshot", return_value={}):
            root = Path(directory) / "run"
            record = MODULE.observe([sys.executable, "-c",
                "x=bytearray(16*1024*1024); print('Load model to DDR cost 12.5ms.'); "
                "print('Infer time: 0.25 ms'); print('Infer time: 0.5 ms')"], root)
            self.assertEqual(record["exit_code"], 0)
            self.assertEqual(record["infer_records"], 2)
            self.assertEqual(record["model_load_ms"], [12.5])
            self.assertGreater(record["peak_rss_mib"], 16)
            self.assertFalse(record["wall_time_is_model_latency"])
            self.assertEqual(json.loads((root / "command.json").read_text()), record)

    def test_nonzero_exit_preserves_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(MODULE, "snapshot", return_value={}):
            root = Path(directory) / "failure"
            with self.assertRaisesRegex(RuntimeError, "command failed"):
                MODULE.observe([sys.executable, "-c", "raise SystemExit(7)"], root)
            evidence = json.loads((root / "command.json").read_text())
            self.assertEqual(evidence["exit_code"], 7)
            self.assertEqual(evidence["status"], "failed")

    def test_foreign_client_during_run_is_preserved_but_rejected(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(MODULE, "snapshot", return_value={}), \
                patch.object(MODULE, "registered_bpu_users", side_effect=[set(), {987654321}, {987654321}]):
            root = Path(directory) / "interference"
            with self.assertRaisesRegex(RuntimeError, "exclude this entire run"):
                MODULE.observe([sys.executable, "-c", "import time; print('Infer time: 1 ms', flush=True); time.sleep(10)"], root,
                               watch_bpu=True)
            record = json.loads((root / "command.json").read_text())
            self.assertEqual(record["exit_code"], -15)
            self.assertTrue(record["contention_stop_requested"])
            self.assertEqual(record["resource_status"], "interference-detected")
            self.assertEqual(record["foreign_bpu_users"], [987654321])
            self.assertEqual(record["infer_records"], 1)

    def test_reuse_admits_clean_run_and_rejects_foreign_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stdout.log").write_text("Infer time: 1 ms\n")
            record = {"status": "completed", "exit_code": 0, "infer_records": 1,
                      "pid": 123, "stdout_sha256": MODULE.sha256(root / "stdout.log")}
            (root / "command.json").write_text(json.dumps(record))
            idle = {"files": {"/sys/devices/system/bpu/users": "user ratio\n",
                               "/sys/devices/system/bpu/bpu0/users": "user ratio\n"}}
            for name in ("before.json", "after.json"):
                (root / name).write_text(json.dumps(idle))
            self.assertEqual(MODULE.reusable_record(root, 1), (record, None))
            idle["files"]["/sys/devices/system/bpu/users"] = "1160177 54\n"
            (root / "after.json").write_text(json.dumps(idle))
            reused, reason = MODULE.reusable_record(root, 1)
            self.assertIsNone(reused)
            self.assertIn("foreign BPU", reason)


if __name__ == "__main__":
    unittest.main()
