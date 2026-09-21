import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "tools/summarize_horizon_campaign.py"
SPEC = importlib.util.spec_from_file_location("summarize_horizon_campaign", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    stage = {"name": "fixture", "hbm": "/fixture/model.hbm", "sha256": "fixture-only",
             "inputs": ["/fixture/input.bin"]}
    manifest = {"stages": [stage], "warmup": 2, "measured": 2, "processes": 2,
                "executable": "hrt_model_exec"}
    campaign = {"status": "completed", "phase": "formal", "manifest": manifest,
                "manifest_sha256": "fixture-only", "inputs": {}, "runs": []}
    idle = {"uname": ["Linux", "fixture", "fixture-kernel", "fixture-version", "aarch64"],
            "files": {"/proc/device-tree/serial-number": "fixture-board",
                      "/sys/devices/system/bpu/users": "user\tratio\n",
                      "/sys/devices/system/bpu/bpu0/users": "user\tratio\n"}}
    for i, latencies in enumerate(([99, 88, 1, 3], [77, 66, 2, 4])):
        label = f"run-{i + 1}"
        path = root / "formal/fixture" / label
        path.mkdir(parents=True)
        source = path / "stdout.log"
        source.write_text("\n".join(f"-----Frame {n} begin-----\nInfer time: {v} ms"
                                    for n, v in enumerate(latencies)) + "\n")
        record = {"status": "completed", "exit_code": 0, "pid": i + 100,
                  "started_ns": i * 100 + 1, "finished_ns": i * 100 + 99,
                  "infer_records": 4, "peak_rss_mib": 12 + i, "model_load_ms": [1],
                  "stdout_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "command": ["hrt_model_exec", "infer", "--model_file", stage["hbm"],
                              "--input_file", stage["inputs"][0], "--frame_count", "4",
                              "--thread_num", "1"]}
        save(path / "command.json", record)
        save(path / "before.json", idle)
        save(path / "after.json", idle)
        campaign["runs"].append({"stage": "fixture", "label": label, **record})
    save(root / "formal/campaign.json", campaign)
    smoke = {**campaign, "phase": "smoke", "runs": []}
    for i, label in enumerate(("model-info", "infer", "perf")):
        path = root / "smoke/fixture" / label
        path.mkdir(parents=True)
        source = path / "stdout.log"
        source.write_text("fixture-only smoke output\n")
        mode = "model_info" if label == "model-info" else label
        command = ["hrt_model_exec", mode, "--model_file", stage["hbm"]]
        if mode != "model_info":
            command += ["--input_file", stage["inputs"][0], "--frame_count",
                        "1" if mode == "infer" else "5"]
        if mode == "perf":
            command += ["--thread_num", "1"]
        record = {"status": "completed", "exit_code": 0, "pid": i + 300,
                  "started_ns": i * 100 - 999, "finished_ns": i * 100 - 901,
                  "stdout_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "command": command}
        save(path / "command.json", record)
        save(path / "before.json", idle)
        save(path / "after.json", idle)
        smoke["runs"].append({"stage": "fixture", "label": label, **record})
    save(root / "smoke/campaign.json", smoke)
    return campaign


class CampaignSummaryTest(unittest.TestCase):
    def test_exact_warmup_exclusion_and_percentiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            subprocess.run([sys.executable, str(SOURCE), "--campaign", str(root / "formal"),
                            "--smoke", str(root / "smoke"), "--output", str(root / "summary")],
                           check=True, capture_output=True, text=True)
            report = json.loads((root / "summary/report.json").read_text())
            row = report["stages"][0]
            self.assertEqual(row["samples"], 4)
            self.assertEqual(row["mean_ms"], 2.5)
            self.assertEqual(row["p50_ms"], 2)
            self.assertEqual(row["p95_ms"], 4)
            self.assertEqual(row["p99_ms"], 4)
            self.assertEqual(row["inverse_mean_stage_calls_per_s"], 400)
            self.assertFalse(report["full_chain_e2e"])
            self.assertEqual(report["board_identity"]["serial_number"], "fixture-board")
            self.assertEqual(report["resource_evidence_by_process"][0]["status"],
                             "before-after-snapshots-only")

    def test_rejects_truncated_raw_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            (root / "formal/fixture/run-1/stdout.log").write_text("truncated")
            with self.assertRaisesRegex(ValueError, "log hash mismatch"):
                MODULE.audit(root / "formal", root / "smoke")

    def test_rejects_external_client_in_after_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            path = root / "formal/fixture/run-2/after.json"
            snapshot = json.loads(path.read_text())
            snapshot["files"]["/sys/devices/system/bpu/users"] += "1160177\t23\n"
            save(path, snapshot)
            with self.assertRaisesRegex(ValueError, "foreign BPU"):
                MODULE.audit(root / "formal", root / "smoke")

    def test_rejects_missing_smoke_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            path = root / "smoke/campaign.json"
            smoke = json.loads(path.read_text())
            smoke["runs"].pop()
            save(path, smoke)
            with self.assertRaisesRegex(ValueError, "smoke command"):
                MODULE.audit(root / "formal", root / "smoke")

    def test_rejects_corrupted_smoke_raw_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            (root / "smoke/fixture/perf/stdout.log").write_text("corrupted")
            with self.assertRaisesRegex(ValueError, "log hash mismatch"):
                MODULE.audit(root / "formal", root / "smoke")

    def test_rejects_different_board_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            for name in ("before.json", "after.json"):
                path = root / "formal/fixture/run-2" / name
                snapshot = json.loads(path.read_text())
                snapshot["files"]["/proc/device-tree/serial-number"] = "other-board"
                save(path, snapshot)
            with self.assertRaisesRegex(ValueError, "board identity differs from smoke"):
                MODULE.audit(root / "formal", root / "smoke")

    def test_rejects_contradictory_resource_status(self):
        changes = [
            {"resource_checks": 0}, {"foreign_bpu_users": [987]},
            {"resource_errors": ["monitor unavailable"]},
            {"resource_events": [{"foreign_bpu_users": [987]}]},
            {"contention_stop_requested": True}, {"resource_status": "unknown"},
        ]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                campaign = fixture(root)
                path = root / "formal/fixture/run-2/command.json"
                record = json.loads(path.read_text())
                record.update(resource_status="clear-at-poll-points", resource_checks=3,
                              foreign_bpu_users=[], resource_errors=[], resource_events=[])
                record.update(change)
                save(path, record)
                campaign["runs"][1].update(record)
                save(root / "formal/campaign.json", campaign)
                with self.assertRaisesRegex(ValueError, "resource monitoring"):
                    MODULE.audit(root / "formal", root / "smoke")

    def test_rejects_extra_measurement_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = fixture(root)
            path = root / "formal/fixture/run-2/command.json"
            record = json.loads(path.read_text())
            record["command"] += ["--enable_dump", "true"]
            save(path, record)
            campaign["runs"][1].update(record)
            save(root / "formal/campaign.json", campaign)
            with self.assertRaisesRegex(ValueError, "measurement command"):
                MODULE.audit(root / "formal", root / "smoke")

    def test_complete_stage_does_not_claim_complete_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = fixture(root)
            campaign["status"] = "running"
            save(root / "formal/campaign.json", campaign)
            subprocess.run([sys.executable, str(SOURCE), "--campaign", str(root / "formal"),
                            "--smoke", str(root / "smoke"), "--output", str(root / "summary"),
                            "--stage", "fixture"], check=True, capture_output=True, text=True)
            report = json.loads((root / "summary/report.json").read_text())
            self.assertFalse(report["whole_campaign_audited"])
            self.assertEqual(report["campaign_status_at_audit"], "running")
            self.assertEqual(report["audited_stages"], ["fixture"])
            self.assertEqual(report["stages"][0]["samples"], 4)
            with self.assertRaisesRegex(ValueError, "not completed"):
                MODULE.audit(root / "formal", root / "smoke")

    def test_stage_selection_still_requires_all_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = fixture(root)
            campaign["status"] = "running"
            campaign["runs"].pop()
            save(root / "formal/campaign.json", campaign)
            with self.assertRaisesRegex(ValueError, "number of completed processes"):
                MODULE.audit(root / "formal", root / "smoke", ["fixture"])
            with self.assertRaisesRegex(ValueError, "stage selection"):
                MODULE.audit(root / "formal", root / "smoke", ["not-in-manifest"])

    def test_rejects_running_campaign_and_overlapping_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = fixture(root)
            campaign["status"] = "running"
            save(root / "formal/campaign.json", campaign)
            with self.assertRaisesRegex(ValueError, "not completed"):
                MODULE.audit(root / "formal", root / "smoke")
            campaign["status"] = "completed"
            campaign["runs"][1]["started_ns"] = 50
            record = json.loads((root / "formal/fixture/run-2/command.json").read_text())
            record["started_ns"] = 50
            save(root / "formal/fixture/run-2/command.json", record)
            save(root / "formal/campaign.json", campaign)
            with self.assertRaisesRegex(ValueError, "overlap"):
                MODULE.audit(root / "formal", root / "smoke")


if __name__ == "__main__":
    unittest.main()
