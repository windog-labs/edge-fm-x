"""Verify a serialized EP, remove unused state and check the complete example."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "python"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--numerical-context", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-process-global", action="store_true", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from cogact_gpu_monitor import child_handshake

    child_handshake()
    import torch
    from benchmark_operator_examples import digest, output_identity, output_metrics
    from vlaforge.analysis.unused_state import prune_unused_state, save_pruned_state
    from vlaforge.numerical_context import NumericalContext, offline_restore, snapshot

    if digest(args.export) != args.expected_sha256:
        raise ValueError("source EP differs from its expected SHA256")
    required = NumericalContext.from_json(args.numerical_context.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time_ns()
    report = {
        "schema": "vlaforge.unused_state_example_verification/1",
        "status": "running",
        "pid": os.getpid(),
        "torch": str(torch.__version__),
        "source": str(args.export),
        "source_sha256": args.expected_sha256,
        "numerical_context": required.to_dict(),
        "context_file_sha256": digest(args.numerical_context),
        "driver_sha256": digest(Path(__file__)),
        "pass_source_sha256": digest(
            SOURCE / "python/vlaforge/analysis/unused_state.py"
        ),
        "caller_context_before": snapshot().to_dict(),
        "full_model_output_verified": False,
        "runtime_peak_reduction_verified": False,
        "device_remap": False,
        "scope": "complete serialized Region example, not complete model or latency",
    }
    try:
        with (
            offline_restore(
                required, acknowledge_process_global=args.acknowledge_process_global
            ),
            torch.inference_mode(),
        ):
            required.require_current()
            original = torch.export.load(args.export)
            if original.example_inputs is None:
                raise ValueError("serialized example inputs are required")
            positional, keywords = original.example_inputs
            reference = original.module()(*positional, **keywords)
            result = prune_unused_state(
                original, source_artifact_sha256=args.expected_sha256
            )
            candidate = result.program.module()(*positional, **keywords)
            comparisons = {"in_memory": output_metrics(reference, candidate)}
            if not all(item["bitwise_equal"] for item in comparisons["in_memory"]):
                raise ValueError("pruned complete Region output differs from original")
            ledger = save_pruned_state(result, args.output / "pruned")
            reloaded = torch.export.load(args.output / "pruned/program.pt2")
            saved = reloaded.module()(*positional, **keywords)
            comparisons["saved_and_reloaded"] = output_metrics(reference, saved)
            if not all(
                item["bitwise_equal"] for item in comparisons["saved_and_reloaded"]
            ):
                raise ValueError(
                    "saved pruned complete Region output differs from original"
                )
            required.require_current()
            torch.save(
                {"source": reference, "pruned": candidate, "reloaded": saved},
                args.output / "outputs.pt",
            )
            report.update(
                comparisons=comparisons,
                output_identity=output_identity(reference),
                output_archive_sha256=digest(args.output / "outputs.pt"),
                ledger_sha256=digest(args.output / "pruned/ledger.json"),
                original_unique_storage_bytes=ledger["source_unique_storage_bytes"],
                pruned_unique_storage_bytes=ledger["result_unique_storage_bytes"],
                removed_state_names=len(ledger["removed_state"]),
                removed_placeholders=len(ledger["removed_placeholders"]),
            )
        if digest(args.export) != args.expected_sha256:
            raise ValueError("source EP changed during pruning")
        report.update(status="passed", caller_context_after=snapshot().to_dict())
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["elapsed_ns"] = time.time_ns() - started
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps({"status": "passed", "report": str(args.output / "report.json")}))


if __name__ == "__main__":
    main()
