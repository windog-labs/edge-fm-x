"""Capture explicit real RDT regions after a strict online reference run."""

from __future__ import annotations

import json
from pathlib import Path
import traceback

from vlaforge.adapters.rdt.rdt_assets import file_identity


def capture_real_regions(loaded, reference, output: Path):
    import numpy as np
    import torch
    from vlaforge.adapters.rdt.rdt_fresh import build_rdt_fresh_program, inputs_from_official_reference
    from vlaforge.analysis.operator_inventory import exported_operator_inventory
    from vlaforge.frontend import capture_region, save_exported_region
    from vlaforge.ir.serializer import canonical_json
    from vlaforge.validation.contracts import NumericContract
    from vlaforge.validation.deployment_metrics import compare_action_chunk

    output.mkdir(parents=True, exist_ok=False)
    supplied = inputs_from_official_reference(loaded, reference)
    report = {"schema": "vlaforge.rdt_fresh_capture/1", "status": "started", "regions": [],
              "compiled": False, "no_python_deployment": False, "tool": file_identity(Path(__file__))}
    def save():
        (output / "capture.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    built = build_rdt_fresh_program(loaded, supplied)
    (output / "module.json").write_text(canonical_json(built.program.module, indent=2) + "\n")
    report["module"] = file_identity(output / "module.json")
    torch.save(dict(built.input_tensors), output / "inputs.pt")
    report["inputs"] = file_identity(output / "inputs.pt")
    arrays = {"candidate": built.eager_actions.float().cpu().numpy(),
              "reference": reference["unified_actions"].float().cpu().numpy()}
    for index, value in enumerate(built.eager_samples):
        arrays[f"tensor_carry_sample_{index}"] = value.float().cpu().numpy()
        arrays[f"tensor_carry_model_output_{index}"] = built.eager_model_outputs[index].float().cpu().numpy()
    np.savez_compressed(output / "full_fresh_eager.npz", **arrays)
    report["full_fresh_eager"] = file_identity(output / "full_fresh_eager.npz")
    report["same_torch_parity"] = {
        "online_t5_exact": torch.equal(built.eager_language, reference["language_tokens"]),
        "online_vision_exact": torch.equal(built.eager_images, reference["image_tokens"].reshape_as(built.eager_images)),
        "model_outputs_exact": [torch.equal(left, right) for left, right in zip(built.eager_model_outputs, reference["model_outputs"], strict=True)],
        "full_chunk_exact": torch.equal(built.eager_actions, reference["unified_actions"]),
    }
    difference = arrays["candidate"].astype(np.float64) - arrays["reference"].astype(np.float64)
    report["same_torch_parity"].update(mse=float(np.mean(difference**2)), max_abs=float(np.max(np.abs(difference))))
    fidelity = compare_action_chunk(arrays["reference"], arrays["candidate"], sample_id="recorded-real-input",
                                    space="unified-model-action-space", contract=NumericContract(0, 0))
    (output / "fidelity.json").write_text(json.dumps(fidelity, indent=2, allow_nan=False) + "\n")
    report["fidelity"] = file_identity(output / "fidelity.json")
    save()
    # Small state/solver captures fail early without serializing encoder weights.
    regions = sorted(built.program.module.regions, key=lambda region: sum(
        parameter.numel() for parameter in built.program.regions[region.name].parameters()))
    for region in regions:
        record = {"name": region.name, "status": "started", "parameter_elements": sum(
            parameter.numel() for parameter in built.program.regions[region.name].parameters())}
        report["regions"].append(record)
        save()
        print(json.dumps({"capture_region": region.name, "status": "started"}), flush=True)
        try:
            module = built.program.regions[region.name]
            arguments = built.region_examples[region.name]
            outcome = capture_region(region, module, arguments, strict=False,
                                     absolute_tolerance=0.0, relative_tolerance=0.0).require_supported()
            path = output / "exports" / f"{region.name}.pt2e"
            evidence = path.with_suffix(".capture.json")
            save_exported_region(outcome, program_path=path, evidence_path=evidence)
            example_path = path.with_suffix(".inputs.pt")
            torch.save(tuple(value.detach().cpu().clone() for value in arguments), example_path)
            record.update(status="captured", export=file_identity(path), evidence=file_identity(evidence),
                          examples=file_identity(example_path))
            try:
                inventory = exported_operator_inventory(outcome.exported_program, region_name=region.name)
                inventory_path = path.with_suffix(".operators.json")
                inventory_path.write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
                record["operator_inventory"] = file_identity(inventory_path)
            except Exception as error:
                record["operator_inventory"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        save()
        print(json.dumps({"capture_region": region.name, "status": record["status"]}), flush=True)
    report["status"] = "captured_all_regions" if all(item["status"] == "captured" for item in report["regions"]) else "partial_capture"
    report["capture_is_paper_acceptance"] = False
    save()
    return report
