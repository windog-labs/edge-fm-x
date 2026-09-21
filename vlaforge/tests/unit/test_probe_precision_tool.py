import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vlaforge.analysis.precision_probe import tensor_bundle_digest
from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.serializer import module_to_data
from vlaforge.ir.types import TensorType
from vlaforge.numerical_context import snapshot


def tool():
    spec = importlib.util.spec_from_file_location("probe_tool", Path(__file__).resolve().parents[2] / "tools/probe_precision.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def protocol_case(tmp_path):
    api = tool()
    vector = TensorType((2, 3), "f32")
    flag = TensorType((1,), "bool")
    class Encode(torch.nn.Module):
        def forward(self, value): return value * 2
    class Update(torch.nn.Module):
        def forward(self, value, noise): return value + noise, value.square()
    @tensor_region("encode", inputs=(Value("value", vector),), outputs=(vector,))
    def encode(value): return Encode()(value)
    @tensor_region("update", inputs=(Value("value", vector), Value("noise", vector)), outputs=(vector, vector))
    def update(value, noise): return Update()(value, noise)
    builder = InvocationBuilder("probe_example", inputs=(InputPort("observation", vector), InputPort("noise", vector),
                                                         InputPort("accepted", flag)),
                                 outputs=(OutputPort("action", vector), OutputPort("auxiliary", vector)))
    (initial,) = builder.call(encode, builder.input("observation"))
    noise = builder.input("noise")
    value, auxiliary = builder.iterate((initial, initial),
                                      lambda step, current, unused: builder.call(update, current, noise), steps=2)
    program = builder.finish({"action": value, "auxiliary": auxiliary}, accepted=builder.input("accepted"))
    module_path = tmp_path / "module.json"
    api.write(module_path, module_to_data(program.module))
    def ref(path, fmt=None):
        result = {"path": str(path), "sha256": api.file_sha(path)}
        if fmt is not None: result["format"] = fmt
        return result
    regions = []
    for name, model, args, stage in (("encode", Encode(), (torch.ones(2, 3),), "context"),
                                     ("update", Update(), (torch.ones(2, 3), torch.ones(2, 3)), "iteration")):
        path = tmp_path / (name + ".pt2")
        torch.export.save(torch.export.export(model, args, strict=True), path)
        regions.append({"name": name, "path": str(path), "artifact_sha256": api.file_sha(path), "stage": stage})
    samples = []
    for index in range(2):
        actual = {"observation": torch.full((2, 3), index + 1., dtype=torch.float32),
                  "noise": torch.ones(2, 3) / 2, "accepted": torch.tensor([True])}
        output = Encode()(actual["observation"])
        for _ in range(2): output, aux = Update()(output, actual["noise"])
        input_refs, output_refs = {}, {}
        for name, tensor in actual.items():
            path = tmp_path / f"{index}-{name}.npy"
            np.save(path, tensor.numpy())
            input_refs[name] = ref(path, "npy")
        for name, tensor in (("action", output), ("auxiliary", aux)):
            path = tmp_path / f"{index}-{name}.npy"
            np.save(path, tensor.numpy())
            output_refs[name] = ref(path, "npy")
        samples.append({"sample_id": f"sample-{index}", "partition_key": f"episode-{index}",
                        "input_sha256": tensor_bundle_digest({key: value for key, value in actual.items() if key != "noise"}),
                        "noise_sha256": tensor_bundle_digest({"noise": actual["noise"]}),
                        "split": "calibration" if index == 0 else "held-out", "inputs": input_refs,
                        "noise_names": ["noise"], "expected_outputs": output_refs,
                        "source_identity": {"fixture": "real CPU torch.export test", "episode": index}})
    protocol = {"schema": api.SCHEMA, "module": ref(module_path), "invocation": program.module.invocations[0].name,
                "regions": regions,
                "sites": [{"name": "activation", "region": "update", "node": "add", "artifact_sha256": regions[1]["artifact_sha256"],
                           "stage": "iteration", "quantize": True}],
                "step_keys": ["actual-step0", "actual-step1"], "step_groups": [[0, 1]],
                "profile": {"N": 2, "cpu_fixture": True}, "numerical_context": snapshot().to_dict(),
                "samples": samples, "evidence": [ref(module_path)],
                "retain_sites": ["activation"],
                "execution_supervision": "external-exclusive-device-owner-monitor"}
    return api, protocol


def test_actual_public_interpreter_runs_all_regions_and_all_outputs(tmp_path):
    api, protocol = protocol_case(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    api.write(out / "protocol.json", protocol)
    api.run(protocol, out)
    report = api.read(out / "report.json")
    assert report["status"] == "validated" and len(report["samples"]) == 2
    assert not report["held_out_used_for_fit"] and not report["real_low_precision_kernel_verified"]
    assert len(report["plans"]) == 4
    for index in range(2):
        sample = api.read(out / f"sample-{index:06d}/report.json")
        assert sample["source_module"] == protocol["module"]
        assert len(sample["outputs"]) == 2 and len(sample["calls"]) == 3
        assert len(sample["retained_activations"]) == 2
        for retained in sample["retained_activations"]:
            assert api.file_sha(out / f"sample-{index:06d}" / retained["file"]) == retained["sha256"]
            assert retained["split"] == ("calibration" if index == 0 else "held-out")
        for name in ("action", "auxiliary"):
            expected = np.load(protocol["samples"][index]["expected_outputs"][name]["path"]).tobytes()
            assert (out / f"sample-{index:06d}/{name}.bin").read_bytes() == expected
    calibration = api.read(out / "calibration.json")
    assert {item["sample_id"] for item in calibration["observations"]} == {"sample-0"}


@pytest.mark.parametrize("mutation", ["unknown", "source_hash", "missing_input", "missing_output", "input_hash", "noise_hash", "shape"])
def test_protocol_rejects_unbound_or_partial_real_data(tmp_path, mutation):
    api, protocol = protocol_case(tmp_path)
    if mutation == "unknown": protocol["skip_output_gate"] = True
    if mutation == "source_hash": protocol["module"]["sha256"] = "0" * 64
    if mutation == "missing_input": del protocol["samples"][0]["inputs"]["noise"]
    if mutation == "missing_output": del protocol["samples"][0]["expected_outputs"]["auxiliary"]
    if mutation == "input_hash": protocol["samples"][0]["input_sha256"] = "0" * 64
    if mutation == "noise_hash": protocol["samples"][0]["noise_sha256"] = "0" * 64
    if mutation == "shape":
        item = protocol["samples"][0]["expected_outputs"]["action"]
        np.save(item["path"], np.zeros(3, dtype=np.float32))
        item["sha256"] = api.file_sha(item["path"])
    with pytest.raises(ValueError): api.validate_protocol(protocol)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "ambiguous.json"
    path.write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError, match="duplicate"): tool().read(path)
