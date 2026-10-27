import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("materialize_native_observations", ROOT / "tools/materialize_native_observations.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def fixture():
    types = [("image", "f32", [1, 2]), ("state", "f32", [1, 2]),
             ("ids", "i32", [1, 4]), ("mask", "bool", [1, 4]),
             ("present", "bool", [1]), ("copied", "f32", [1, 2])]
    inputs = [{"id": index, "name": name, "payload": {"kind": "tensor", "dtype": dtype, "shape": shape}}
              for index, (name, dtype, shape) in enumerate(types)]
    profile = {"schema": "vlaforge.native_host_profile/1", "raw_tensors": [
        {"name": "pixels", "dtype": "uint8", "shape": [2]}, {"name": "native_state", "dtype": "float32", "shape": [2]}],
        "texts": ["command"], "operations": [
            {"kind": "scale-u8", "source": "pixels", "output": "image", "divisor": 255},
            {"kind": "standardize-f32", "source": "native_state", "output": "state", "mean": [0, 1], "stddev": [2, 4], "epsilon": 0},
            {"kind": "fill-bool", "output": "present", "value": True},
            {"kind": "tokenize", "source": "command", "output": "ids", "mask": "mask", "max_length": 4,
             "pad_id": 17, "add_special_tokens": True, "truncate": True,
             "padding_side": "left", "truncation_side": "right", "suffix": "\n"},
            {"kind": "copy", "source": "native_state", "output": "copied"}]}
    return profile, inputs


@pytest.mark.parametrize("change", ["duplicate", "missing", "token-length", "bool-policy", "pad-range", "negative-std", "nan-mean", "dtype", "divisor"])
def test_invalid_or_partial_host_contract_is_rejected(change):
    profile, inputs = fixture()
    if change == "duplicate":
        profile["operations"].append(copy.deepcopy(profile["operations"][0]))
    elif change == "missing":
        profile["operations"].pop()
    elif change == "token-length":
        profile["operations"][3]["max_length"] = 5
    elif change == "bool-policy":
        profile["operations"][3]["truncate"] = 1
    elif change == "pad-range":
        profile["operations"][3]["pad_id"] = 2147483648
    elif change == "negative-std":
        profile["operations"][1]["stddev"][0] = -1
    elif change == "nan-mean":
        profile["operations"][1]["mean"][0] = float("nan")
    elif change == "dtype":
        inputs[0]["payload"]["dtype"] = "i32"
    else:
        profile["operations"][0]["divisor"] = 0
    with pytest.raises(ValueError):
        TOOL.render_profile(profile, inputs, 2)


def test_generated_profile_executes_complete_fresh_host_operations(tmp_path):
    """Actual C++ policy execution with an explicit token fixture, not HF/board evidence."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    profile, inputs = fixture()
    (tmp_path / "host_profile.h").write_text(TOOL.render_profile(profile, inputs, 2))
    source = r'''
#include "host_profile.h"
#include <vector>
int main() {
  std::vector<vlaforge::tools::HostTensorBuffer> raw, output;
  for (const auto& spec : raw_specs) raw.emplace_back(spec);
  for (const auto& spec : output_specs) output.emplace_back(spec);
  static_cast<std::uint8_t*>(raw[0].data())[0] = 10;
  static_cast<std::uint8_t*>(raw[0].data())[1] = 20;
  static_cast<float*>(raw[1].data())[0] = 2;
  static_cast<float*>(raw[1].data())[1] = 5;
  int token_calls = 0;
  vlaforge::tools::DeclaredHostPipeline pipeline(operations,
      [&](std::string_view text, bool special, std::vector<std::int32_t>* ids) {
        ++token_calls;
        if (text != "abc\n" || !special) return vlaforge::runtime::Status::Error(
            vlaforge::runtime::StatusCode::kInvalidArgument, 0, "bad fixture prompt");
        *ids = {4, 2};
        return vlaforge::runtime::Status::Ok();
      });
  if (!pipeline.Prepare(raw, {"abc"}, &output).ok()) return 1;
  if (static_cast<float*>(output[0].data())[0] != 10.0F / 255.0F ||
      static_cast<float*>(output[1].data())[1] != 1.0F ||
      static_cast<std::int32_t*>(output[2].data())[0] != 17 ||
      static_cast<std::int32_t*>(output[2].data())[3] != 2 ||
      static_cast<std::uint8_t*>(output[3].data())[0] != 0 ||
      static_cast<std::uint8_t*>(output[3].data())[3] != 1 ||
      static_cast<std::uint8_t*>(output[4].data())[0] != 1 ||
      static_cast<float*>(output[5].data())[1] != 5.0F) return 2;
  static_cast<std::uint8_t*>(raw[0].data())[0] = 100;
  if (!pipeline.Prepare(raw, {"abc\n"}, &output).ok() || token_calls != 2 ||
      static_cast<float*>(output[0].data())[0] != 100.0F / 255.0F) return 3;
  return 0;
}
'''
    path = tmp_path / "probe.cpp"
    path.write_text(source)
    binary = tmp_path / "probe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I" + str(tmp_path), "-I" + str(ROOT / "tools"), "-I" + str(ROOT / "include"),
                    str(path), str(ROOT / "runtime/host_preprocessing.cpp"), "-o", str(binary)], check=True, capture_output=True)
    result = subprocess.run([str(binary)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def extended_fixture():
    declarations = [("pixels", "f32", [1, 3, 2, 2]), ("state", "f32", [1, 1, 5]),
                    ("available", "f32", [1, 1, 5]), ("ids", "i32", [1, 4]), ("mask", "i32", [1, 4])]
    inputs = [{"id": index, "name": name, "payload": {"kind": "tensor", "dtype": dtype, "shape": shape}}
              for index, (name, dtype, shape) in enumerate(declarations)]
    profile = {"schema": "vlaforge.native_host_profile/1", "raw_tensors": [
        {"name": "camera", "dtype": "uint8", "shape": [2, 2, 3]},
        {"name": "joints", "dtype": "float32", "shape": [1, 2]}], "texts": ["instruction"], "operations": [
        {"kind": "images-u8-to-chw-f32", "sources": ["camera"], "output": "pixels", "width": 2, "height": 2,
         "pad_to_square": False, "background": [127, 127, 127], "filter": "bicubic", "rescale": 1 / 255,
         "mean": [.5, .5, .5], "stddev": [.5, .5, .5], "rounding": "bf16-rne-f32"},
        {"kind": "scatter-divide-f32", "source": "joints", "output": "state", "mask": "available",
         "indices": [3, 1], "divisors": [1, 2], "rounding": "bf16-rne-f32"},
        {"kind": "tokenize", "source": "instruction", "output": "ids", "mask": "mask", "max_length": 4,
         "pad_id": 17, "add_special_tokens": True, "truncate": False, "padding_side": "right", "truncation_side": "right", "suffix": ""}]}
    return profile, inputs


def wide_copy_fixture():
    profile = {"schema": "vlaforge.native_host_profile/1", "raw_tensors": [
        {"name": "state", "dtype": "float64", "shape": [4]},
        {"name": "indices", "dtype": "int64", "shape": [3]}],
        "texts": [], "operations": [
            {"kind": "copy", "source": "state", "output": "state"},
            {"kind": "copy", "source": "indices", "output": "indices"}]}
    inputs = [{"id": index, "name": name, "payload": {"kind": "tensor", "dtype": dtype, "shape": shape}}
              for index, (name, dtype, shape) in enumerate([("state", "f64", [4]), ("indices", "i64", [3])])]
    return profile, inputs


@pytest.mark.parametrize("index,dtype", [(0, "f32"), (1, "i32")])
def test_wide_host_copy_never_implicitly_narrows(index, dtype):
    profile, inputs = wide_copy_fixture()
    inputs[index]["payload"]["dtype"] = dtype
    with pytest.raises(ValueError, match="correctly typed target"):
        TOOL.render_profile(profile, inputs, 1)


def test_wide_host_storage_copy_finiteness_and_overflow(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    profile, inputs = wide_copy_fixture()
    (tmp_path / "host_profile.h").write_text(TOOL.render_profile(profile, inputs, 1))
    path = tmp_path / "wide.cpp"
    path.write_text(r'''
#include "host_profile.h"
int main() {
  using namespace vlaforge::tools;
  std::vector<HostTensorBuffer> raw, output;
  for (const auto& spec : raw_specs) raw.emplace_back(spec);
  for (const auto& spec : output_specs) output.emplace_back(spec);
  const double state[] = {1.0 + std::ldexp(1.0, -40), -0.0, std::numeric_limits<double>::denorm_min(), 1e200};
  const std::int64_t indices[] = {std::numeric_limits<std::int64_t>::min() + 1,
                                (std::int64_t{1} << 60) + 3, std::numeric_limits<std::int64_t>::max()};
  if (raw[0].bytes() != sizeof(state) || raw[1].bytes() != sizeof(indices)) return 1;
  std::memcpy(raw[0].data(), state, sizeof(state));
  std::memcpy(raw[1].data(), indices, sizeof(indices));
  DeclaredHostPipeline pipeline(operations, {});
  if (!raw[0].finite() || !raw[1].finite() || !pipeline.Prepare(raw, {}, &output).ok() ||
      std::memcmp(output[0].data(), state, sizeof(state)) ||
      std::memcmp(output[1].data(), indices, sizeof(indices))) return 2;
  static_cast<double*>(raw[0].data())[0] = std::nextafter(1.0, 2.0);
  static_cast<std::int64_t*>(raw[1].data())[0] += 1;
  if (!pipeline.Prepare(raw, {}, &output).ok() ||
      std::memcmp(output[0].data(), raw[0].data(), sizeof(state)) ||
      std::memcmp(output[1].data(), raw[1].data(), sizeof(indices))) return 3;
  for (double value : {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    static_cast<double*>(raw[0].data())[3] = value;
    if (raw[0].finite()) return 4;
  }
  HostTensorBuffer narrow({VLAFORGE_DTYPE_F32, {1}});
  static_cast<float*>(narrow.data())[0] = std::numeric_limits<float>::quiet_NaN();
  if (narrow.finite()) return 5;
  HostTensorBuffer boolean({VLAFORGE_DTYPE_BOOL, {1}}), pixels({VLAFORGE_DTYPE_U8, {1}});
  static_cast<std::uint8_t*>(boolean.data())[0] = 2;
  static_cast<std::uint8_t*>(pixels.data())[0] = 255;
  if (boolean.finite() || !pixels.finite()) return 6;
  for (auto dtype : {VLAFORGE_DTYPE_I64, VLAFORGE_DTYPE_F64}) {
    try {
      HostTensorBuffer huge({dtype, {static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / 8 + 1)}});
      return 7;
    } catch (const std::invalid_argument&) {}
  }
  return 0;
}
''')
    binary = tmp_path / "wide"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I" + str(tmp_path), "-I" + str(ROOT / "tools"), "-I" + str(ROOT / "include"),
                    str(path), str(ROOT / "runtime/host_preprocessing.cpp"), "-o", str(binary)],
                   check=True, capture_output=True)
    result = subprocess.run([str(binary)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_materialized_wide_payloads_keep_every_source_byte(tmp_path):
    import numpy as np

    profile, inputs = wide_copy_fixture()
    values = {"state": np.array([1 + 2.0 ** -40, -0.0, np.nextafter(0., 1.), 1e200], dtype=np.float64),
              "indices": np.array([-(2 ** 63) + 1, 2 ** 60 + 3, 2 ** 63 - 1], dtype=np.int64)}
    source = tmp_path / "observation.npz"
    np.savez(source, **values)
    ir = tmp_path / "ir.json"
    ir.write_text(json.dumps({"inputs": inputs}))
    manifest = tmp_path / "raw.json"
    manifest.write_text(json.dumps({"records": [{"index": 0, "path": source.name,
        "sha256": TOOL.digest(source), "arrays": {name: {"sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                                                  for name, value in values.items()}, "text": {}}]}))
    profile.update(raw_manifest_sha256=TOOL.digest(manifest), model_ir_sha256=TOOL.digest(ir), assets={})
    declaration = tmp_path / "profile.json"
    declaration.write_text(json.dumps(profile))
    output = tmp_path / "prepared"
    result = subprocess.run([sys.executable, str(ROOT / "tools/materialize_native_observations.py"),
                             "--profile", str(declaration), "--raw-manifest", str(manifest),
                             "--model-ir", str(ir), "--output", str(output)], check=True, capture_output=True)
    assert json.loads(result.stdout)["status"] == "prepared-not-executed"
    for position, value in enumerate(values.values()):
        assert (output / "raw" / "0" / f"{position}.bin").read_bytes() == value.tobytes()


def test_runner_finiteness_keeps_frozen_buffer_interface_compatible(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    template = (ROOT / "tools/raw_observation_runner.cpp.in").read_text()
    marker = "bool finite(const Buffer& buffer) {"
    assert template.count(marker) == 1
    implementation = marker + template.split(marker, 1)[1].split("\n}  // namespace", 1)[0]
    path = tmp_path / "legacy.cpp"
    path.write_text(r'''
#include "vlaforge/runtime/region_executable.h"
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
struct Buffer {
  VLAForgeDType dtype;
  std::size_t count;
  const void* payload;
  const void* data() const { return payload; }
};
''' + implementation + r'''
int main() {
  double value = 1e200;
  Buffer wide{VLAFORGE_DTYPE_F64, 1, &value};
  if (!finite(wide)) return 1;
  for (double invalid : {std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    value = invalid;
    if (finite(wide)) return 2;
  }
  float narrow = std::numeric_limits<float>::quiet_NaN();
  if (finite({VLAFORGE_DTYPE_F32, 1, &narrow})) return 3;
  std::uint8_t byte = 2;
  if (finite({VLAFORGE_DTYPE_BOOL, 1, &byte}) || !finite({VLAFORGE_DTYPE_U8, 1, &byte})) return 4;
  std::int64_t integer = std::numeric_limits<std::int64_t>::max();
  return finite({VLAFORGE_DTYPE_I64, 1, &integer}) ? 0 : 5;
}
''')
    binary = tmp_path / "legacy"
    result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                             "-I" + str(ROOT / "include"), str(path), "-o", str(binary)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    result = subprocess.run([str(binary)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("change", ["duplicate-index", "outside-index", "zero-divisor", "wrong-source-width", "mask-shape",
    "rounding", "channels", "filter", "background", "stddev", "rescale", "image-size", "duplicate-view", "token-mask"])
def test_extended_contract_rejects_ambiguous_or_invalid_mapping(change):
    profile, inputs = extended_fixture()
    image, scatter, _ = profile["operations"]
    if change == "duplicate-index": scatter["indices"] = [1, 1]
    elif change == "outside-index": scatter["indices"] = [3, 5]
    elif change == "zero-divisor": scatter["divisors"][0] = 0
    elif change == "wrong-source-width": profile["raw_tensors"][1]["shape"] = [2, 1]
    elif change == "mask-shape": inputs[2]["payload"]["shape"] = [1, 1, 4]
    elif change == "rounding": image["rounding"] = "stochastic"
    elif change == "channels": profile["raw_tensors"][0]["shape"][-1] = 4
    elif change == "filter": image["filter"] = "guess"
    elif change == "background": image["background"][0] = 256
    elif change == "stddev": image["stddev"][0] = 0
    elif change == "rescale": image["rescale"] = float("nan")
    elif change == "image-size": image["height"] = 3
    elif change == "duplicate-view": image["sources"] *= 2
    else: inputs[4]["payload"]["dtype"] = "f32"
    with pytest.raises(ValueError):
        TOOL.render_profile(profile, inputs, 2)


def test_extended_fresh_pipeline_with_injected_image_and_text_fixtures(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    profile, inputs = extended_fixture()
    (tmp_path / "host_profile.h").write_text(TOOL.render_profile(profile, inputs, 2))
    path = tmp_path / "extended.cpp"
    path.write_text(r'''
#include "host_profile.h"
int main() {
  using namespace vlaforge::tools;
  std::vector<HostTensorBuffer> raw, output;
  for (const auto& spec : raw_specs) raw.emplace_back(spec);
  for (const auto& spec : output_specs) output.emplace_back(spec);
  for (int i = 0; i < 12; ++i) static_cast<std::uint8_t*>(raw[0].data())[i] = i;
  static_cast<float*>(raw[1].data())[0] = 1.00390625F;
  static_cast<float*>(raw[1].data())[1] = 5.0F;
  int image_calls = 0, token_calls = 0;
  DeclaredHostPipeline pipeline(operations,
    [&](std::string_view, bool, std::vector<std::int32_t>* ids) {
      ++token_calls; *ids = {4, 2}; return vlaforge::runtime::Status::Ok();
    }, [&](const vlaforge::backends::RgbImageU8& image, const vlaforge::backends::RgbResizeSpec&, std::uint8_t* out, std::size_t bytes) {
      ++image_calls;
      if (image.bytes != bytes) return vlaforge::runtime::Status::Error(vlaforge::runtime::StatusCode::kInvalidArgument, 0, "fixture shape");
      std::memcpy(out, image.data, bytes); return vlaforge::runtime::Status::Ok();
    });
  if (!pipeline.Prepare(raw, {"a"}, &output).ok()) return 1;
  if (static_cast<float*>(output[0].data())[0] != -1.0F ||
      static_cast<float*>(output[1].data())[3] != 1.0F || static_cast<float*>(output[1].data())[1] != 2.5F ||
      static_cast<float*>(output[2].data())[3] != 1.0F || static_cast<float*>(output[2].data())[0] != 0 ||
      static_cast<std::int32_t*>(output[3].data())[2] != 17 ||
      static_cast<std::int32_t*>(output[4].data())[0] != 1 || static_cast<std::int32_t*>(output[4].data())[2] != 0) return 2;
  static_cast<std::uint8_t*>(raw[0].data())[0] = 255;
  static_cast<float*>(raw[1].data())[0] = 2.0F;
  if (!pipeline.Prepare(raw, {"b"}, &output).ok() || image_calls != 2 || token_calls != 2 ||
      static_cast<float*>(output[0].data())[0] != 1.0F || static_cast<float*>(output[1].data())[3] != 2.0F) return 3;
  return 0;
}
''')
    binary = tmp_path / "extended"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I" + str(tmp_path),
        "-I" + str(ROOT / "tools"), "-I" + str(ROOT / "include"), str(path), str(ROOT / "runtime/host_preprocessing.cpp"),
        "-o", str(binary)], check=True, capture_output=True)
    result = subprocess.run([str(binary)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
