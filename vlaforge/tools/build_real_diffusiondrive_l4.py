#!/usr/bin/env python3
"""Build and audit the real DiffusionDrive no-Python CUDA Session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.adapters.diffusiondrive_artifact import (  # noqa: E402
    DIFFUSIONDRIVE_ARTIFACT_EVIDENCE_SCHEMA,
    DIFFUSIONDRIVE_REGIONS,
)
from vlaforge.adapters.diffusiondrive_real import (  # noqa: E402
    DIFFUSIONDRIVE_CHECKPOINT_SHA256,
    DIFFUSIONDRIVE_HF_REVISION,
    DIFFUSIONDRIVE_UPSTREAM_REVISION,
    build_real_diffusiondrive_program,
)
from vlaforge.codegen import CppValidatorDefinition  # noqa: E402
from vlaforge.deployment import (  # noqa: E402
    ArtifactIdentity,
    ArtifactKind,
    BackendCapability,
    EffectAudit,
    RegionArtifactContract,
    ValueContract,
    WorkspaceContract,
    build_artifact_compile_bundle,
    load_bundle_manifest,
)
from vlaforge.ir.serializer import canonical_json, io_schema_digest  # noqa: E402


_REPORT_SCHEMA = "vlaforge.diffusiondrive_real_l4/1"
_OUTPUTS = (
    ("candidate_trajectories", (1, 20, 8, 3)),
    ("candidate_scores", (1, 20)),
    ("trajectory", (1, 8, 3)),
    ("bev_semantic_map", (1, 7, 128, 256)),
    ("agent_states", (1, 30, 5)),
    ("agent_labels", (1, 30)),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
        env=environment,
    )


def _git(command: list[str]) -> str:
    return _run(["git", *command], environment=dict(os.environ)).stdout.strip()


def _verify_l3(
    l3_root: Path,
    *,
    target: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frontend_path = l3_root / "frontend.json"
    l3_path = l3_root / "artifact-l3.json"
    frontend = _json(frontend_path)
    evidence = _json(l3_path)
    if (
        frontend.get("schema") != "vlaforge.diffusiondrive_real_frontend/1"
        or frontend.get("status") != "passed"
        or frontend.get("evidence_level") != "L2"
    ):
        raise ValueError("DiffusionDrive frontend evidence is not passing L2")
    if (
        evidence.get("schema")
        != DIFFUSIONDRIVE_ARTIFACT_EVIDENCE_SCHEMA
        or evidence.get("status") != "passed"
        or evidence.get("evidence_level") != "L3"
        or evidence.get("target") != target
    ):
        raise ValueError("DiffusionDrive artifact evidence is not passing L3")
    if (
        frontend.get("checkpoint", {}).get("sha256")
        != DIFFUSIONDRIVE_CHECKPOINT_SHA256
        or evidence.get("checkpoint", {}).get("sha256")
        != DIFFUSIONDRIVE_CHECKPOINT_SHA256
    ):
        raise ValueError("DiffusionDrive L2/L3 checkpoint identity mismatch")
    return frontend, evidence


def _artifact_contracts(
    module: Any,
    *,
    l3_root: Path,
    checkpoint_sha256: str,
    upstream_revision: str,
    target: str,
    backend_variant: str,
) -> tuple[
    dict[str, RegionArtifactContract],
    dict[str, Path],
    list[dict[str, object]],
]:
    contracts: dict[str, RegionArtifactContract] = {}
    sources: dict[str, Path] = {}
    records: list[dict[str, object]] = []
    digest = io_schema_digest(module)
    for region_id, region in enumerate(module.regions):
        name = region.name
        if name not in DIFFUSIONDRIVE_REGIONS:
            raise ValueError(f"unexpected DiffusionDrive L4 Region: {name}")
        capture_path = l3_root / "exports" / f"{name}.capture.json"
        export_path = l3_root / "exports" / f"{name}.pt2e"
        artifact_path = l3_root / "artifacts" / f"{name}.pt2"
        compile_path = l3_root / "artifacts" / f"{name}.compile.json"
        for path in (
            capture_path,
            export_path,
            artifact_path,
            compile_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        capture = _json(capture_path)
        compiled = _json(compile_path)
        if (
            capture.get("schema") != "vlaforge.frontend_capture/2"
            or capture.get("region_name") != name
            or not capture.get("effect_audit", {}).get("passed", False)
        ):
            raise ValueError(f"{name}: invalid capture contract")
        if (
            compiled.get("schema")
            != "vlaforge.compile_artifact_result/1"
            or compiled.get("status") != "passed"
            or compiled.get("backend") != "aoti"
            or compiled.get("target") != target
            or compiled.get("exported_program", {}).get("sha256")
            != _sha256(export_path)
            or compiled.get("artifact", {}).get("sha256")
            != _sha256(artifact_path)
            or compiled.get("artifact", {}).get("size_bytes")
            != artifact_path.stat().st_size
        ):
            raise ValueError(f"{name}: invalid compiled artifact chain")
        contract = RegionArtifactContract(
            region_id=region_id,
            region_name=name,
            inputs=tuple(
                ValueContract.from_dict(item)
                for item in capture.get("inputs", ())
            ),
            outputs=tuple(
                ValueContract.from_dict(item)
                for item in capture.get("outputs", ())
            ),
            io_schema_digest=digest,
            identity=ArtifactIdentity(
                model_name="DiffusionDrive NAVSIM 88.1 PDMS",
                upstream_revision=upstream_revision,
                checkpoint_identity=f"sha256:{checkpoint_sha256}",
                graph_sha256=str(capture["graph_digest"]),
            ),
            artifact_kind=ArtifactKind.AOTI_PACKAGE,
            artifact_path=f"artifacts/{name}.pt2",
            artifact_sha256=_sha256(artifact_path),
            artifact_size_bytes=artifact_path.stat().st_size,
            workspace=WorkspaceContract(device="cuda:0"),
            capability=BackendCapability(
                backend="aoti",
                target=target,
                supported_dtypes=("f32", "i64"),
                supports_dynamic_shapes=False,
                supports_device_resident_io=True,
                requires_synchronize=True,
            ),
            effect_audit=EffectAudit.from_dict(capture["effect_audit"]),
            backend_variant=backend_variant,
        )
        contracts[name] = contract
        sources[name] = artifact_path
        records.append(
            {
                "region_id": region_id,
                "name": name,
                "capture_graph_sha256": capture["graph_digest"],
                "artifact_sha256": contract.artifact_sha256,
                "artifact_size_bytes": contract.artifact_size_bytes,
                "compile_seconds": compiled["compile_seconds"],
                "graph_nodes": compiled["graph_nodes"],
            }
        )
    return contracts, sources, records


def _write_input_fixtures(root: Path) -> dict[str, dict[str, object]]:
    import torch

    root.mkdir(parents=True, exist_ok=True)
    values = {
        "camera_feature": torch.linspace(
            -1.0,
            1.0,
            3 * 256 * 1024,
            device="cuda:0",
            dtype=torch.float32,
        ).reshape(1, 3, 256, 1024),
        "lidar_feature": torch.linspace(
            -0.5,
            0.5,
            256 * 256,
            device="cuda:0",
            dtype=torch.float32,
        ).reshape(1, 1, 256, 256),
        "status_feature": torch.linspace(
            -0.25,
            0.25,
            8,
            device="cuda:0",
            dtype=torch.float32,
        ).reshape(1, 8),
        "noise": torch.linspace(
            -1.0,
            1.0,
            20 * 8 * 2,
            device="cuda:0",
            dtype=torch.float32,
        ).reshape(1, 20, 8, 2),
    }
    torch.cuda.synchronize()
    records = {}
    for name, tensor in values.items():
        path = root / f"{name}.bin"
        payload = (
            tensor.detach()
            .contiguous()
            .view(torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
        )
        path.write_bytes(payload)
        records[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": len(payload),
            "shape": list(tensor.shape),
            "dtype": "float32",
            "device_generated": "cuda:0",
        }
    _write_json(
        root / "inputs.json",
        {
            "schema": "vlaforge.diffusiondrive_l4_inputs/1",
            "inputs": records,
        },
    )
    return records


def _runner_source() -> str:
    source = r"""
#include "session_generated.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool Cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation,
               cudaGetErrorString(status));
  return false;
}

template <typename T>
class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    status_ = cudaMalloc(reinterpret_cast<void**>(&data_),
                         count_ * sizeof(T));
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  bool ok() const { return status_ == cudaSuccess && data_ != nullptr; }
  T* data() { return data_; }
  std::size_t bytes() const { return count_ * sizeof(T); }
  bool Upload(const T* source) {
    return Cuda(cudaMemcpy(data_, source, bytes(), cudaMemcpyHostToDevice),
                "cudaMemcpy H2D");
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
  cudaError_t status_ = cudaSuccess;
};

VLAForgeBoundTensor Tensor(
    void* data, std::uint64_t bytes, const std::int64_t* dimensions,
    std::uint32_t rank) {
  return VLAForgeBoundTensor{
      sizeof(VLAForgeBoundTensor),
      {data, bytes, dimensions, rank, VLAFORGE_DTYPE_F32,
       {VLAFORGE_DEVICE_CUDA, 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS,
      64u};
}

VLAForgeInputStamp Stamp(std::uint64_t revision, bool present = true) {
  VLAForgeInputStamp stamp{};
  stamp.struct_size = sizeof(VLAForgeInputStamp);
  stamp.has_revision = present ? 1u : 0u;
  stamp.revision = revision;
  return stamp;
}

struct TraceCounts final {
  std::uint64_t cache_hits = 0;
  std::uint64_t cache_misses = 0;
  std::uint64_t state_commits = 0;
  std::uint64_t transaction_commits = 0;
  std::uint64_t transaction_aborts = 0;
  std::uint64_t output_commits = 0;
  std::uint64_t resets = 0;
};

void Trace(void* context, const vlaforge::runtime::TraceEvent* event) {
  auto* counts = static_cast<TraceCounts*>(context);
  using vlaforge::runtime::TraceKind;
  switch (event->kind) {
    case TraceKind::kCacheHit:
      ++counts->cache_hits;
      break;
    case TraceKind::kCacheMiss:
      ++counts->cache_misses;
      break;
    case TraceKind::kStateCommit:
      ++counts->state_commits;
      break;
    case TraceKind::kTransactionCommit:
      ++counts->transaction_commits;
      break;
    case TraceKind::kTransactionAbort:
      ++counts->transaction_aborts;
      break;
    case TraceKind::kOutputGroupCommit:
      ++counts->output_commits;
      break;
    case TraceKind::kReset:
      ++counts->resets;
      break;
    default:
      break;
  }
}

struct Inputs final {
  DeviceBuffer<float> camera{3u * 256u * 1024u};
  DeviceBuffer<float> lidar{256u * 256u};
  DeviceBuffer<float> status{8u};
  DeviceBuffer<float> noise{20u * 8u * 2u};
  std::vector<float> host_camera;
  std::vector<float> host_lidar;
  std::vector<float> host_status;
  std::vector<float> host_noise;
  VLAForgeBoundTensor camera_view{};
  VLAForgeBoundTensor lidar_view{};
  VLAForgeBoundTensor status_view{};
  VLAForgeBoundTensor noise_view{};

  bool Read(const std::string& root, const char* name,
            std::size_t count, std::vector<float>* output) {
    output->resize(count);
    const std::string path = root + "/" + name + ".bin";
    std::ifstream stream(path, std::ios::binary);
    if (!stream.good()) {
      std::fprintf(stderr, "cannot open input fixture: %s\n", path.c_str());
      return false;
    }
    const auto size = static_cast<std::streamsize>(count * sizeof(float));
    stream.read(reinterpret_cast<char*>(output->data()), size);
    if (stream.gcount() != size ||
        stream.peek() != std::ifstream::traits_type::eof()) {
      std::fprintf(stderr, "input fixture size mismatch: %s\n", path.c_str());
      return false;
    }
    return true;
  }

  bool Initialize(const std::string& root) {
    if (!camera.ok() || !lidar.ok() || !status.ok() || !noise.ok() ||
        !Read(root, "camera_feature", 3u * 256u * 1024u, &host_camera) ||
        !Read(root, "lidar_feature", 256u * 256u, &host_lidar) ||
        !Read(root, "status_feature", 8u, &host_status) ||
        !Read(root, "noise", 20u * 8u * 2u, &host_noise) ||
        !camera.Upload(host_camera.data()) ||
        !lidar.Upload(host_lidar.data()) ||
        !status.Upload(host_status.data()) ||
        !noise.Upload(host_noise.data())) {
      return false;
    }
    static constexpr std::int64_t kCameraShape[] = {1, 3, 256, 1024};
    static constexpr std::int64_t kLidarShape[] = {1, 1, 256, 256};
    static constexpr std::int64_t kStatusShape[] = {1, 8};
    static constexpr std::int64_t kNoiseShape[] = {1, 20, 8, 2};
    camera_view = Tensor(
        camera.data(), camera.bytes(), kCameraShape, 4u);
    lidar_view = Tensor(
        lidar.data(), lidar.bytes(), kLidarShape, 4u);
    status_view = Tensor(
        status.data(), status.bytes(), kStatusShape, 2u);
    noise_view = Tensor(
        noise.data(), noise.bytes(), kNoiseShape, 4u);
    return true;
  }

  bool SetNoiseFirst(float value) {
    host_noise[0] = value;
    return noise.Upload(host_noise.data());
  }

  vlaforge_generated::ModelInputs Bind(
      std::uint64_t revision, bool revision_present = true) const {
    vlaforge_generated::ModelInputs result{};
    const auto stamp = Stamp(revision, revision_present);
    result.camera_feature = camera_view;
    result.camera_feature_stamp = stamp;
    result.lidar_feature = lidar_view;
    result.lidar_feature_stamp = stamp;
    result.status_feature = status_view;
    result.status_feature_stamp = stamp;
    result.noise = noise_view;
    result.noise_stamp = stamp;
    return result;
  }
};

using OutputSet = std::array<std::vector<std::uint8_t>, 6>;
constexpr std::array<std::size_t, 6> kOutputBytes = {
    20u * 8u * 3u * sizeof(float),
    20u * sizeof(float),
    8u * 3u * sizeof(float),
    7u * 128u * 256u * sizeof(float),
    30u * 5u * sizeof(float),
    30u * sizeof(float)};
constexpr std::array<const char*, 6> kOutputNames = {
    "candidate_trajectories", "candidate_scores", "trajectory",
    "bev_semantic_map", "agent_states", "agent_labels"};

bool ReadOutput(const VLAForgeBoundTensor& output, std::size_t expected,
                std::vector<std::uint8_t>* bytes) {
  if (output.tensor.device.kind != VLAFORGE_DEVICE_CUDA ||
      output.tensor.dtype != VLAFORGE_DTYPE_F32 ||
      output.tensor.size_bytes != expected) {
    return false;
  }
  bytes->resize(expected);
  return Cuda(
      cudaMemcpy(bytes->data(), output.tensor.data, expected,
                 cudaMemcpyDeviceToHost),
      "cudaMemcpy output D2H");
}

bool ReadTyped(const vlaforge_generated::ModelOutputs& outputs,
               OutputSet* result) {
  return ReadOutput(
             outputs.candidate_trajectories, kOutputBytes[0],
             &(*result)[0]) &&
         ReadOutput(
             outputs.candidate_scores, kOutputBytes[1], &(*result)[1]) &&
         ReadOutput(outputs.trajectory, kOutputBytes[2], &(*result)[2]) &&
         ReadOutput(
             outputs.bev_semantic_map, kOutputBytes[3], &(*result)[3]) &&
         ReadOutput(outputs.agent_states, kOutputBytes[4], &(*result)[4]) &&
         ReadOutput(outputs.agent_labels, kOutputBytes[5], &(*result)[5]);
}

bool Equal(const OutputSet& left, const OutputSet& right) {
  return left == right;
}

bool WriteOutputs(const std::string& root, const OutputSet& outputs) {
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const std::string path =
        root + "/" + kOutputNames[index] + ".bin";
    std::ofstream stream(path, std::ios::binary);
    if (!stream.good()) {
      return false;
    }
    stream.write(
        reinterpret_cast<const char*>(outputs[index].data()),
        static_cast<std::streamsize>(outputs[index].size()));
    if (!stream.good()) {
      return false;
    }
  }
  return true;
}

bool RunTyped(vlaforge_generated::ModelSession* session,
              const vlaforge_generated::ModelInputs& inputs,
              OutputSet* result) {
  vlaforge_generated::ModelOutputs outputs{};
  const auto status = session->Run(inputs, &outputs);
  if (!status.ok()) {
    std::fprintf(stderr, "typed Run failed: %s\n", status.message);
    return false;
  }
  return ReadTyped(outputs, result);
}

bool BindGeneric(const VLAForgeSessionApi* api, VLAForgeSession* session,
                 const Inputs& inputs, const VLAForgeInputStamp& stamp) {
  return api->bind_tensor(
             session, 0u, &inputs.camera_view, &stamp).code ==
             VLAFORGE_STATUS_OK &&
         api->bind_tensor(
             session, 1u, &inputs.lidar_view, &stamp).code ==
             VLAFORGE_STATUS_OK &&
         api->bind_tensor(
             session, 2u, &inputs.status_view, &stamp).code ==
             VLAFORGE_STATUS_OK &&
         api->bind_tensor(
             session, 3u, &inputs.noise_view, &stamp).code ==
             VLAFORGE_STATUS_OK;
}

bool ReadGeneric(const VLAForgeSessionApi* api, VLAForgeSession* session,
                 OutputSet* result) {
  for (std::uint32_t index = 0; index < result->size(); ++index) {
    VLAForgeBoundTensor output{};
    if (api->read_output_tensor(session, index, &output).code !=
            VLAFORGE_STATUS_OK ||
        !ReadOutput(output, kOutputBytes[index], &(*result)[index])) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(
        stderr, "usage: %s BUNDLE_ROOT INPUT_ROOT OUTPUT_ROOT\n", argv[0]);
    return 2;
  }
  Inputs inputs;
  if (!inputs.Initialize(argv[2])) {
    return 3;
  }

  vlaforge_generated::ModelSession typed(argv[1]);
  if (!typed.initialization_status().ok()) {
    std::fprintf(stderr, "typed Session initialization failed: %s\n",
                 typed.initialization_status().message);
    return 4;
  }
  TraceCounts trace{};
  typed.SetTraceSink({&trace, &Trace});
  OutputSet baseline;
  OutputSet current;
  if (!RunTyped(&typed, inputs.Bind(100u), &baseline) ||
      !RunTyped(&typed, inputs.Bind(100u), &current) ||
      !Equal(baseline, current) ||
      !RunTyped(&typed, inputs.Bind(101u), &current) ||
      !Equal(baseline, current) ||
      !RunTyped(&typed, inputs.Bind(0u, false), &current) ||
      !Equal(baseline, current) ||
      !typed.ResetEpisode(1u).ok() ||
      !RunTyped(&typed, inputs.Bind(100u), &current) ||
      !Equal(baseline, current) ||
      !WriteOutputs(argv[3], baseline)) {
    return 5;
  }
  std::printf(
      "TRACE_SUMMARY,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
      static_cast<unsigned long long>(trace.cache_hits),
      static_cast<unsigned long long>(trace.cache_misses),
      static_cast<unsigned long long>(trace.state_commits),
      static_cast<unsigned long long>(trace.transaction_commits),
      static_cast<unsigned long long>(trace.transaction_aborts),
      static_cast<unsigned long long>(trace.output_commits),
      static_cast<unsigned long long>(trace.resets));
  if (trace.cache_hits != 1u || trace.cache_misses != 4u ||
      trace.state_commits != 0u || trace.transaction_commits != 5u ||
      trace.transaction_aborts != 0u || trace.output_commits != 5u ||
      trace.resets != 1u) {
    return 6;
  }

  VLAForgeSession* generic = nullptr;
  const std::string bundle_root(argv[1]);
  if (vlaforge_model_session_create_from_bundle(
          bundle_root.data(), bundle_root.size(), &generic).code !=
      VLAFORGE_STATUS_OK) {
    return 7;
  }
  const auto* api = vlaforge_model_session_api();
  if (vlaforge_session_api_validate(
          api, vlaforge_generated::kSchemaDigest,
          VLAFORGE_SCHEMA_DIGEST_HEX_SIZE).code != VLAFORGE_STATUS_OK) {
    api->destroy(generic);
    return 8;
  }
  const auto generic_stamp = Stamp(100u);
  OutputSet generic_outputs;
  if (!BindGeneric(api, generic, inputs, generic_stamp) ||
      api->run(generic).code != VLAFORGE_STATUS_OK ||
      !ReadGeneric(api, generic, &generic_outputs)) {
    api->destroy(generic);
    return 9;
  }
  api->destroy(generic);
  const bool typed_generic_equal = Equal(baseline, generic_outputs);
  std::printf(
      "TYPED_GENERIC,%u\n", typed_generic_equal ? 1u : 0u);
  if (!typed_generic_equal) {
    return 10;
  }

  vlaforge_generated::ModelSession failure(argv[1]);
  if (!failure.initialization_status().ok()) {
    return 11;
  }
  TraceCounts failure_trace{};
  failure.SetTraceSink({&failure_trace, &Trace});
  const float original_noise = inputs.host_noise[0];
  if (!inputs.SetNoiseFirst(std::numeric_limits<float>::quiet_NaN())) {
    return 12;
  }
  vlaforge_generated::ModelOutputs failed_outputs{};
  const auto failure_status =
      failure.Run(inputs.Bind(300u), &failed_outputs);
  VLAForgeBoundTensor unavailable{};
  const auto unavailable_status =
      failure.ReadOutputTensor(0u, &unavailable);
  if (failure_status.code !=
          vlaforge::runtime::StatusCode::kValidationFailed ||
      unavailable_status.ok()) {
    return 13;
  }
  if (!inputs.SetNoiseFirst(original_noise) ||
      !RunTyped(&failure, inputs.Bind(300u), &current) ||
      !Equal(baseline, current)) {
    return 14;
  }
  std::printf(
      "FAILURE_SUMMARY,%u,%llu,%llu,%llu,%llu,%llu,%llu\n",
      static_cast<unsigned>(failure_status.code),
      static_cast<unsigned long long>(failure_trace.cache_hits),
      static_cast<unsigned long long>(failure_trace.cache_misses),
      static_cast<unsigned long long>(failure_trace.state_commits),
      static_cast<unsigned long long>(failure_trace.transaction_commits),
      static_cast<unsigned long long>(failure_trace.transaction_aborts),
      static_cast<unsigned long long>(failure_trace.output_commits));
  if (failure_trace.cache_hits != 1u ||
      failure_trace.cache_misses != 1u ||
      failure_trace.state_commits != 0u ||
      failure_trace.transaction_commits != 1u ||
      failure_trace.transaction_aborts != 1u ||
      failure_trace.output_commits != 1u) {
    return 15;
  }
  std::printf("OUTPUT_PARITY,1\n");
  return Cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ? 0 : 16;
}
"""
    return textwrap.dedent(source).lstrip()


def _parse_runner_output(text: str) -> dict[str, object]:
    trace: list[int] | None = None
    failure: list[int] | None = None
    typed_generic = False
    output_parity = False
    for line in text.splitlines():
        fields = line.split(",")
        if fields[0] == "TRACE_SUMMARY":
            trace = [int(item) for item in fields[1:]]
        elif fields[0] == "FAILURE_SUMMARY":
            failure = [int(item) for item in fields[1:]]
        elif fields[0] == "TYPED_GENERIC":
            typed_generic = fields[1:] == ["1"]
        elif fields[0] == "OUTPUT_PARITY":
            output_parity = fields[1:] == ["1"]
    if trace != [1, 4, 0, 5, 0, 5, 1]:
        raise RuntimeError(f"generated runner trace mismatch: {trace}")
    if failure != [7, 1, 1, 0, 1, 1, 1]:
        raise RuntimeError(f"generated runner failure mismatch: {failure}")
    if not typed_generic or not output_parity:
        raise RuntimeError("generated runner output parity failed")
    return {
        "trace_summary": trace,
        "failure_summary": failure,
        "typed_generic_equal": typed_generic,
        "output_parity": output_parity,
    }


def _read_input(
    torch: Any,
    root: Path,
    name: str,
    shape: tuple[int, ...],
) -> Any:
    payload = bytearray((root / f"{name}.bin").read_bytes())
    return (
        torch.frombuffer(payload, dtype=torch.float32)
        .clone()
        .reshape(shape)
        .to("cuda:0")
    )


def _direct_artifact_outputs(
    l3_root: Path,
    input_root: Path,
) -> dict[str, Any]:
    import torch
    import torch._inductor.codecache  # noqa: F401

    callables = {
        name: torch._inductor.aoti_load_package(
            str(l3_root / "artifacts" / f"{name}.pt2")
        )
        for name in DIFFUSIONDRIVE_REGIONS
    }
    values = {
        "camera_feature": _read_input(
            torch, input_root, "camera_feature", (1, 3, 256, 1024)
        ),
        "lidar_feature": _read_input(
            torch, input_root, "lidar_feature", (1, 1, 256, 256)
        ),
        "status_feature": _read_input(
            torch, input_root, "status_feature", (1, 8)
        ),
        "noise": _read_input(
            torch, input_root, "noise", (1, 20, 8, 2)
        ),
    }

    def outputs(value: Any) -> tuple[Any, ...]:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return (value,)

    with torch.inference_mode():
        condition = outputs(
            callables["condition_encoder"](
                values["camera_feature"],
                values["lidar_feature"],
                values["status_feature"],
            )
        )
        state = outputs(
            callables["initialize_planner_state"](values["noise"])
        )[0]
        for step in range(2):
            timestep = outputs(
                callables["make_denoise_timestep"](
                    torch.tensor(step, dtype=torch.int64)
                )
            )[0]
            state = outputs(
                callables["denoise_planner_step"](
                    state,
                    timestep,
                    condition[0],
                    condition[1],
                    condition[2],
                    condition[3],
                )
            )[0]
        candidates, scores, trajectory = outputs(
            callables["decode_planner_outputs"](state)
        )
    torch.cuda.synchronize()
    return {
        "candidate_trajectories": candidates.detach().cpu(),
        "candidate_scores": scores.detach().cpu(),
        "trajectory": trajectory.detach().cpu(),
        "bev_semantic_map": condition[4].detach().cpu(),
        "agent_states": condition[5].detach().cpu(),
        "agent_labels": condition[6].detach().cpu(),
    }


def _cpp_output_metrics(
    direct: dict[str, Any],
    output_root: Path,
) -> dict[str, dict[str, object]]:
    import torch

    result = {}
    for name, shape in _OUTPUTS:
        payload = bytearray((output_root / f"{name}.bin").read_bytes())
        actual = (
            torch.frombuffer(payload, dtype=torch.float32)
            .clone()
            .reshape(shape)
        )
        expected = direct[name]
        difference = (
            expected.to(torch.float64) - actual.to(torch.float64)
        ).abs()
        result[name] = {
            "shape": list(shape),
            "maximum_absolute_error": float(difference.max().item()),
            "mean_absolute_error": float(difference.mean().item()),
            "exact": bool(torch.equal(expected, actual)),
        }
    return result


def _audit(
    *,
    l3_root: Path,
    support_root: Path,
    bundle_root: Path,
    checkpoint: Path,
    upstream_revision: str,
    checkpoint_revision: str,
    target: str,
    reuse_bundle: bool,
) -> dict[str, object]:
    import torch

    if upstream_revision != DIFFUSIONDRIVE_UPSTREAM_REVISION:
        raise ValueError("DiffusionDrive upstream revision is not pinned")
    if checkpoint_revision != DIFFUSIONDRIVE_HF_REVISION:
        raise ValueError("DiffusionDrive checkpoint revision is not pinned")
    if not torch.cuda.is_available():
        raise RuntimeError("DiffusionDrive L4 requires CUDA")
    major, minor = torch.cuda.get_device_capability(0)
    actual_target = f"sm_{major}{minor}"
    if actual_target != target:
        raise RuntimeError(
            f"requested {target}, current device is {actual_target}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != DIFFUSIONDRIVE_CHECKPOINT_SHA256:
        raise ValueError("DiffusionDrive checkpoint digest mismatch")
    frontend, l3_evidence = _verify_l3(l3_root, target=target)
    bundle_present = (
        bundle_root.is_dir()
        and (bundle_root / "bundle.json").is_file()
    )
    if bundle_root.exists() and any(bundle_root.iterdir()) and not (
        reuse_bundle and bundle_present
    ):
        raise ValueError(f"bundle output must be empty: {bundle_root}")

    input_root = support_root / "inputs"
    output_root = support_root / "cpp_outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    input_records = _write_input_fixtures(input_root)
    module = build_real_diffusiondrive_program()
    semantic_path = support_root / "diffusiondrive_real_l4.ir.json"
    semantic_path.write_text(
        canonical_json(module, indent=2) + "\n",
        encoding="utf-8",
    )
    backend_variant = f"torch-{torch.__version__}"
    contracts, sources, artifact_records = _artifact_contracts(
        module,
        l3_root=l3_root,
        checkpoint_sha256=checkpoint_sha256,
        upstream_revision=upstream_revision,
        target=target,
        backend_variant=backend_variant,
    )
    for name, contract in contracts.items():
        _write_json(
            support_root / "contracts" / f"{name}.artifact.json",
            contract.to_dict(),
        )

    runner_source = _runner_source()
    validators = {
        "finite_planner_state": CppValidatorDefinition(
            "finite_planner_state",
            """if (data == nullptr || size_bytes != 820u * sizeof(float)) {
  return false;
}
const auto* values = static_cast<const float*>(data);
for (std::size_t index = 0; index < 820u; ++index) {
  if (!std::isfinite(values[index])) {
    return false;
  }
}
return true;""",
        )
    }
    revision = _git(["rev-parse", "HEAD"])
    dirty = bool(
        _git(["status", "--porcelain", "--untracked-files=no"])
    )
    if reuse_bundle and bundle_present:
        manifest = load_bundle_manifest(bundle_root / "bundle.json")
        manifest.verify_files(bundle_root)
        bundled_runner = bundle_root / "generated" / "runner.cpp"
        if (
            manifest.io_schema_digest != io_schema_digest(module)
            or tuple(
                artifact.region_name
                for artifact in manifest.region_artifacts
            )
            != DIFFUSIONDRIVE_REGIONS
            or not bundled_runner.is_file()
            or bundled_runner.read_text(encoding="utf-8")
            != runner_source
        ):
            raise ValueError(
                "existing bundle does not match real DiffusionDrive L4"
            )
        build_seconds: float | None = None
    else:
        build_started = time.perf_counter()
        manifest = build_artifact_compile_bundle(
            module,
            bundle_root,
            region_artifacts=contracts,
            artifact_sources=sources,
            validators=validators,
            runner_source=runner_source,
            runtime_root=_SOURCE_ROOT,
            cmake_prefix_path=torch.utils.cmake_prefix_path,
            backend_versions={
                "aoti": backend_variant,
                "cuda": str(torch.version.cuda),
            },
            profile="verified",
            source_revision=revision,
            source_dirty=dirty,
            environment={"TORCH_CUDA_ARCH_LIST": "8.6"},
            default_device="cuda:0",
            state_device="cuda:0",
        )
        build_seconds = time.perf_counter() - build_started

    runner = bundle_root / "bin" / "vlaforge_generated_runner"
    clean_environment = {
        **dict(os.environ),
        "PYTHONHOME": "/definitely/not/a/python/home",
        "PYTHONPATH": "/definitely/not/a/python/path",
    }
    run_started = time.perf_counter()
    completed = _run(
        [
            str(runner),
            str(bundle_root),
            str(input_root),
            str(output_root),
        ],
        environment=clean_environment,
    )
    run_seconds = time.perf_counter() - run_started
    stdout_path = support_root / "diffusiondrive_real_l4_runner.stdout"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    parsed = _parse_runner_output(completed.stdout)

    direct = _direct_artifact_outputs(l3_root, input_root)
    parity = _cpp_output_metrics(direct, output_root)
    if not all(item["exact"] for item in parity.values()):
        raise RuntimeError(
            "generated C++ Session differs from direct AOTI outputs"
        )
    linked = _run(["ldd", str(runner)]).stdout
    if "libpython" in linked.lower():
        raise RuntimeError("generated DiffusionDrive Session links libpython")
    linked_path = support_root / "diffusiondrive_real_l4_runner.ldd"
    linked_path.write_text(linked, encoding="utf-8")
    manifest.verify_files(bundle_root)

    certificate = manifest.compilation_certificate
    arena = certificate.arena
    memory_plan = _json(
        bundle_root / "metadata" / "physical_memory_plan.json"
    )
    derived_cache_bytes = sum(
        int(item["size_bytes"])
        for item in memory_plan["arena"]["physical_buffers"]
        if item["buffer_class"] == "derived_cache"
    )
    return {
        "schema": _REPORT_SCHEMA,
        "status": "passed",
        "passed": True,
        "evidence_kind": "real-checkpoint-generated-session-parity",
        "evidence_level": "L4",
        "model": "DiffusionDrive NAVSIM 88.1 PDMS",
        "license": {
            "upstream_source": "MIT",
            "checkpoint": "non-commercial per Hugging Face model card",
        },
        "upstream_revision": upstream_revision,
        "checkpoint": {
            "revision": checkpoint_revision,
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint.stat().st_size,
        },
        "repository": {
            "revision": revision,
            "source_dirty": dirty,
        },
        "environment": {
            "host": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "target": target,
        },
        "semantic_ir": {
            "path": str(semantic_path),
            "io_schema_digest": io_schema_digest(module),
            "regions": list(DIFFUSIONDRIVE_REGIONS),
            "core_op_delta": 0,
            "adapter_template": "DiffusionPlanner",
            "persistent_state_slots": 0,
            "bounded_denoise_steps": 2,
        },
        "deterministic_inputs": {
            "manifest": str(input_root / "inputs.json"),
            "values": input_records,
        },
        "artifacts": artifact_records,
        "artifact_total_bytes": sum(
            int(item["artifact_size_bytes"])
            for item in artifact_records
        ),
        "bundle": {
            "path": str(bundle_root),
            "manifest": str(bundle_root / "bundle.json"),
            "digest": manifest.digest(),
            "build_seconds": build_seconds,
            "reused_existing_verified_bundle": (
                reuse_bundle and bundle_present
            ),
            "runner": str(runner),
            "runner_sha256": _sha256(runner),
            "bundle_verified": True,
            "invalid_python_environment": True,
            "links_libpython": False,
            "linked_dependencies": str(linked_path),
        },
        "correctness": {
            "direct_artifact_vs_generated_outputs": parity,
            "all_named_outputs_exact": True,
            "typed_generic_equal": parsed["typed_generic_equal"],
        },
        "cache": {
            "key_inputs": [
                "camera_feature",
                "lidar_feature",
                "status_feature",
            ],
            "key_states": [],
            "hits": parsed["trace_summary"][0],
            "misses": parsed["trace_summary"][1],
            "same_revision_hit": True,
            "new_revision_miss": True,
            "missing_revision_miss": True,
            "episode_reset_invalidates": True,
        },
        "transaction": {
            "state_commits": parsed["trace_summary"][2],
            "successful_transactions": parsed["trace_summary"][3],
            "aborts_in_success_sequence": parsed["trace_summary"][4],
            "committed_output_groups": parsed["trace_summary"][5],
            "resets": parsed["trace_summary"][6],
            "validation_failure_status_code": parsed["failure_summary"][0],
            "failure_retry_cache_hits": parsed["failure_summary"][1],
            "failure_retry_cache_misses": parsed["failure_summary"][2],
            "failure_retry_state_commits": parsed["failure_summary"][3],
            "failure_retry_transaction_commits": (
                parsed["failure_summary"][4]
            ),
            "failure_retry_transaction_aborts": (
                parsed["failure_summary"][5]
            ),
            "failure_retry_output_commits": parsed["failure_summary"][6],
            "failure_exposed_no_uncommitted_output": True,
        },
        "memory": {
            "arena_baseline_bytes": arena.baseline_bytes,
            "arena_compiled_bytes": arena.compiled_bytes,
            "arena_saved_bytes": arena.saved_bytes,
            "derived_cache_bytes": derived_cache_bytes,
            "authoritative_state_bytes": 0,
        },
        "execution": {
            "runner_seconds": run_seconds,
            "successful_typed_runs": 6,
            "successful_generic_runs": 1,
            "failure_injection_runs": 1,
            "stdout": str(stdout_path),
        },
        "l3_chain": {
            "frontend_report": str(l3_root / "frontend.json"),
            "artifact_report": str(l3_root / "artifact-l3.json"),
            "frontend_report_sha256": _sha256(
                l3_root / "frontend.json"
            ),
            "artifact_report_sha256": _sha256(
                l3_root / "artifact-l3.json"
            ),
            "artifact_evidence_level": l3_evidence["evidence_level"],
            "frontend_evidence_level": frontend["evidence_level"],
        },
        "reproduction": {
            "command": [
                sys.executable,
                "vlaforge/tools/build_real_diffusiondrive_l4.py",
                "--l3-root",
                str(l3_root),
                "--support-root",
                str(support_root),
                "--bundle-root",
                str(bundle_root),
                "--checkpoint",
                str(checkpoint),
                "--upstream-revision",
                upstream_revision,
                "--checkpoint-revision",
                checkpoint_revision,
                "--target",
                target,
                "--report",
                "<report.json>",
            ]
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l3-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--upstream-revision",
        default=DIFFUSIONDRIVE_UPSTREAM_REVISION,
    )
    parser.add_argument(
        "--checkpoint-revision",
        default=DIFFUSIONDRIVE_HF_REVISION,
    )
    parser.add_argument("--target", default="sm_86")
    parser.add_argument("--reuse-bundle", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = _audit(
        l3_root=args.l3_root.resolve(),
        support_root=args.support_root.resolve(),
        bundle_root=args.bundle_root.resolve(),
        checkpoint=args.checkpoint.resolve(),
        upstream_revision=args.upstream_revision,
        checkpoint_revision=args.checkpoint_revision,
        target=args.target,
        reuse_bundle=args.reuse_bundle,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report is not None:
        _write_json(args.report.resolve(), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
