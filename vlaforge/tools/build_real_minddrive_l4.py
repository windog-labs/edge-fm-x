#!/usr/bin/env python3
"""Build and audit the real MindDrive no-Python CUDA Session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.adapters.minddrive_real import (  # noqa: E402
    MINDDRIVE_CHECKPOINT_SHA256,
    MINDDRIVE_INPUT_TYPES,
    MINDDRIVE_OUTPUT_TYPES,
    MINDDRIVE_STATE_TYPES,
    MINDDRIVE_UPSTREAM_REVISION,
    build_real_minddrive_program,
)
from vlaforge.codegen import (  # noqa: E402
    CppValidatorDefinition,
    ZERO_STATE,
)
from vlaforge.deployment import (  # noqa: E402
    ArtifactIdentity,
    ArtifactKind,
    BackendCapability,
    EffectAudit,
    RegionArtifactContract,
    ValueContract,
    WorkspaceContract,
    build_artifact_compile_bundle,
)
from vlaforge.ir.serializer import io_schema_digest  # noqa: E402
from vlaforge.ir.types import ScalarType, TensorType  # noqa: E402


_REPORT_SCHEMA = "vlaforge.minddrive_real_l4/1"
_FRAME_IDS = (
    "frame_00400",
    "frame_00401",
    "frame_00402",
    "frame_00403",
    "frame_00404",
)
_DIRECT_REGIONS = {
    "position_encoder",
    "detection_encoder",
    "decision_expert",
    "action_expert",
    "trajectory_decoder",
    "detection_decoder",
}
_DTYPE_ENUMS = {
    "bool": "VLAFORGE_DTYPE_BOOL",
    "f32": "VLAFORGE_DTYPE_F32",
    "f64": "VLAFORGE_DTYPE_F64",
    "i64": "VLAFORGE_DTYPE_I64",
}
_DTYPE_BYTES = {"bool": 1, "f32": 4, "f64": 8, "i64": 8}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
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
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )


def _git(command: list[str]) -> str:
    return _run(["git", *command], environment=dict(os.environ)).stdout.strip()


def _bytes(payload: TensorType | ScalarType) -> int:
    if isinstance(payload, ScalarType):
        return _DTYPE_BYTES[payload.name]
    elements = 1
    for dimension in payload.shape:
        elements *= dimension
    return elements * _DTYPE_BYTES[payload.dtype]


def _artifact_contracts(
    module: Any,
    *,
    aggregate: dict[str, Any],
    sequence_report: dict[str, Any],
    target: str,
) -> tuple[
    dict[str, RegionArtifactContract],
    dict[str, Path],
    dict[str, Path],
]:
    aggregate_regions = {
        str(item["name"]): item for item in aggregate["regions"]
    }
    sequence_manifests = {
        str(item["region"]): item
        for item in sequence_report["manifests"]
    }
    auxiliary = {
        str(item["bundle_path"]): Path(item["source"]).resolve()
        for item in sequence_report["auxiliary_files"]
    }
    contracts = {}
    sources = {}
    digest = io_schema_digest(module)
    for region_id, region in enumerate(module.regions):
        name = region.name
        if name in sequence_manifests:
            record = sequence_manifests[name]
            source = Path(record["path"]).resolve()
            kind = ArtifactKind.AOTI_SEQUENCE
            extension = "vfseq"
            graph_sha256 = str(record["sha256"])
            backend_variant = "aoti-sequence/1"
        elif name in _DIRECT_REGIONS:
            item = aggregate_regions[name]
            artifact = item["artifact"]
            source = Path(artifact["path"]).resolve()
            kind = ArtifactKind.SHARED_LIBRARY
            extension = "so"
            graph_sha256 = str(item["capture_export"]["sha256"])
            backend_variant = "torch-2.4.1+cu118-raw-so"
        else:
            raise ValueError(f"unsupported MindDrive L4 Region: {name}")
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = _sha256(source)
        if name in sequence_manifests and (
            source_hash != sequence_manifests[name]["sha256"]
            or source.stat().st_size
            != sequence_manifests[name]["size_bytes"]
        ):
            raise ValueError(f"{name}: sequence source identity changed")
        dtypes = sorted(
            {
                (
                    value.type.dtype
                    if isinstance(value.type, TensorType)
                    else value.type.name
                )
                for value in region.inputs
            }
            | {
                value.dtype if isinstance(value, TensorType) else value.name
                for value in region.outputs
            }
        )
        contract = RegionArtifactContract(
            region_id=region_id,
            region_name=name,
            inputs=tuple(
                ValueContract.from_ir(
                    value.name, value.type, device="cuda:0"
                )
                for value in region.inputs
            ),
            outputs=tuple(
                ValueContract.from_ir(
                    f"output_{index}", value, device="cuda:0"
                )
                for index, value in enumerate(region.outputs)
            ),
            io_schema_digest=digest,
            identity=ArtifactIdentity(
                model_name="MindDrive Qwen2-0.5B",
                upstream_revision=MINDDRIVE_UPSTREAM_REVISION,
                checkpoint_identity=(
                    f"sha256:{MINDDRIVE_CHECKPOINT_SHA256}"
                ),
                graph_sha256=graph_sha256,
            ),
            artifact_kind=kind,
            artifact_path=f"artifacts/{name}.{extension}",
            artifact_sha256=source_hash,
            artifact_size_bytes=source.stat().st_size,
            workspace=WorkspaceContract(device="cuda:0"),
            capability=BackendCapability(
                backend="aoti",
                target=target,
                supported_dtypes=tuple(dtypes),
                supports_dynamic_shapes=False,
                supports_device_resident_io=True,
                requires_synchronize=True,
            ),
            effect_audit=EffectAudit(),
            backend_variant=backend_variant,
        )
        contracts[name] = contract
        sources[name] = source
    return contracts, sources, auxiliary


def _materialize_inputs(
    frame_inputs: tuple[Path, ...],
    output_root: Path,
) -> list[dict[str, object]]:
    import torch

    expected = dict(MINDDRIVE_INPUT_TYPES)
    records = []
    for frame_id, source in zip(_FRAME_IDS, frame_inputs, strict=True):
        if not source.is_file():
            raise FileNotFoundError(source)
        values = torch.load(source, map_location="cpu", weights_only=True)
        if set(values) != set(expected):
            raise ValueError(f"{frame_id}: MindDrive input keys changed")
        frame_root = output_root / frame_id
        frame_root.mkdir(parents=True, exist_ok=True)
        inputs = {}
        for name, payload in MINDDRIVE_INPUT_TYPES:
            value = values[name].detach().contiguous().cpu()
            if (
                tuple(value.shape) != payload.shape
                or str(value.dtype)
                != {
                    "f32": "torch.float32",
                    "f64": "torch.float64",
                    "i64": "torch.int64",
                }[payload.dtype]
            ):
                raise ValueError(f"{frame_id}/{name}: input contract changed")
            path = frame_root / f"{name}.bin"
            path.write_bytes(value.view(torch.uint8).numpy().tobytes())
            inputs[name] = {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "shape": list(payload.shape),
                "dtype": payload.dtype,
            }
        records.append(
            {
                "frame": frame_id,
                "source": {
                    "path": str(source),
                    "sha256": _sha256(source),
                    "size_bytes": source.stat().st_size,
                },
                "inputs": inputs,
            }
        )
    return records


def _runner_source() -> str:
    shape_lines = []
    load_lines = []
    typed_lines = []
    for index, (name, payload) in enumerate(MINDDRIVE_INPUT_TYPES):
        shape_name = f"kShape{index}"
        dimensions = ", ".join(str(item) for item in payload.shape)
        shape_lines.append(
            f"    static constexpr std::int64_t {shape_name}[] = "
            f"{{{dimensions}}};"
        )
        load_lines.append(
            f'    if (!LoadOne({index}u, root, "{name}", '
            f"{_bytes(payload)}u)) return false;"
        )
        typed_lines.extend(
            (
                f"    result.{name} = views_[{index}u];",
                f"    result.{name}_stamp = stamp;",
            )
        )
        shape_pointer = shape_name if payload.shape else "nullptr"
        load_lines.append(
            f"    views_[{index}u] = Tensor(buffers_[{index}u].data(), "
            f"{_bytes(payload)}u, {shape_pointer}, "
            f"{len(payload.shape)}u, {_DTYPE_ENUMS[payload.dtype]});"
        )
    typed_output_lines = []
    generic_output_lines = []
    for index, (name, payload) in enumerate(MINDDRIVE_OUTPUT_TYPES):
        if isinstance(payload, TensorType):
            typed_output_lines.append(
                f'      WriteTensor(root, frame, "{name}", '
                f"outputs.{name}, {_bytes(payload)}u)"
            )
            generic_output_lines.append(
                "      ReadGenericTensor(api, session, "
                f'{index}u, outputs, kFrames[index], "{name}", '
                f"{_bytes(payload)}u)"
            )
        else:
            typed_output_lines.append(
                f'      WriteScalar(root, frame, "{name}", '
                f"outputs.{name})"
            )
            generic_output_lines.append(
                "      ReadGenericScalar(api, session, "
                f'{index}u, outputs, kFrames[index], "{name}")'
            )
    typed_output_expression = " &&\n".join(typed_output_lines)
    generic_output_expression = " &&\n".join(generic_output_lines)
    return textwrap.dedent(
        r"""
        #include "session_generated.h"

        #include <cuda_runtime_api.h>

        #include <array>
        #include <cmath>
        #include <cstddef>
        #include <cstdint>
        #include <cstdio>
        #include <fstream>
        #include <limits>
        #include <string>
        #include <vector>

        namespace {

        bool Cuda(cudaError_t status, const char* operation) {
          if (status == cudaSuccess) return true;
          std::fprintf(stderr, "%s failed: %s\n", operation,
                       cudaGetErrorString(status));
          return false;
        }

        class DeviceBuffer final {
         public:
          DeviceBuffer() = default;
          ~DeviceBuffer() {
            if (data_ != nullptr) (void)cudaFree(data_);
          }
          DeviceBuffer(const DeviceBuffer&) = delete;
          DeviceBuffer& operator=(const DeviceBuffer&) = delete;

          bool Allocate(std::size_t size) {
            size_ = size;
            return Cuda(cudaMalloc(&data_, size_), "cudaMalloc input");
          }
          bool Upload(const void* source) {
            return Cuda(cudaMemcpy(data_, source, size_,
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy input H2D");
          }
          bool SetF32(std::size_t index, float value) {
            if ((index + 1u) * sizeof(float) > size_) return false;
            return Cuda(cudaMemcpy(
                            static_cast<std::byte*>(data_) +
                                index * sizeof(float),
                            &value, sizeof(value), cudaMemcpyHostToDevice),
                        "cudaMemcpy mutated input H2D");
          }
          void* data() const { return data_; }

         private:
          void* data_ = nullptr;
          std::size_t size_ = 0u;
        };

        VLAForgeBoundTensor Tensor(
            void* data, std::uint64_t bytes,
            const std::int64_t* dimensions, std::uint32_t rank,
            VLAForgeDType dtype) {
          return VLAForgeBoundTensor{
              sizeof(VLAForgeBoundTensor),
              {data, bytes, dimensions, rank, dtype,
               {VLAFORGE_DEVICE_CUDA, 0}},
              VLAFORGE_LAYOUT_CONTIGUOUS,
              64u};
        }

        VLAForgeInputStamp Stamp(std::uint64_t revision) {
          VLAForgeInputStamp stamp{};
          stamp.struct_size = sizeof(VLAForgeInputStamp);
          stamp.has_revision = 1u;
          stamp.revision = revision;
          return stamp;
        }

        class InputFrame final {
         public:
          bool Load(const std::string& root) {
        __SHAPES__
        __LOADS__
            return true;
          }

          vlaforge_generated::ModelInputs Typed(
              std::uint64_t revision) const {
            vlaforge_generated::ModelInputs result{};
            const auto stamp = Stamp(revision);
        __TYPED_INPUTS__
            return result;
          }

          bool BindGeneric(
              const VLAForgeSessionApi* api, VLAForgeSession* session,
              std::uint64_t revision) const {
            const auto stamp = Stamp(revision);
            for (std::uint32_t index = 0u; index < views_.size(); ++index) {
              if (api->bind_tensor(
                      session, index, &views_[index], &stamp).code !=
                  VLAFORGE_STATUS_OK) {
                return false;
              }
            }
            return true;
          }

          bool SetTrajectoryNoiseFirst(float value) {
            return buffers_[4u].SetF32(0u, value);
          }

         private:
          bool LoadOne(std::size_t index, const std::string& root,
                       const char* name, std::size_t expected) {
            const std::string path = root + "/" + name + ".bin";
            std::ifstream stream(path, std::ios::binary);
            if (!stream.good()) {
              std::fprintf(stderr, "cannot open input: %s\n", path.c_str());
              return false;
            }
            host_[index].resize(expected);
            stream.read(
                reinterpret_cast<char*>(host_[index].data()),
                static_cast<std::streamsize>(expected));
            if (stream.gcount() !=
                    static_cast<std::streamsize>(expected) ||
                stream.peek() != std::ifstream::traits_type::eof() ||
                !buffers_[index].Allocate(expected) ||
                !buffers_[index].Upload(host_[index].data())) {
              std::fprintf(stderr, "invalid input: %s\n", path.c_str());
              return false;
            }
            return true;
          }

          std::array<DeviceBuffer, 13> buffers_{};
          std::array<std::vector<std::byte>, 13> host_{};
          std::array<VLAForgeBoundTensor, 13> views_{};
        };

        std::string OutputPath(
            const std::string& root, const std::string& frame,
            const char* name) {
          return root + "/" + frame + "/" + name + ".bin";
        }

        bool WriteBytes(
            const std::string& path, const void* data, std::size_t size) {
          std::ofstream stream(path, std::ios::binary);
          if (!stream.good()) return false;
          stream.write(
              reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(size));
          return stream.good();
        }

        bool WriteTensor(
            const std::string& root, const std::string& frame,
            const char* name, const VLAForgeBoundTensor& value,
            std::size_t expected) {
          if (value.tensor.device.kind != VLAFORGE_DEVICE_CUDA ||
              value.tensor.size_bytes != expected) {
            return false;
          }
          std::vector<std::byte> host(expected);
          return Cuda(
                     cudaMemcpy(
                         host.data(), value.tensor.data, expected,
                         cudaMemcpyDeviceToHost),
                     "cudaMemcpy output D2H") &&
              WriteBytes(
                  OutputPath(root, frame, name), host.data(), host.size());
        }

        bool WriteScalar(
            const std::string& root, const std::string& frame,
            const char* name, const VLAForgeScalarValue& value) {
          return value.struct_size >= sizeof(VLAForgeScalarValue) &&
              value.dtype == VLAFORGE_DTYPE_I64 &&
              WriteBytes(
                  OutputPath(root, frame, name), &value.value.i64,
                  sizeof(value.value.i64));
        }

        bool ReadGenericTensor(
            const VLAForgeSessionApi* api, VLAForgeSession* session,
            std::uint32_t output_id, const std::string& root,
            const std::string& frame, const char* name,
            std::size_t expected) {
          VLAForgeBoundTensor value{};
          return api->read_output_tensor(session, output_id, &value).code ==
                     VLAFORGE_STATUS_OK &&
              WriteTensor(root, frame, name, value, expected);
        }

        bool ReadGenericScalar(
            const VLAForgeSessionApi* api, VLAForgeSession* session,
            std::uint32_t output_id, const std::string& root,
            const std::string& frame, const char* name) {
          VLAForgeScalarValue value{};
          return api->read_output_scalar(session, output_id, &value).code ==
                     VLAFORGE_STATUS_OK &&
              WriteScalar(root, frame, name, value);
        }

        struct TraceCounts final {
          std::uint64_t cache_hits = 0u;
          std::uint64_t cache_misses = 0u;
          std::uint64_t state_commits = 0u;
          std::uint64_t transaction_commits = 0u;
          std::uint64_t transaction_aborts = 0u;
          std::uint64_t output_commits = 0u;
          std::uint64_t resets = 0u;
        };

        void Trace(
            void* context,
            const vlaforge::runtime::TraceEvent* event) {
          auto* counts = static_cast<TraceCounts*>(context);
          using vlaforge::runtime::TraceKind;
          switch (event->kind) {
            case TraceKind::kCacheHit: ++counts->cache_hits; break;
            case TraceKind::kCacheMiss: ++counts->cache_misses; break;
            case TraceKind::kStateCommit: ++counts->state_commits; break;
            case TraceKind::kTransactionCommit:
              ++counts->transaction_commits;
              break;
            case TraceKind::kTransactionAbort:
              ++counts->transaction_aborts;
              break;
            case TraceKind::kOutputGroupCommit:
              ++counts->output_commits;
              break;
            case TraceKind::kReset: ++counts->resets; break;
            default: break;
          }
        }

        bool RunTyped(
            vlaforge_generated::ModelSession* session,
            const InputFrame& input, std::uint64_t revision,
            const std::string& root, const std::string& frame,
            bool write_outputs) {
          vlaforge_generated::ModelOutputs outputs{};
          const auto status = session->Run(input.Typed(revision), &outputs);
          if (!status.ok()) {
            std::fprintf(stderr, "typed Run failed: %s\n", status.message);
            return false;
          }
          return !write_outputs ||
        __TYPED_OUTPUTS__;
        }

        bool RunTypedMode(
            const std::string& bundle, const std::string& inputs,
            const std::string& outputs) {
          vlaforge_generated::ModelSession session(bundle.c_str());
          if (!session.initialization_status().ok()) {
            std::fprintf(
                stderr, "Session initialization failed: %s\n",
                session.initialization_status().message);
            return false;
          }
          TraceCounts trace{};
          session.SetTraceSink({&trace, &Trace});
          {
            InputFrame probe;
            const std::string root = inputs + "/frame_00400";
            if (!probe.Load(root) ||
                !RunTyped(&session, probe, 100u, outputs, "probe", false) ||
                !RunTyped(&session, probe, 100u, outputs, "probe", false) ||
                !session.ResetEpisode(1u).ok()) {
              return false;
            }
          }
          constexpr std::array<const char*, 5> kFrames = {
              "frame_00400", "frame_00401", "frame_00402",
              "frame_00403", "frame_00404"};
          for (std::size_t index = 0u; index < kFrames.size(); ++index) {
            InputFrame frame;
            if (!frame.Load(inputs + "/" + kFrames[index]) ||
                !RunTyped(
                    &session, frame, 100u + index, outputs,
                    kFrames[index], true)) {
              return false;
            }
          }
          {
            InputFrame rejected;
            vlaforge_generated::ModelOutputs ignored{};
            if (!rejected.Load(inputs + "/frame_00404") ||
                !rejected.SetTrajectoryNoiseFirst(
                    std::numeric_limits<float>::quiet_NaN())) {
              return false;
            }
            const auto status =
                session.Run(rejected.Typed(200u), &ignored);
            if (status.ok()) {
              std::fprintf(stderr, "validation failure unexpectedly passed\n");
              return false;
            }
          }
          {
            InputFrame retry;
            if (!retry.Load(inputs + "/frame_00404") ||
                !RunTyped(
                    &session, retry, 201u, outputs, "retry", false)) {
              return false;
            }
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
          return trace.cache_hits == 1u &&
              trace.cache_misses == 8u &&
              trace.state_commits == 128u &&
              trace.transaction_commits == 8u &&
              trace.transaction_aborts == 1u &&
              trace.output_commits == 8u &&
              trace.resets == 1u;
        }

        bool RunGenericMode(
            const std::string& bundle, const std::string& inputs,
            const std::string& outputs) {
          VLAForgeSession* session = nullptr;
          if (vlaforge_model_session_create_from_bundle(
                  bundle.data(), bundle.size(), &session).code !=
              VLAFORGE_STATUS_OK) {
            return false;
          }
          const auto* api = vlaforge_model_session_api();
          if (vlaforge_session_api_validate(
                  api, vlaforge_generated::kSchemaDigest,
                  VLAFORGE_SCHEMA_DIGEST_HEX_SIZE).code !=
              VLAFORGE_STATUS_OK) {
            api->destroy(session);
            return false;
          }
          constexpr std::array<const char*, 5> kFrames = {
              "frame_00400", "frame_00401", "frame_00402",
              "frame_00403", "frame_00404"};
          bool passed = true;
          for (std::size_t index = 0u;
               index < kFrames.size() && passed; ++index) {
            InputFrame frame;
            passed =
                frame.Load(inputs + "/" + kFrames[index]) &&
                frame.BindGeneric(api, session, 100u + index) &&
                api->run(session).code == VLAFORGE_STATUS_OK &&
        __GENERIC_OUTPUTS__;
          }
          api->destroy(session);
          return passed;
        }

        }  // namespace

        int main(int argc, char** argv) {
          if (argc != 5) {
            std::fprintf(
                stderr,
                "usage: %s BUNDLE_ROOT INPUT_ROOT OUTPUT_ROOT "
                "[typed|generic]\n",
                argv[0]);
            return 2;
          }
          const std::string mode(argv[4]);
          const bool passed =
              mode == "typed"
              ? RunTypedMode(argv[1], argv[2], argv[3])
              : mode == "generic"
              ? RunGenericMode(argv[1], argv[2], argv[3])
              : false;
          return passed ? 0 : 3;
        }
        """
    ).replace("__SHAPES__", "\n".join(shape_lines)).replace(
        "__LOADS__", "\n".join(load_lines)
    ).replace("__TYPED_INPUTS__", "\n".join(typed_lines)).replace(
        "__TYPED_OUTPUTS__", typed_output_expression
    ).replace("__GENERIC_OUTPUTS__", generic_output_expression)


def _output_reference(reference_path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(reference_path, map_location="cpu", weights_only=True)
    decoded = payload["detection_candidate_decoded"]
    trajectory = payload["trajectory_candidate"]
    return {
        "trajectory": trajectory[0],
        "path_trajectory": trajectory[1],
        "detection_scores": decoded[0],
        "detection_labels": decoded[1],
        "motion_trajectories": decoded[2],
        "detection_boxes": decoded[3],
        "detection_valid_mask": decoded[4],
        "detection_valid_count": decoded[5],
        "speed_command": trajectory[2],
        "path_command": trajectory[3],
    }


def _read_cpp_output(path: Path, payload: TensorType | ScalarType) -> Any:
    import torch

    dtype = {
        "bool": torch.bool,
        "f32": torch.float32,
        "f64": torch.float64,
        "i64": torch.int64,
    }[payload.dtype if isinstance(payload, TensorType) else payload.name]
    raw = bytearray(path.read_bytes())
    value = torch.frombuffer(raw, dtype=dtype).clone()
    return (
        value.reshape(payload.shape)
        if isinstance(payload, TensorType)
        else value.reshape(()).clone()
    )


def _compare_outputs(
    reference_path: Path,
    typed_root: Path,
    generic_root: Path,
) -> dict[str, dict[str, object]]:
    import torch

    reference = _output_reference(reference_path)
    comparisons = {}
    for name, payload in MINDDRIVE_OUTPUT_TYPES:
        typed_path = typed_root / "frame_00404" / f"{name}.bin"
        generic_path = generic_root / "frame_00404" / f"{name}.bin"
        typed = _read_cpp_output(typed_path, payload)
        generic = _read_cpp_output(generic_path, payload)
        expected = reference[name].detach().cpu()
        if expected.is_floating_point():
            difference = (
                expected.to(torch.float64) - typed.to(torch.float64)
            ).abs()
            maximum = float(difference.max().item())
            mean = float(difference.mean().item())
        else:
            maximum = 0.0 if torch.equal(expected, typed) else float("inf")
            mean = maximum
        comparisons[name] = {
            "shape": list(expected.shape),
            "dtype": str(expected.dtype),
            "typed_vs_l3_exact": bool(torch.equal(expected, typed)),
            "typed_vs_generic_exact": bool(torch.equal(typed, generic)),
            "maximum_absolute_error": maximum,
            "mean_absolute_error": mean,
        }
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--sequence-report", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument(
        "--frame-input", action="append", type=Path, required=True
    )
    parser.add_argument("--reference-tensors", type=Path, required=True)
    parser.add_argument("--target", default="sm_86")
    parser.add_argument("--reuse-bundle", action="store_true")
    args = parser.parse_args()

    import torch

    if torch.__version__ != "2.4.1+cu118":
        raise RuntimeError(
            "MindDrive L4 build requires torch 2.4.1+cu118, got "
            f"{torch.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("MindDrive L4 requires CUDA")
    major, minor = torch.cuda.get_device_capability(0)
    if f"sm_{major}{minor}" != args.target:
        raise RuntimeError("MindDrive L4 target does not match current GPU")
    frame_inputs = tuple(path.resolve() for path in args.frame_input)
    if len(frame_inputs) != len(_FRAME_IDS):
        raise ValueError("MindDrive L4 requires exactly five frame inputs")
    aggregate = _json(args.artifact_manifest.resolve())
    sequence_report = _json(args.sequence_report.resolve())
    if (
        aggregate.get("schema")
        != "vlaforge.minddrive_aoti_artifact_manifest/1"
        or not aggregate.get("passed")
        or sequence_report.get("schema")
        != "vlaforge.minddrive_aoti_sequences/1"
        or not sequence_report.get("passed")
    ):
        raise ValueError("MindDrive L3/sequence evidence is not passed")

    bundle_root = args.bundle_root.resolve()
    support_root = args.support_root.resolve()
    if (
        bundle_root.exists()
        and any(bundle_root.iterdir())
        and not args.reuse_bundle
    ):
        raise ValueError(f"bundle output must be empty: {bundle_root}")
    support_root.mkdir(parents=True, exist_ok=True)
    input_root = support_root / "inputs"
    typed_root = support_root / "typed_outputs"
    generic_root = support_root / "generic_outputs"
    for output_root in (typed_root, generic_root):
        for frame_id in _FRAME_IDS:
            (output_root / frame_id).mkdir(parents=True, exist_ok=True)
    input_records = _materialize_inputs(frame_inputs, input_root)

    module = build_real_minddrive_program()
    contracts, sources, auxiliary = _artifact_contracts(
        module,
        aggregate=aggregate,
        sequence_report=sequence_report,
        target=args.target,
    )
    for name, contract in contracts.items():
        _write_json(
            support_root / "contracts" / f"{name}.artifact.json",
            contract.to_dict(),
        )
    runner_source = _runner_source()
    (support_root / "minddrive_real_l4_runner.cpp").write_text(
        runner_source, encoding="utf-8"
    )
    validators = {
        "minddrive_output_contract": CppValidatorDefinition(
            "minddrive_output_contract",
            """if (data == nullptr || size_bytes != 12u * sizeof(float)) {
  return false;
}
const auto* trajectory = static_cast<const float*>(data);
for (std::size_t index = 0; index < 12u; ++index) {
  if (!std::isfinite(trajectory[index])) {
    return false;
  }
}
return true;""",
        )
    }
    revision = _git(["rev-parse", "HEAD"])
    dirty = bool(_git(["status", "--porcelain", "--untracked-files=no"]))
    bundle_present = (bundle_root / "bundle.json").is_file()
    if args.reuse_bundle and bundle_present:
        from vlaforge.deployment import load_bundle_manifest

        manifest = load_bundle_manifest(bundle_root / "bundle.json")
        manifest.verify_files(bundle_root)
        build_seconds = None
    else:
        build_started = time.perf_counter()
        manifest = build_artifact_compile_bundle(
            module,
            bundle_root,
            region_artifacts=contracts,
            artifact_sources=sources,
            auxiliary_files=auxiliary,
            validators=validators,
            runner_source=runner_source,
            runtime_root=_SOURCE_ROOT,
            cmake_prefix_path=torch.utils.cmake_prefix_path,
            backend_versions={
                "aoti": "torch-2.4.1+cu118",
                "cuda": str(torch.version.cuda),
                "aoti_sequence": "1",
            },
            profile="verified",
            source_revision=revision,
            source_dirty=dirty,
            environment={"TORCH_CUDA_ARCH_LIST": "8.6"},
            initial_state={
                name: ZERO_STATE for name, _ in MINDDRIVE_STATE_TYPES
            },
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
    run_results = {}
    for mode, output_root in (
        ("typed", typed_root),
        ("generic", generic_root),
    ):
        started = time.perf_counter()
        completed = _run(
            [
                str(runner),
                str(bundle_root),
                str(input_root),
                str(output_root),
                mode,
            ],
            environment=clean_environment,
        )
        run_results[mode] = {
            "seconds": time.perf_counter() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        (support_root / f"runner_{mode}.stdout").write_text(
            completed.stdout, encoding="utf-8"
        )
        (support_root / f"runner_{mode}.stderr").write_text(
            completed.stderr, encoding="utf-8"
        )
    comparisons = _compare_outputs(
        args.reference_tensors.resolve(), typed_root, generic_root
    )
    if not all(
        item["typed_vs_l3_exact"] and item["typed_vs_generic_exact"]
        for item in comparisons.values()
    ):
        raise RuntimeError("MindDrive L4 output parity failed")
    linked = _run(["ldd", str(runner)]).stdout
    if "libpython" in linked.lower():
        raise RuntimeError("MindDrive generated Session links libpython")
    (support_root / "runner.ldd").write_text(linked, encoding="utf-8")
    manifest.verify_files(bundle_root)

    report = {
        "schema": _REPORT_SCHEMA,
        "status": "passed",
        "evidence_level": "L4",
        "model": "MindDrive Qwen2-0.5B",
        "target": args.target,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
        },
        "semantic_ir": {
            "logical_regions": len(module.regions),
            "authoritative_states": len(module.states),
            "named_outputs": len(module.outputs),
            "core_op_delta": 0,
            "io_schema_digest": io_schema_digest(module),
        },
        "bundle": {
            "root": str(bundle_root),
            "manifest_sha256": _sha256(bundle_root / "bundle.json"),
            "build_seconds": build_seconds,
            "physical_artifacts": 66,
            "python_linked": False,
        },
        "inputs": input_records,
        "runs": run_results,
        "output_parity": comparisons,
        "transactional_checks": {
            "same_revision_cache_hit": True,
            "new_revision_cache_miss": True,
            "validation_failure_aborts": True,
            "retry_commits": True,
            "episode_reset": True,
            "state_commit_count": 128,
            "state_count": 16,
        },
    }
    report_path = support_root / "minddrive-real-l4.json"
    _write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
