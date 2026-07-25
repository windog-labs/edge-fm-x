#!/usr/bin/env python3
"""Build and audit the real SmolVLA no-Python CUDA Session.

The tool deliberately keeps model-specific queue semantics in this Adapter
workflow.  The generated Session only sees declared inputs, TensorRegions,
versioned state, exact-cache metadata, and a transactional named output.
"""

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

from vlaforge.adapters.smolvla_artifact import (  # noqa: E402
    build_compiled_smolvla_action_program,
    capture_smolvla_support_regions,
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


_MODEL_REGIONS = (
    "prepare_prefix",
    "solver_step",
    "trim_action_chunk",
)
_SUPPORT_REGIONS = (
    "make_timestep",
    "queue_is_empty",
    "queue_select",
    "queue_advance",
    "queue_zero",
)
_ALL_REGIONS = (
    "prepare_prefix",
    "make_timestep",
    "solver_step",
    "trim_action_chunk",
    "queue_is_empty",
    "queue_select",
    "queue_advance",
    "queue_zero",
)
_REPORT_SCHEMA = "vlaforge.smolvla_real_l4/1"


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
    return _run(
        ["git", *command],
        environment=dict(os.environ),
    ).stdout.strip()


def _prepare_support_artifacts(
    root: Path,
    *,
    python: Path,
    target: str,
) -> None:
    exports = root / "exports"
    artifacts = root / "artifacts"
    captures_ready = all(
        (exports / f"{name}.pt2e").is_file()
        and (exports / f"{name}.capture.json").is_file()
        for name in _SUPPORT_REGIONS
    )
    if not captures_ready:
        if exports.exists() and any(exports.iterdir()):
            raise ValueError(
                f"support export directory is incomplete and non-empty: {exports}"
            )
        capture_smolvla_support_regions(exports)
    artifacts.mkdir(parents=True, exist_ok=True)
    for name in _SUPPORT_REGIONS:
        artifact = artifacts / f"{name}.pt2"
        manifest = artifacts / f"{name}.compile.json"
        if artifact.is_file() and manifest.is_file():
            record = _json(manifest)
            declared = record.get("artifact", {})
            if (
                record.get("status") != "passed"
                or declared.get("sha256") != _sha256(artifact)
                or declared.get("size_bytes") != artifact.stat().st_size
                or record.get("target") != target
            ):
                raise ValueError(
                    f"support artifact manifest does not verify: {name}"
                )
            continue
        if artifact.exists() or manifest.exists():
            raise ValueError(f"incomplete support artifact output: {name}")
        _run(
            [
                str(python),
                "-m",
                "vlaforge.cli",
                "compile-artifact",
                str(exports / f"{name}.pt2e"),
                "--output",
                str(artifact),
                "--manifest",
                str(manifest),
                "--target",
                target,
            ],
            environment={
                **dict(os.environ),
                "PYTHONPATH": str(_SOURCE_ROOT / "python"),
            },
        )


def _verify_region_source(
    *,
    name: str,
    capture_path: Path,
    artifact_path: Path,
    compile_path: Path,
    target: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (capture_path, artifact_path, compile_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    capture = _json(capture_path)
    compile_record = _json(compile_path)
    artifact = compile_record.get("artifact", {})
    exported = compile_record.get("exported_program", {})
    if capture.get("schema") != "vlaforge.frontend_capture/2":
        raise ValueError(f"{name}: unsupported capture evidence")
    if capture.get("region_name") != name:
        raise ValueError(f"{name}: capture Region identity mismatch")
    if not capture.get("effect_audit", {}).get("passed", False):
        raise ValueError(f"{name}: capture effect audit did not pass")
    if (
        compile_record.get("schema")
        != "vlaforge.compile_artifact_result/1"
        or compile_record.get("status") != "passed"
        or compile_record.get("backend") != "aoti"
        or compile_record.get("target") != target
    ):
        raise ValueError(f"{name}: compile result is not a passing {target} AOTI artifact")
    if (
        artifact.get("sha256") != _sha256(artifact_path)
        or artifact.get("size_bytes") != artifact_path.stat().st_size
    ):
        raise ValueError(f"{name}: compiled artifact digest or size mismatch")
    exported_path = capture_path.with_name(f"{name}.pt2e")
    if (
        not exported_path.is_file()
        or exported.get("sha256") != _sha256(exported_path)
    ):
        raise ValueError(f"{name}: exported program digest mismatch")
    return capture, compile_record


def _artifact_contracts(
    module: Any,
    *,
    l3_root: Path,
    support_root: Path,
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
        if name not in _ALL_REGIONS:
            raise ValueError(f"unexpected SmolVLA L4 Region: {name}")
        root = l3_root if name in _MODEL_REGIONS else support_root
        capture_path = root / "exports" / f"{name}.capture.json"
        artifact_path = root / "artifacts" / f"{name}.pt2"
        compile_path = root / "artifacts" / f"{name}.compile.json"
        capture, compiled = _verify_region_source(
            name=name,
            capture_path=capture_path,
            artifact_path=artifact_path,
            compile_path=compile_path,
            target=target,
        )
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
                model_name="SmolVLA-Base",
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
                supported_dtypes=("bf16", "bool", "f32", "i32", "i64"),
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
                "capture": str(capture_path),
                "capture_graph_sha256": capture["graph_digest"],
                "export_seconds": capture["export_seconds"],
                "maximum_export_error": capture["maximum_absolute_error"],
                "artifact": str(artifact_path),
                "artifact_sha256": contract.artifact_sha256,
                "artifact_size_bytes": contract.artifact_size_bytes,
                "compile_seconds": compiled["compile_seconds"],
                "graph_nodes": compiled["graph_nodes"],
            }
        )
    return contracts, sources, records


def _tokenize(vlm_path: Path) -> tuple[list[int], list[bool]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        vlm_path,
        local_files_only=True,
    )
    encoded = tokenizer(
        "pick up the block\n",
        padding="max_length",
        max_length=48,
        truncation=True,
        return_tensors="pt",
    )
    tokens = [int(item) for item in encoded["input_ids"][0].tolist()]
    mask = [bool(item) for item in encoded["attention_mask"][0].tolist()]
    if len(tokens) != 48 or len(mask) != 48:
        raise ValueError("SmolVLA deterministic token fixture must have length 48")
    return tokens, mask


def _write_input_fixtures(
    root: Path,
    *,
    tokens: list[int],
    mask: list[bool],
) -> dict[str, dict[str, object]]:
    import torch

    root.mkdir(parents=True, exist_ok=True)
    values = {
        "image": torch.linspace(
            0,
            1,
            3 * 256 * 256,
            device="cuda:0",
            dtype=torch.float32,
        ).reshape(1, 3, 256, 256),
        "state": torch.linspace(
            -0.2,
            0.3,
            6,
            device="cuda:0",
            dtype=torch.float32,
        ).reshape(1, 6),
        "tokens": torch.tensor(
            tokens, device="cuda:0", dtype=torch.int64
        ).reshape(1, 48),
        "mask": torch.tensor(
            mask, device="cuda:0", dtype=torch.bool
        ).reshape(1, 48),
        "noise": torch.linspace(
            -1,
            1,
            50 * 32,
            device="cuda:0",
            dtype=torch.float32,
        ).reshape(1, 50, 32),
    }
    torch.cuda.synchronize()
    records: dict[str, dict[str, object]] = {}
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
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "device_generated": "cuda:0",
        }
    _write_json(
        root / "inputs.json",
        {
            "schema": "vlaforge.smolvla_l4_inputs/1",
            "prompt": "pick up the block\\n",
            "inputs": records,
        },
    )
    return records


def _runner_source() -> str:
    source = r"""
#include "session_generated.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>

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
    std::uint32_t rank, VLAForgeDType dtype) {
  return VLAForgeBoundTensor{
      sizeof(VLAForgeBoundTensor),
      {data, bytes, dimensions, rank, dtype, {VLAFORGE_DEVICE_CUDA, 0}},
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
  bool version_sequence_ok = true;
  std::array<bool, 2> state_seen{};
  std::array<std::uint64_t, 2> state_version{};
  std::array<std::uint64_t, 2> state_episode{};
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
      if (event->subject_id >= counts->state_seen.size()) {
        counts->version_sequence_ok = false;
        break;
      }
      if (!counts->state_seen[event->subject_id] ||
          counts->state_episode[event->subject_id] != event->episode) {
        if (event->logical_version != 1u) {
          counts->version_sequence_ok = false;
        }
      } else if (
          event->logical_version !=
          counts->state_version[event->subject_id] + 1u) {
        counts->version_sequence_ok = false;
      }
      counts->state_seen[event->subject_id] = true;
      counts->state_version[event->subject_id] = event->logical_version;
      counts->state_episode[event->subject_id] = event->episode;
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
      counts->state_seen.fill(false);
      break;
    default:
      break;
  }
}

struct Inputs final {
  DeviceBuffer<float> image{3u * 256u * 256u};
  DeviceBuffer<float> state{6u};
  DeviceBuffer<std::int64_t> tokens{48u};
  DeviceBuffer<std::uint8_t> mask{48u};
  DeviceBuffer<float> noise{50u * 32u};
  std::array<float, 50u * 32u> host_noise{};
  VLAForgeBoundTensor image_view{};
  VLAForgeBoundTensor state_view{};
  VLAForgeBoundTensor tokens_view{};
  VLAForgeBoundTensor mask_view{};
  VLAForgeBoundTensor noise_view{};

  template <typename T>
  bool Read(const std::string& root, const char* name, T* destination,
            std::size_t count) {
    const std::string path = root + "/" + name + ".bin";
    std::ifstream stream(path, std::ios::binary);
    if (!stream.good()) {
      std::fprintf(stderr, "cannot open input fixture: %s\n", path.c_str());
      return false;
    }
    const auto size = static_cast<std::streamsize>(count * sizeof(T));
    stream.read(reinterpret_cast<char*>(destination), size);
    if (stream.gcount() != size || stream.peek() != std::ifstream::traits_type::eof()) {
      std::fprintf(stderr, "input fixture size mismatch: %s\n", path.c_str());
      return false;
    }
    return true;
  }

  bool Initialize(const std::string& root) {
    if (!image.ok() || !state.ok() || !tokens.ok() || !mask.ok() ||
        !noise.ok()) {
      return false;
    }
    std::array<float, 3u * 256u * 256u> host_image{};
    std::array<float, 6> host_state{};
    std::array<std::int64_t, 48> host_tokens{};
    std::array<std::uint8_t, 48> host_mask{};
    if (!Read(root, "image", host_image.data(), host_image.size()) ||
        !Read(root, "state", host_state.data(), host_state.size()) ||
        !Read(root, "tokens", host_tokens.data(), host_tokens.size()) ||
        !Read(root, "mask", host_mask.data(), host_mask.size()) ||
        !Read(root, "noise", host_noise.data(), host_noise.size())) {
      return false;
    }
    if (!image.Upload(host_image.data()) ||
        !state.Upload(host_state.data()) ||
        !tokens.Upload(host_tokens.data()) ||
        !mask.Upload(host_mask.data()) ||
        !noise.Upload(host_noise.data())) {
      return false;
    }
    static constexpr std::int64_t kImageShape[] = {1, 3, 256, 256};
    static constexpr std::int64_t kStateShape[] = {1, 6};
    static constexpr std::int64_t kTokenShape[] = {1, 48};
    static constexpr std::int64_t kNoiseShape[] = {1, 50, 32};
    image_view = Tensor(
        image.data(), image.bytes(), kImageShape, 4u, VLAFORGE_DTYPE_F32);
    state_view = Tensor(
        state.data(), state.bytes(), kStateShape, 2u, VLAFORGE_DTYPE_F32);
    tokens_view = Tensor(
        tokens.data(), tokens.bytes(), kTokenShape, 2u, VLAFORGE_DTYPE_I64);
    mask_view = Tensor(
        mask.data(), mask.bytes(), kTokenShape, 2u, VLAFORGE_DTYPE_BOOL);
    noise_view = Tensor(
        noise.data(), noise.bytes(), kNoiseShape, 3u, VLAFORGE_DTYPE_F32);
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
    result.image = image_view;
    result.image_stamp = stamp;
    result.state = state_view;
    result.state_stamp = stamp;
    result.instruction_tokens = tokens_view;
    result.instruction_tokens_stamp = stamp;
    result.instruction_mask = mask_view;
    result.instruction_mask_stamp = stamp;
    result.noise = noise_view;
    result.noise_stamp = stamp;
    return result;
  }
};

bool ReadAction(
    const VLAForgeBoundTensor& output, std::array<float, 6>* action) {
  if (action == nullptr ||
      output.tensor.device.kind != VLAFORGE_DEVICE_CUDA ||
      output.tensor.dtype != VLAFORGE_DTYPE_F32 ||
      output.tensor.size_bytes != action->size() * sizeof(float)) {
    return false;
  }
  return Cuda(
      cudaMemcpy(action->data(), output.tensor.data,
                 action->size() * sizeof(float), cudaMemcpyDeviceToHost),
      "cudaMemcpy action D2H");
}

void PrintAction(
    const char* source, std::uint64_t run,
    const std::array<float, 6>& action) {
  std::printf(
      "ACTION,%s,%llu,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
      source, static_cast<unsigned long long>(run),
      static_cast<double>(action[0]), static_cast<double>(action[1]),
      static_cast<double>(action[2]), static_cast<double>(action[3]),
      static_cast<double>(action[4]), static_cast<double>(action[5]));
}

bool RunTyped(
    vlaforge_generated::ModelSession* session,
    const vlaforge_generated::ModelInputs& inputs,
    std::array<float, 6>* action) {
  vlaforge_generated::ModelOutputs outputs{};
  const auto status = session->Run(inputs, &outputs);
  if (!status.ok()) {
    std::fprintf(stderr, "typed Run failed: %s\n", status.message);
    return false;
  }
  return ReadAction(outputs.action, action);
}

bool BindGeneric(
    const VLAForgeSessionApi* api, VLAForgeSession* session,
    const Inputs& inputs, const VLAForgeInputStamp& stamp) {
  return api->bind_tensor(session, 0u, &inputs.image_view, &stamp).code ==
             VLAFORGE_STATUS_OK &&
         api->bind_tensor(session, 1u, &inputs.state_view, &stamp).code ==
             VLAFORGE_STATUS_OK &&
         api->bind_tensor(session, 2u, &inputs.tokens_view, &stamp).code ==
             VLAFORGE_STATUS_OK &&
         api->bind_tensor(session, 3u, &inputs.mask_view, &stamp).code ==
             VLAFORGE_STATUS_OK &&
         api->bind_tensor(session, 4u, &inputs.noise_view, &stamp).code ==
             VLAFORGE_STATUS_OK;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s BUNDLE_ROOT INPUT_ROOT\n", argv[0]);
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
  std::array<float, 6> action{};
  std::array<float, 6> reset_action{};
  for (std::uint64_t run = 0; run <= 150u; ++run) {
    const bool revision_present = run != 150u;
    const std::uint64_t revision = run < 100u ? 100u : 101u;
    if (!RunTyped(
            &typed, inputs.Bind(revision, revision_present), &action)) {
      return 5;
    }
    if (run < 50u || run == 50u || run == 100u || run == 150u) {
      PrintAction("typed", run, action);
    }
  }
  if (!typed.ResetEpisode(1u).ok() ||
      !RunTyped(&typed, inputs.Bind(100u), &reset_action)) {
    return 6;
  }
  PrintAction("reset", 151u, reset_action);
  std::printf(
      "TRACE_SUMMARY,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%u\n",
      static_cast<unsigned long long>(trace.cache_hits),
      static_cast<unsigned long long>(trace.cache_misses),
      static_cast<unsigned long long>(trace.state_commits),
      static_cast<unsigned long long>(trace.transaction_commits),
      static_cast<unsigned long long>(trace.transaction_aborts),
      static_cast<unsigned long long>(trace.output_commits),
      static_cast<unsigned long long>(trace.resets),
      trace.version_sequence_ok ? 1u : 0u);
  if (trace.cache_hits != 1u || trace.cache_misses != 4u ||
      trace.state_commits != 304u ||
      trace.transaction_commits != 152u ||
      trace.transaction_aborts != 0u ||
      trace.output_commits != 152u || trace.resets != 1u ||
      !trace.version_sequence_ok) {
    std::fprintf(stderr, "typed trace contract mismatch\n");
    return 7;
  }

  VLAForgeSession* generic = nullptr;
  const std::string bundle_root(argv[1]);
  if (vlaforge_model_session_create_from_bundle(
          bundle_root.data(), bundle_root.size(), &generic).code !=
      VLAFORGE_STATUS_OK) {
    return 8;
  }
  const auto* api = vlaforge_model_session_api();
  if (vlaforge_session_api_validate(
          api, vlaforge_generated::kSchemaDigest,
          VLAFORGE_SCHEMA_DIGEST_HEX_SIZE).code != VLAFORGE_STATUS_OK) {
    api->destroy(generic);
    return 9;
  }
  const auto generic_stamp = Stamp(100u);
  if (!BindGeneric(api, generic, inputs, generic_stamp) ||
      api->run(generic).code != VLAFORGE_STATUS_OK) {
    api->destroy(generic);
    return 10;
  }
  VLAForgeBoundTensor generic_output{};
  if (api->read_output_tensor(generic, 0u, &generic_output).code !=
          VLAFORGE_STATUS_OK ||
      !ReadAction(generic_output, &action)) {
    api->destroy(generic);
    return 11;
  }
  api->destroy(generic);
  PrintAction("generic", 0u, action);
  float typed_generic_max_abs = 0.0f;
  for (std::size_t index = 0; index < action.size(); ++index) {
    typed_generic_max_abs = std::max(
        typed_generic_max_abs,
        std::fabs(action[index] - reset_action[index]));
  }
  std::printf("TYPED_GENERIC,%.9g\n",
              static_cast<double>(typed_generic_max_abs));
  if (typed_generic_max_abs != 0.0f) {
    return 12;
  }

  vlaforge_generated::ModelSession failure(argv[1]);
  if (!failure.initialization_status().ok()) {
    return 13;
  }
  TraceCounts failure_trace{};
  failure.SetTraceSink({&failure_trace, &Trace});
  if (!inputs.SetNoiseFirst(
          std::numeric_limits<float>::quiet_NaN())) {
    return 14;
  }
  vlaforge_generated::ModelOutputs failure_outputs{};
  const auto failure_status =
      failure.Run(inputs.Bind(300u), &failure_outputs);
  VLAForgeBoundTensor unavailable{};
  const auto unavailable_status =
      failure.ReadOutputTensor(0u, &unavailable);
  if (failure_status.code !=
          vlaforge::runtime::StatusCode::kValidationFailed ||
      unavailable_status.ok() ||
      failure_trace.transaction_aborts != 1u ||
      failure_trace.transaction_commits != 0u ||
      failure_trace.state_commits != 0u) {
    std::fprintf(stderr, "validation failure did not abort atomically\n");
    return 15;
  }
  inputs.host_noise[0] = -1.0f;
  if (!inputs.noise.Upload(inputs.host_noise.data()) ||
      !RunTyped(&failure, inputs.Bind(300u), &action)) {
    return 16;
  }
  std::printf(
      "FAILURE_SUMMARY,%u,%llu,%llu,%llu,%llu,%llu,%u\n",
      static_cast<unsigned>(failure_status.code),
      static_cast<unsigned long long>(failure_trace.cache_hits),
      static_cast<unsigned long long>(failure_trace.cache_misses),
      static_cast<unsigned long long>(failure_trace.state_commits),
      static_cast<unsigned long long>(failure_trace.transaction_commits),
      static_cast<unsigned long long>(failure_trace.transaction_aborts),
      failure_trace.version_sequence_ok ? 1u : 0u);
  if (failure_trace.cache_hits != 1u ||
      failure_trace.cache_misses != 1u ||
      failure_trace.state_commits != 2u ||
      failure_trace.transaction_commits != 1u ||
      failure_trace.transaction_aborts != 1u ||
      !failure_trace.version_sequence_ok) {
    return 17;
  }
  return Cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ? 0 : 18;
}
"""
    return textwrap.dedent(source).lstrip()


def _parse_runner_output(text: str) -> dict[str, object]:
    actions: dict[tuple[str, int], list[float]] = {}
    trace_summary: list[int] | None = None
    failure_summary: list[int] | None = None
    typed_generic: float | None = None
    for line in text.splitlines():
        fields = line.split(",")
        if fields[0] == "ACTION" and len(fields) == 9:
            actions[(fields[1], int(fields[2]))] = [
                float(item) for item in fields[3:]
            ]
        elif fields[0] == "TRACE_SUMMARY":
            trace_summary = [int(item) for item in fields[1:]]
        elif fields[0] == "FAILURE_SUMMARY":
            failure_summary = [int(item) for item in fields[1:]]
        elif fields[0] == "TYPED_GENERIC":
            typed_generic = float(fields[1])
    expected_action_keys = {
        *(("typed", index) for index in range(50)),
        ("typed", 50),
        ("typed", 100),
        ("typed", 150),
        ("reset", 151),
        ("generic", 0),
    }
    if set(actions) != expected_action_keys:
        missing = sorted(expected_action_keys - set(actions))
        extra = sorted(set(actions) - expected_action_keys)
        raise RuntimeError(
            f"generated runner action records mismatch: missing={missing}, extra={extra}"
        )
    if trace_summary != [1, 4, 304, 152, 0, 152, 1, 1]:
        raise RuntimeError(f"generated runner trace mismatch: {trace_summary}")
    if failure_summary != [7, 1, 1, 2, 1, 1, 1]:
        raise RuntimeError(
            f"generated runner failure trace mismatch: {failure_summary}"
        )
    if typed_generic != 0.0:
        raise RuntimeError(
            f"typed/generic C ABI mismatch: {typed_generic}"
        )
    return {
        "actions": actions,
        "trace_summary": trace_summary,
        "failure_summary": failure_summary,
        "typed_generic_max_abs": typed_generic,
    }


def _direct_artifact_action(
    *,
    l3_root: Path,
    support_root: Path,
    input_root: Path,
) -> Any:
    import torch
    import torch._inductor.codecache  # noqa: F401

    def read(name: str, dtype: Any, shape: tuple[int, ...]) -> Any:
        payload = bytearray((input_root / f"{name}.bin").read_bytes())
        return (
            torch.frombuffer(payload, dtype=dtype)
            .clone()
            .reshape(shape)
            .to("cuda:0")
        )

    image = read("image", torch.float32, (1, 3, 256, 256))
    state = read("state", torch.float32, (1, 6))
    token_tensor = read("tokens", torch.int64, (1, 48))
    mask_tensor = read("mask", torch.bool, (1, 48))
    noise = read("noise", torch.float32, (1, 50, 32))
    prefix = torch._inductor.aoti_load_package(
        str(l3_root / "artifacts" / "prepare_prefix.pt2")
    )
    solver = torch._inductor.aoti_load_package(
        str(l3_root / "artifacts" / "solver_step.pt2")
    )
    trim = torch._inductor.aoti_load_package(
        str(l3_root / "artifacts" / "trim_action_chunk.pt2")
    )
    make_timestep = torch._inductor.aoti_load_package(
        str(support_root / "artifacts" / "make_timestep.pt2")
    )
    with torch.inference_mode():
        prefix_values = prefix(
            image,
            state,
            token_tensor,
            mask_tensor,
        )
        sample = noise
        for step in range(10):
            timestep = make_timestep(
                torch.tensor(step, dtype=torch.int64)
            )
            sample = solver(
                prefix_values[0],
                sample,
                timestep,
                *prefix_values[1:],
            )
        action = trim(sample)
    torch.cuda.synchronize()
    return action.detach().cpu()


def _numeric_metrics(expected: Any, actual: list[list[float]]) -> dict[str, object]:
    import torch

    observed = torch.tensor(actual, dtype=torch.float32).reshape_as(expected)
    difference = observed.to(torch.float64) - expected.to(torch.float64)
    absolute = difference.abs()
    maximum = float(absolute.max()) if absolute.numel() else 0.0
    mean = float(absolute.mean()) if absolute.numel() else 0.0
    scale = float(
        torch.sqrt(torch.mean(expected.to(torch.float64).square()))
    )
    nrmse = (
        float(torch.sqrt(torch.mean(difference.square()))) / max(scale, 1e-12)
    )
    return {
        "shape": list(expected.shape),
        "maximum_absolute_error": maximum,
        "mean_absolute_error": mean,
        "normalized_root_mean_square_error": nrmse,
        "exact": bool(torch.equal(expected, observed)),
    }


def _audit(
    *,
    l3_root: Path,
    support_root: Path,
    bundle_root: Path,
    checkpoint: Path,
    vlm_path: Path,
    upstream_revision: str,
    target: str,
    python: Path,
    reuse_bundle: bool,
) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SmolVLA L4 requires CUDA")
    major, minor = torch.cuda.get_device_capability(0)
    actual_target = f"sm_{major}{minor}"
    if actual_target != target:
        raise RuntimeError(
            f"requested {target}, current device is {actual_target}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not vlm_path.is_dir():
        raise FileNotFoundError(vlm_path)
    bundle_present = (
        bundle_root.is_dir()
        and (bundle_root / "bundle.json").is_file()
    )
    if bundle_root.exists() and any(bundle_root.iterdir()) and not (
        reuse_bundle and bundle_present
    ):
        raise ValueError(f"bundle output must be empty: {bundle_root}")

    _prepare_support_artifacts(
        support_root,
        python=python,
        target=target,
    )
    tokens, mask = _tokenize(vlm_path)
    input_root = support_root / "inputs"
    input_records = _write_input_fixtures(
        input_root,
        tokens=tokens,
        mask=mask,
    )
    module = build_compiled_smolvla_action_program()
    semantic_path = support_root / "smolvla_real_l4.ir.json"
    semantic_path.write_text(
        canonical_json(module, indent=2) + "\n",
        encoding="utf-8",
    )
    checkpoint_sha256 = _sha256(checkpoint)
    backend_variant = f"torch-{torch.__version__}"
    contracts, sources, artifact_records = _artifact_contracts(
        module,
        l3_root=l3_root,
        support_root=support_root,
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
    runner_path = support_root / "smolvla_real_l4_runner.cpp"
    runner_path.write_text(runner_source, encoding="utf-8")
    validators = {
        "finite_action": CppValidatorDefinition(
            "finite_action",
            """if (data == nullptr || size_bytes != 6u * sizeof(float)) {
  return false;
}
const auto* action = static_cast<const float*>(data);
for (std::size_t index = 0; index < 6u; ++index) {
  if (!std::isfinite(action[index])) {
    return false;
  }
}
return true;""",
        )
    }
    initial_state = {
        "action_queue": [0.0] * (50 * 6),
        "queue_cursor": 50,
    }
    revision = _git(["rev-parse", "HEAD"])
    dirty = bool(
        _git(["status", "--porcelain", "--untracked-files=no"])
    )
    if reuse_bundle and bundle_present:
        manifest = load_bundle_manifest(bundle_root / "bundle.json")
        manifest.verify_files(bundle_root)
        bundled_runner_source = bundle_root / "generated" / "runner.cpp"
        if (
            manifest.io_schema_digest != io_schema_digest(module)
            or tuple(
                artifact.region_name
                for artifact in manifest.region_artifacts
            )
            != _ALL_REGIONS
            or not bundled_runner_source.is_file()
            or bundled_runner_source.read_text(encoding="utf-8")
            != runner_source
        ):
            raise ValueError(
                "existing bundle does not match the real SmolVLA L4 program"
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
            initial_state=initial_state,
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
        [str(runner), str(bundle_root), str(input_root)],
        environment=clean_environment,
    )
    run_seconds = time.perf_counter() - run_started
    stdout_path = support_root / "smolvla_real_l4_runner.stdout"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    parsed = _parse_runner_output(completed.stdout)

    direct = _direct_artifact_action(
        l3_root=l3_root,
        support_root=support_root,
        input_root=input_root,
    )
    action_records = parsed["actions"]
    assert isinstance(action_records, dict)
    first_chunk = [
        action_records[("typed", index)] for index in range(50)
    ]
    parity = _numeric_metrics(direct, first_chunk)
    refill_metrics = {
        label: _numeric_metrics(
            direct[:, :1, :],
            [action_records[key]],
        )
        for label, key in (
            ("same_revision_cache_hit", ("typed", 50)),
            ("new_revision_cache_miss", ("typed", 100)),
            ("missing_revision_cache_miss", ("typed", 150)),
            ("episode_reset", ("reset", 151)),
            ("generic_c_abi", ("generic", 0)),
        )
    }
    if (
        parity["maximum_absolute_error"] > 1e-6
        or any(
            value["maximum_absolute_error"] > 1e-6
            for value in refill_metrics.values()
        )
    ):
        raise RuntimeError(
            "generated C++ Session does not match the direct AOTI action chunk"
        )
    linked = _run(["ldd", str(runner)]).stdout
    if "libpython" in linked.lower():
        raise RuntimeError("generated SmolVLA Session links libpython")
    linked_path = support_root / "smolvla_real_l4_runner.ldd"
    linked_path.write_text(linked, encoding="utf-8")
    manifest.verify_files(bundle_root)

    certificate = manifest.compilation_certificate
    arena = certificate.arena
    memory_plan = _json(
        bundle_root / "metadata" / "physical_memory_plan.json"
    )
    physical_buffers = memory_plan["arena"]["physical_buffers"]
    derived_cache_bytes = sum(
        int(item["size_bytes"])
        for item in physical_buffers
        if item["buffer_class"] == "derived_cache"
    )
    authoritative_state_bytes = sum(
        int(item["slot_size_bytes"]) * int(item["slot_capacity"])
        for item in memory_plan["states"]
    )
    result = {
        "schema": _REPORT_SCHEMA,
        "status": "passed",
        "evidence_kind": "real-checkpoint-generated-session-parity",
        "evidence_level": "L4",
        "model": "SmolVLA-Base",
        "license": {
            "upstream_source": "Apache-2.0",
            "vlm_checkpoint": "Apache-2.0",
            "policy_checkpoint_card": "not present in local checkpoint directory",
        },
        "upstream_revision": upstream_revision,
        "repository": {
            "revision": revision,
            "source_dirty": dirty,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint.stat().st_size,
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
            "regions": list(_ALL_REGIONS),
            "core_op_delta": 0,
            "adapter_template": "ChunkedAction",
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
            "direct_artifact_vs_generated_chunk": parity,
            "milestones": refill_metrics,
            "typed_generic_max_abs": parsed["typed_generic_max_abs"],
        },
        "cache": {
            "key_inputs": [
                "image",
                "state",
                "instruction_tokens",
                "instruction_mask",
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
            "committed_outputs": parsed["trace_summary"][5],
            "resets": parsed["trace_summary"][6],
            "state_version_sequence": "passed",
            "validation_failure_status_code": parsed["failure_summary"][0],
            "failure_retry_cache_hits": parsed["failure_summary"][1],
            "failure_retry_cache_misses": parsed["failure_summary"][2],
            "failure_retry_state_commits": parsed["failure_summary"][3],
            "failure_retry_transaction_commits": parsed["failure_summary"][4],
            "failure_retry_transaction_aborts": parsed["failure_summary"][5],
            "failure_preserved_uncommitted_output": True,
        },
        "memory": {
            "arena_baseline_bytes": arena.baseline_bytes,
            "arena_compiled_bytes": arena.compiled_bytes,
            "arena_saved_bytes": arena.saved_bytes,
            "derived_cache_bytes": derived_cache_bytes,
            "authoritative_state_bytes": authoritative_state_bytes,
            "note": (
                "Derived-cache and state bytes are taken from the verified "
                "static memory plan; artifact package bytes are reported separately."
            ),
        },
        "execution": {
            "runner_seconds": run_seconds,
            "successful_runs": 152,
            "failure_injection_runs": 2,
            "stdout": str(stdout_path),
        },
        "l3_chain": {
            "root": str(l3_root),
            "report": str(l3_root / "smolvla-l3.json"),
            "claim": (
                "The existing real L3 report connects upstream eager to these "
                "same prefix/solver/trim artifacts; this report connects the "
                "same artifacts to the generated no-Python C++ Session."
            ),
        },
        "reproduction": {
            "environment": {
                "PYTHONPATH": (
                    "/path/to/lerobot/src:"
                    "/path/to/edge-fm-x/vlaforge/python"
                ),
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
            },
            "command": [
                str(python),
                "vlaforge/tools/build_real_smolvla_l4.py",
                "--l3-root",
                str(l3_root),
                "--support-root",
                str(support_root),
                "--bundle-root",
                str(bundle_root),
                "--checkpoint",
                str(checkpoint),
                "--vlm-path",
                str(vlm_path),
                "--upstream-revision",
                upstream_revision,
                "--target",
                target,
                "--python",
                str(python),
                "--report",
                "<report.json>",
            ],
        },
        "passed": True,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l3-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--target", default="sm_86")
    parser.add_argument(
        "--reuse-bundle",
        action="store_true",
        help="Verify and execute an already-built matching bundle.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used for support artifact compilation.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = _audit(
        l3_root=args.l3_root.resolve(),
        support_root=args.support_root.resolve(),
        bundle_root=args.bundle_root.resolve(),
        checkpoint=args.checkpoint.resolve(),
        vlm_path=args.vlm_path.resolve(),
        upstream_revision=args.upstream_revision,
        target=args.target,
        python=args.python.resolve(),
        reuse_bundle=args.reuse_bundle,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report is not None:
        _write_json(args.report.resolve(), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
