#!/usr/bin/env python3
"""Assemble and audit the real OpenVLA-7B weight-paged C++ Session."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.adapters.openvla_artifact import (  # noqa: E402
    OPENVLA_L4_SUPPORT_REGIONS,
    build_compiled_openvla_program,
    capture_openvla_l4_support_regions,
)
from vlaforge.adapters.openvla_partitioned import (  # noqa: E402
    OPENVLA_UPSTREAM_REVISION,
    artifact_region_names,
)
from vlaforge.codegen import CppValidatorDefinition  # noqa: E402
from vlaforge.deployment import (  # noqa: E402
    ArtifactIdentity,
    ArtifactKind,
    ArtifactResidency,
    BackendCapability,
    EffectAudit,
    RegionArtifactContract,
    ValueContract,
    WorkspaceContract,
    build_artifact_compile_bundle,
)
from vlaforge.ir.serializer import io_schema_digest  # noqa: E402
from vlaforge.ir.types import TensorType  # noqa: E402
from vlaforge.plan import storage_size_bytes  # noqa: E402


_REPORT_SCHEMA = "vlaforge.openvla_real_l4/1"
_COMPILE_SCHEMA = "vlaforge.real_aoti_compile/1"
_EXPECTED_TOKENS = [31857, 31864, 31900, 31840, 31860, 31868, 31872]


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


def _git(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPOSITORY_ROOT,
    ).stdout.strip()


def _prepare_support(
    root: Path,
    *,
    target: str,
    python: Path,
) -> None:
    exports = root / "exports"
    artifacts = root / "artifacts"
    if not all(
        (exports / f"{name}.pt2e").is_file()
        and (exports / f"{name}.capture.json").is_file()
        for name in OPENVLA_L4_SUPPORT_REGIONS
    ):
        if exports.exists() and any(exports.iterdir()):
            raise ValueError(
                f"incomplete non-empty support export root: {exports}"
            )
        capture_openvla_l4_support_regions(exports)
    artifacts.mkdir(parents=True, exist_ok=True)
    for name in OPENVLA_L4_SUPPORT_REGIONS:
        artifact = artifacts / f"{name}.pt2"
        manifest = artifacts / f"{name}.compile.json"
        if artifact.is_file() and manifest.is_file():
            record = _json(manifest)
            if (
                record.get("status") != "passed"
                or record.get("target") != target
                or record.get("artifact", {}).get("sha256")
                != _sha256(artifact)
                or record.get("artifact", {}).get("size_bytes")
                != artifact.stat().st_size
            ):
                raise ValueError(f"{name}: support artifact does not verify")
            continue
        if artifact.exists() or manifest.exists():
            raise ValueError(f"{name}: incomplete support artifact output")
        subprocess.run(
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
                "--inductor-profile",
                "conservative",
            ],
            check=True,
            env={
                **dict(os.environ),
                "PYTHONPATH": str(_SOURCE_ROOT / "python"),
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
                "MAX_JOBS": "1",
            },
        )


def _verify_l3(
    report_path: Path,
    *,
    target: str,
) -> dict[str, Any]:
    report = _json(report_path)
    if (
        report.get("schema") != "vlaforge.openvla_real_l3/1"
        or report.get("status") != "passed"
        or report.get("evidence_level") != "L3"
        or report.get("environment", {}).get("target") != target
        or report.get("checkpoint", {}).get("revision")
        != OPENVLA_UPSTREAM_REVISION
        or report.get("correctness", {}).get("action_tokens_equal")
        is not True
        or report.get("correctness", {}).get("repeated_pipeline_exact")
        is not True
    ):
        raise ValueError("OpenVLA L3 evidence is not passing")
    pipelines = report["correctness"]["pipelines"]
    if (
        not pipelines
        or any(item["tokens"] != _EXPECTED_TOKENS for item in pipelines)
    ):
        raise ValueError("OpenVLA L3 token identity mismatch")
    return report


def _stable_model_artifact(
    *,
    l3_root: Path,
    name: str,
    package_path: Path,
) -> tuple[Path, str, dict[str, Path], dict[str, object]]:
    """Resolve one verified, stably extracted raw AOTI artifact tree."""

    extracted = l3_root / "extracted_artifacts" / name
    marker = extracted / ".package-sha256"
    package_digest = _sha256(package_path)
    if (
        not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != package_digest
    ):
        raise ValueError(
            f"{name}: stable extraction does not match AOTI package"
        )
    wrappers = tuple(sorted(extracted.rglob("*.wrapper.so")))
    if len(wrappers) != 1:
        raise ValueError(
            f"{name}: expected one stable wrapper.so, got {len(wrappers)}"
        )
    wrapper = wrappers[0]
    runtime_root = wrapper.parent
    bundle_root = f"artifacts/{name}"
    artifact_path = f"{bundle_root}/{wrapper.name}"
    auxiliary: dict[str, Path] = {}
    runtime_files = []
    for source in sorted(runtime_root.rglob("*")):
        if not source.is_file() or source == wrapper:
            continue
        relative = source.relative_to(runtime_root).as_posix()
        destination = f"{bundle_root}/{relative}"
        auxiliary[destination] = source
        runtime_files.append(
            {
                "path": relative,
                "sha256": _sha256(source),
                "size_bytes": source.stat().st_size,
            }
        )
    return (
        wrapper,
        artifact_path,
        auxiliary,
        {
            "package_sha256": package_digest,
            "wrapper_sha256": _sha256(wrapper),
            "wrapper_size_bytes": wrapper.stat().st_size,
            "runtime_file_count": len(runtime_files),
            "runtime_files_size_bytes": sum(
                int(item["size_bytes"]) for item in runtime_files
            ),
            "runtime_files": runtime_files,
        },
    )


def _artifact_contracts(
    module: Any,
    *,
    capture_root: Path,
    l3_root: Path,
    support_root: Path,
    l3: dict[str, Any],
    target: str,
) -> tuple[
    dict[str, RegionArtifactContract],
    dict[str, Path],
    list[dict[str, object]],
    dict[str, Path],
]:
    model_regions = frozenset(artifact_region_names())
    contracts: dict[str, RegionArtifactContract] = {}
    sources: dict[str, Path] = {}
    records: list[dict[str, object]] = []
    auxiliary_files: dict[str, Path] = {}
    digest = io_schema_digest(module)
    shard_identity = ",".join(
        str(item["sha256"]) for item in l3["checkpoint"]["shards"]
    )
    for region_id, region in enumerate(module.regions):
        name = region.name
        is_model = name in model_regions
        root = (
            capture_root / "source_exports"
            if is_model
            else support_root / "exports"
        )
        artifact_root = (
            l3_root / "artifacts"
            if is_model
            else support_root / "artifacts"
        )
        capture_path = root / f"{name}.capture.json"
        artifact_path = artifact_root / f"{name}.pt2"
        compile_path = (
            l3_root / "compile_reports" / f"{name}.json"
            if is_model
            else artifact_root / f"{name}.compile.json"
        )
        for path in (capture_path, artifact_path, compile_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        capture = _json(capture_path)
        compiled = _json(compile_path)
        if (
            capture.get("schema") != "vlaforge.frontend_capture/2"
            or capture.get("region_name") != name
            or not capture.get("effect_audit", {}).get("passed", False)
        ):
            raise ValueError(f"{name}: capture contract does not verify")
        if is_model:
            entries = compiled.get("regions")
            if (
                compiled.get("schema") != _COMPILE_SCHEMA
                or not isinstance(entries, list)
                or len(entries) != 1
                or entries[0].get("region") != name
                or entries[0].get("package_sha256")
                != _sha256(artifact_path)
                or entries[0].get("package_size_bytes")
                != artifact_path.stat().st_size
            ):
                raise ValueError(f"{name}: model compile record mismatch")
            compile_seconds = float(entries[0]["compile_seconds"])
        else:
            if (
                compiled.get("schema")
                != "vlaforge.compile_artifact_result/1"
                or compiled.get("status") != "passed"
                or compiled.get("target") != target
                or compiled.get("artifact", {}).get("sha256")
                != _sha256(artifact_path)
                or compiled.get("artifact", {}).get("size_bytes")
                != artifact_path.stat().st_size
            ):
                raise ValueError(f"{name}: support compile record mismatch")
            compile_seconds = float(compiled["compile_seconds"])
        inputs = tuple(
            ValueContract.from_dict(item)
            for item in capture.get("inputs", ())
        )
        outputs = tuple(
            ValueContract.from_dict(item)
            for item in capture.get("outputs", ())
        )
        dtypes = tuple(
            sorted(
                {
                    value.type.dtype
                    for value in (*inputs, *outputs)
                    if isinstance(value.type, TensorType)
                }
            )
        )
        if is_model:
            (
                artifact_source,
                bundle_artifact_path,
                region_auxiliary,
                stable_record,
            ) = _stable_model_artifact(
                l3_root=l3_root,
                name=name,
                package_path=artifact_path,
            )
            overlap = set(auxiliary_files).intersection(region_auxiliary)
            if overlap:
                raise ValueError(
                    f"{name}: duplicate runtime artifacts: {sorted(overlap)}"
                )
            auxiliary_files.update(region_auxiliary)
            artifact_kind = ArtifactKind.SHARED_LIBRARY
            backend_variant = (
                "torch-2.10-cu128-stable-raw-weight-paged"
            )
        else:
            artifact_source = artifact_path
            bundle_artifact_path = f"artifacts/{name}.pt2"
            stable_record = None
            artifact_kind = ArtifactKind.AOTI_PACKAGE
            backend_variant = "torch-2.10-cu128-session-package"
        contract = RegionArtifactContract(
            region_id=region_id,
            region_name=name,
            inputs=inputs,
            outputs=outputs,
            io_schema_digest=digest,
            identity=ArtifactIdentity(
                model_name="OpenVLA-7B",
                upstream_revision=OPENVLA_UPSTREAM_REVISION,
                checkpoint_identity=f"sha256-set:{shard_identity}",
                graph_sha256=str(capture["graph_digest"]),
            ),
            artifact_kind=artifact_kind,
            artifact_path=bundle_artifact_path,
            artifact_sha256=_sha256(artifact_source),
            artifact_size_bytes=artifact_source.stat().st_size,
            workspace=WorkspaceContract(device="cuda:0"),
            capability=BackendCapability(
                backend="aoti",
                target=target,
                supported_dtypes=dtypes,
                supports_dynamic_shapes=False,
                supports_device_resident_io=True,
                requires_synchronize=True,
            ),
            effect_audit=EffectAudit.from_dict(capture["effect_audit"]),
            backend_variant=backend_variant,
            residency=(
                ArtifactResidency.INVOCATION
                if is_model
                else ArtifactResidency.SESSION
            ),
        )
        contracts[name] = contract
        sources[name] = artifact_source
        records.append(
            {
                "region_id": region_id,
                "name": name,
                "kind": "model" if is_model else "support",
                "residency": contract.residency.value,
                "artifact_sha256": contract.artifact_sha256,
                "artifact_size_bytes": contract.artifact_size_bytes,
                "compile_seconds": compile_seconds,
                "provider": (
                    "stable-raw-wrapper"
                    if is_model
                    else "session-package"
                ),
                "stable_extraction": stable_record,
            }
        )
    return contracts, sources, records, auxiliary_files


def _write_input_fixtures(
    *,
    l3_root: Path,
    capture_root: Path,
    output_root: Path,
) -> dict[str, dict[str, object]]:
    import torch

    output_root.mkdir(parents=True, exist_ok=True)
    exported = torch.export.load(
        l3_root / "normalized_exports" / "prepare_multimodal_prefix.pt2"
    )
    arguments, keyword_arguments = exported.example_inputs
    if keyword_arguments or len(arguments) != 3:
        raise ValueError("unexpected OpenVLA prepare example inputs")
    names = ("image", "instruction_tokens", "instruction_mask")
    capture = _json(capture_root / "capture.json")
    capture_names = ("pixel_values", "input_ids", "attention_mask")
    records: dict[str, dict[str, object]] = {}
    for name, capture_name, tensor in zip(
        names,
        capture_names,
        arguments,
        strict=True,
    ):
        contiguous = tensor.detach().contiguous()
        payload = contiguous.view(torch.uint8).cpu().numpy().tobytes()
        path = output_root / f"{name}.bin"
        path.write_bytes(payload)
        digest = _sha256(path)
        expected = capture["fixture"][capture_name]
        if (
            digest != expected["sha256"]
            or len(payload) != expected["size_bytes"]
        ):
            raise ValueError(f"{name}: input fixture identity mismatch")
        records[name] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": len(payload),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    del (
        exported,
        arguments,
        keyword_arguments,
        tensor,
        contiguous,
        payload,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return records


def _runner_source(failure_artifact_path: str) -> str:
    failure_path = Path(failure_artifact_path)
    if failure_path.is_absolute() or ".." in failure_path.parts:
        raise ValueError("failure artifact path must stay inside the bundle")
    source = r"""
#include "session_generated.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
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

class DeviceBytes final {
 public:
  explicit DeviceBytes(std::size_t bytes) : bytes_(bytes) {
    status_ = cudaMalloc(&data_, bytes_);
  }
  ~DeviceBytes() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }
  DeviceBytes(const DeviceBytes&) = delete;
  DeviceBytes& operator=(const DeviceBytes&) = delete;
  bool ok() const { return status_ == cudaSuccess && data_ != nullptr; }
  void* data() const { return data_; }
  std::size_t bytes() const { return bytes_; }
  bool Upload(const std::vector<std::uint8_t>& source) {
    return source.size() == bytes_ &&
           Cuda(cudaMemcpy(data_, source.data(), bytes_,
                           cudaMemcpyHostToDevice),
                "cudaMemcpy H2D");
  }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0u;
  cudaError_t status_ = cudaSuccess;
};

bool ReadBytes(const std::string& path, std::size_t expected,
               std::vector<std::uint8_t>* output) {
  output->resize(expected);
  std::ifstream stream(path, std::ios::binary);
  if (!stream.good()) {
    return false;
  }
  stream.read(reinterpret_cast<char*>(output->data()),
              static_cast<std::streamsize>(expected));
  return stream.gcount() == static_cast<std::streamsize>(expected) &&
         stream.peek() == std::ifstream::traits_type::eof();
}

VLAForgeBoundTensor Tensor(
    void* data, std::uint64_t bytes, const std::int64_t* dimensions,
    std::uint32_t rank, VLAForgeDType dtype, std::uint32_t alignment) {
  return VLAForgeBoundTensor{
      sizeof(VLAForgeBoundTensor),
      {data, bytes, dimensions, rank, dtype, {VLAFORGE_DEVICE_CUDA, 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS,
      alignment};
}

VLAForgeInputStamp Stamp(std::uint64_t revision) {
  return VLAForgeInputStamp{
      sizeof(VLAForgeInputStamp), 1u, 0u, {}, revision, 0u};
}

struct Inputs {
  DeviceBytes image{1u * 6u * 224u * 224u * 2u};
  DeviceBytes tokens{1u * 19u * 8u};
  DeviceBytes mask{1u * 19u * 8u};
  VLAForgeBoundTensor image_view{};
  VLAForgeBoundTensor tokens_view{};
  VLAForgeBoundTensor mask_view{};

  bool Initialize(const std::string& root) {
    std::vector<std::uint8_t> host_image;
    std::vector<std::uint8_t> host_tokens;
    std::vector<std::uint8_t> host_mask;
    if (!image.ok() || !tokens.ok() || !mask.ok() ||
        !ReadBytes(root + "/image.bin", image.bytes(), &host_image) ||
        !ReadBytes(root + "/instruction_tokens.bin", tokens.bytes(),
                   &host_tokens) ||
        !ReadBytes(root + "/instruction_mask.bin", mask.bytes(),
                   &host_mask) ||
        !image.Upload(host_image) || !tokens.Upload(host_tokens) ||
        !mask.Upload(host_mask)) {
      return false;
    }
    static constexpr std::int64_t kImageShape[] = {1, 6, 224, 224};
    static constexpr std::int64_t kLanguageShape[] = {1, 19};
    image_view = Tensor(image.data(), image.bytes(), kImageShape, 4u,
                        VLAFORGE_DTYPE_BF16, 2u);
    tokens_view = Tensor(tokens.data(), tokens.bytes(), kLanguageShape, 2u,
                         VLAFORGE_DTYPE_I64, 8u);
    mask_view = Tensor(mask.data(), mask.bytes(), kLanguageShape, 2u,
                       VLAFORGE_DTYPE_I64, 8u);
    return true;
  }

  vlaforge_generated::ModelInputs Bind(std::uint64_t revision) const {
    vlaforge_generated::ModelInputs result{};
    const auto stamp = Stamp(revision);
    result.image = image_view;
    result.image_stamp = stamp;
    result.instruction_tokens = tokens_view;
    result.instruction_tokens_stamp = stamp;
    result.instruction_mask = mask_view;
    result.instruction_mask_stamp = stamp;
    return result;
  }
};

struct TraceCounts {
  std::uint64_t regions = 0u;
  std::uint64_t cache_hits = 0u;
  std::uint64_t cache_misses = 0u;
  std::uint64_t transaction_commits = 0u;
  std::uint64_t transaction_aborts = 0u;
  std::uint64_t output_commits = 0u;
};

void Trace(void* context, const vlaforge::runtime::TraceEvent* event) {
  auto* counts = static_cast<TraceCounts*>(context);
  using vlaforge::runtime::TraceKind;
  if (event->kind == TraceKind::kRegion) {
    ++counts->regions;
  } else if (event->kind == TraceKind::kCacheHit) {
    ++counts->cache_hits;
  } else if (event->kind == TraceKind::kCacheMiss) {
    ++counts->cache_misses;
  } else if (event->kind == TraceKind::kTransactionCommit) {
    ++counts->transaction_commits;
  } else if (event->kind == TraceKind::kTransactionAbort) {
    ++counts->transaction_aborts;
  } else if (event->kind == TraceKind::kOutputGroupCommit) {
    ++counts->output_commits;
  }
}

bool ReadAction(const VLAForgeBoundTensor& output,
                std::array<double, 7>* action) {
  if (output.tensor.device.kind != VLAFORGE_DEVICE_CUDA ||
      output.tensor.dtype != VLAFORGE_DTYPE_F64 ||
      output.tensor.size_bytes != action->size() * sizeof(double)) {
    return false;
  }
  return Cuda(
      cudaMemcpy(action->data(), output.tensor.data,
                 action->size() * sizeof(double), cudaMemcpyDeviceToHost),
      "cudaMemcpy action D2H");
}

bool Equal(const std::array<double, 7>& left,
           const std::array<double, 7>& right) {
  return left == right;
}

class ArtifactMove final {
 public:
  explicit ArtifactMove(const std::filesystem::path& original)
      : original_(original), hidden_(original.string() + ".failure_probe") {
    std::error_code remove_error;
    std::filesystem::remove(hidden_, remove_error);
    std::error_code rename_error;
    std::filesystem::rename(original_, hidden_, rename_error);
    moved_ = !rename_error;
  }

  ~ArtifactMove() { (void)Restore(); }

  bool moved() const { return moved_; }

  bool Restore() {
    if (!moved_) {
      return true;
    }
    std::error_code rename_error;
    std::filesystem::rename(hidden_, original_, rename_error);
    if (rename_error) {
      return false;
    }
    moved_ = false;
    return true;
  }

 private:
  std::filesystem::path original_;
  std::filesystem::path hidden_;
  bool moved_ = false;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s BUNDLE_ROOT INPUT_ROOT RUNS\n", argv[0]);
    return 2;
  }
  const int runs = std::stoi(argv[3]);
  if (runs < 1) {
    return 2;
  }
  Inputs inputs;
  if (!inputs.Initialize(argv[2])) {
    return 3;
  }

  vlaforge_generated::ModelSession typed(argv[1]);
  if (!typed.initialization_status().ok()) {
    std::fprintf(stderr, "typed initialization failed: %s\n",
                 typed.initialization_status().message);
    return 4;
  }
  TraceCounts trace{};
  typed.SetTraceSink({&trace, &Trace});
  std::array<double, 7> baseline{};
  for (int index = 0; index < runs; ++index) {
    vlaforge_generated::ModelOutputs outputs{};
    const auto status = typed.Run(inputs.Bind(7u), &outputs);
    std::array<double, 7> current{};
    if (!status.ok() || !ReadAction(outputs.action, &current) ||
        (index > 0 && !Equal(baseline, current))) {
      std::fprintf(stderr, "typed Run failed: %s\n", status.message);
      return 5;
    }
    if (index == 0) {
      baseline = current;
    }
  }

  const auto commits_before_failure = trace.transaction_commits;
  const auto aborts_before_failure = trace.transaction_aborts;
  const auto outputs_before_failure = trace.output_commits;
  const std::filesystem::path failed_artifact =
      std::filesystem::path(argv[1]) / @FAILURE_ARTIFACT@;
  ArtifactMove failure_probe(failed_artifact);
  if (!failure_probe.moved()) {
    std::fprintf(stderr, "cannot move invocation-resident failure probe\n");
    return 6;
  }
  vlaforge_generated::ModelOutputs failed_outputs{};
  const auto failure_status = typed.Run(inputs.Bind(7u), &failed_outputs);
  VLAForgeBoundTensor preserved_output{};
  std::array<double, 7> preserved_action{};
  const bool output_preserved =
      !failure_status.ok() &&
      typed.ReadOutputTensor(0u, &preserved_output).ok() &&
      ReadAction(preserved_output, &preserved_action) &&
      Equal(baseline, preserved_action);
  const bool failure_trace_ok =
      trace.transaction_commits == commits_before_failure &&
      trace.transaction_aborts == aborts_before_failure + 1u &&
      trace.output_commits == outputs_before_failure;
  const bool artifact_restored = failure_probe.Restore();
  if (!artifact_restored || !output_preserved || !failure_trace_ok) {
    std::fprintf(
        stderr,
        "backend failure transaction probe failed: "
        "restore=%u status=%u message=%s output_preserved=%u "
        "trace_ok=%u commits=%llu/%llu aborts=%llu/%llu "
        "outputs=%llu/%llu\n",
        artifact_restored ? 1u : 0u,
        static_cast<unsigned>(failure_status.code),
        failure_status.message,
        output_preserved ? 1u : 0u,
        failure_trace_ok ? 1u : 0u,
        static_cast<unsigned long long>(trace.transaction_commits),
        static_cast<unsigned long long>(commits_before_failure),
        static_cast<unsigned long long>(trace.transaction_aborts),
        static_cast<unsigned long long>(aborts_before_failure),
        static_cast<unsigned long long>(trace.output_commits),
        static_cast<unsigned long long>(outputs_before_failure));
    return 7;
  }
  vlaforge_generated::ModelOutputs recovered_outputs{};
  std::array<double, 7> recovered_action{};
  const auto recovery_status =
      typed.Run(inputs.Bind(7u), &recovered_outputs);
  const bool recovered =
      recovery_status.ok() &&
      ReadAction(recovered_outputs.action, &recovered_action) &&
      Equal(baseline, recovered_action);
  if (!recovered) {
    std::fprintf(stderr, "backend failure recovery failed: %s\n",
                 recovery_status.message);
    return 8;
  }
  std::printf("FAILURE,%u,%u,%llu,%llu,%llu,%u\n",
              static_cast<unsigned>(failure_status.code),
              output_preserved ? 1u : 0u,
              static_cast<unsigned long long>(
                  trace.transaction_aborts - aborts_before_failure),
              static_cast<unsigned long long>(
                  trace.transaction_commits - commits_before_failure),
              static_cast<unsigned long long>(
                  trace.output_commits - outputs_before_failure),
              recovered ? 1u : 0u);

  VLAForgeSession* generic = nullptr;
  const std::string bundle_root(argv[1]);
  if (vlaforge_model_session_create_from_bundle(
          bundle_root.data(), bundle_root.size(), &generic).code !=
      VLAFORGE_STATUS_OK) {
    return 9;
  }
  const auto* api = vlaforge_model_session_api();
  if (vlaforge_session_api_validate(
          api, vlaforge_generated::kSchemaDigest,
          VLAFORGE_SCHEMA_DIGEST_HEX_SIZE).code != VLAFORGE_STATUS_OK) {
    api->destroy(generic);
    return 10;
  }
  const auto stamp = Stamp(7u);
  const bool bound =
      api->bind_tensor(generic, 0u, &inputs.image_view, &stamp).code ==
          VLAFORGE_STATUS_OK &&
      api->bind_tensor(generic, 1u, &inputs.tokens_view, &stamp).code ==
          VLAFORGE_STATUS_OK &&
      api->bind_tensor(generic, 2u, &inputs.mask_view, &stamp).code ==
          VLAFORGE_STATUS_OK;
  VLAForgeBoundTensor generic_output{};
  std::array<double, 7> generic_action{};
  const bool generic_ok =
      bound && api->run(generic).code == VLAFORGE_STATUS_OK &&
      api->read_output_tensor(generic, 0u, &generic_output).code ==
          VLAFORGE_STATUS_OK &&
      ReadAction(generic_output, &generic_action);
  api->destroy(generic);
  if (!generic_ok || !Equal(baseline, generic_action)) {
    return 11;
  }

  std::printf("ACTION");
  for (double value : baseline) {
    std::printf(",%.17g", value);
  }
  std::printf("\n");
  std::printf("TYPED_GENERIC,1\n");
  std::printf("TRACE,%llu,%llu,%llu,%llu,%llu,%llu\n",
              static_cast<unsigned long long>(trace.regions),
              static_cast<unsigned long long>(trace.cache_hits),
              static_cast<unsigned long long>(trace.cache_misses),
              static_cast<unsigned long long>(trace.transaction_commits),
              static_cast<unsigned long long>(trace.transaction_aborts),
              static_cast<unsigned long long>(trace.output_commits));
  return 0;
}
"""
    return source.replace(
        "@FAILURE_ARTIFACT@",
        json.dumps(failure_artifact_path),
    )


def _parse_runner(text: str) -> dict[str, object]:
    action: list[float] | None = None
    trace: list[int] | None = None
    failure: list[int] | None = None
    typed_generic = False
    for line in text.splitlines():
        fields = line.split(",")
        if fields[0] == "ACTION":
            action = [float(item) for item in fields[1:]]
        elif fields[0] == "TYPED_GENERIC":
            typed_generic = fields[1:] == ["1"]
        elif fields[0] == "TRACE":
            trace = [int(item) for item in fields[1:]]
        elif fields[0] == "FAILURE":
            failure = [int(item) for item in fields[1:]]
    if (
        action is None
        or len(action) != 7
        or trace is None
        or len(trace) != 6
        or failure is None
        or len(failure) != 6
    ):
        raise RuntimeError(f"generated runner output is incomplete: {text}")
    if failure[0] == 0 or failure[1:] != [1, 1, 1, 1, 1]:
        raise RuntimeError(
            f"generated runner failure/recovery mismatch: {failure}"
        )
    return {
        "action": action,
        "typed_generic_equal": typed_generic,
        "trace": {
            "regions": trace[0],
            "cache_hits": trace[1],
            "cache_misses": trace[2],
            "transaction_commits": trace[3],
            "transaction_aborts": trace[4],
            "output_commits": trace[5],
        },
        "backend_failure": {
            "status_code": failure[0],
            "previous_output_preserved": failure[1] == 1,
            "transaction_abort_delta": failure[2],
            "recovery_transaction_commit_delta": failure[3],
            "recovery_output_commit_delta": failure[4],
            "recovered": failure[5] == 1,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--l3-root", type=Path, required=True)
    parser.add_argument("--l3-report", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--target", default="sm_86")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("OpenVLA L4 requires CUDA")
    if args.bundle_root.exists() and any(args.bundle_root.iterdir()):
        raise ValueError(f"bundle root must be empty: {args.bundle_root}")
    l3 = _verify_l3(args.l3_report, target=args.target)
    _prepare_support(
        args.support_root,
        target=args.target,
        python=args.python,
    )
    module = build_compiled_openvla_program()
    (
        contracts,
        sources,
        artifact_records,
        auxiliary_files,
    ) = _artifact_contracts(
        module,
        capture_root=args.capture_root,
        l3_root=args.l3_root,
        support_root=args.support_root,
        l3=l3,
        target=args.target,
    )
    inputs = _write_input_fixtures(
        l3_root=args.l3_root,
        capture_root=args.capture_root,
        output_root=args.input_root,
    )
    revision = _git(["rev-parse", "HEAD"])
    dirty = bool(
        _git(["status", "--porcelain", "--untracked-files=no"])
    )
    build_started = time.perf_counter()
    manifest = build_artifact_compile_bundle(
        module,
        args.bundle_root,
        region_artifacts=contracts,
        artifact_sources=sources,
        validators={
            "finite_action": CppValidatorDefinition(
                "finite_action",
                """if (data == nullptr || size_bytes != 7u * sizeof(double)) {
  return false;
}
const auto* action = static_cast<const double*>(data);
for (std::size_t index = 0; index < 7u; ++index) {
  if (!std::isfinite(action[index])) {
    return false;
  }
}
return true;""",
            )
        },
        runner_source=_runner_source(
            contracts["decode_token_embedding"].artifact_path
        ),
        runtime_root=_SOURCE_ROOT,
        cmake_prefix_path=torch.utils.cmake_prefix_path,
        backend_versions={
            "aoti": f"torch-{torch.__version__}",
            "cuda": str(torch.version.cuda),
        },
        profile="verified",
        source_revision=revision,
        source_dirty=dirty,
        environment={
            "TORCH_CUDA_ARCH_LIST": "8.6",
            "artifact_residency": "weight-paged",
        },
        default_device="cuda:0",
        state_device="cpu",
        auxiliary_files=auxiliary_files,
    )
    build_seconds = time.perf_counter() - build_started
    runner = args.bundle_root / "bin" / "vlaforge_generated_runner"
    environment = {
        **dict(os.environ),
        "PYTHONHOME": "/definitely/not/a/python/home",
        "PYTHONPATH": "/definitely/not/a/python/path",
    }
    run_started = time.perf_counter()
    completed = subprocess.run(
        [
            str(runner),
            str(args.bundle_root),
            str(args.input_root),
            str(args.runs),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    run_seconds = time.perf_counter() - run_started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    stdout_log = args.report.with_suffix(
        args.report.suffix + ".runner.stdout.log"
    )
    stderr_log = args.report.with_suffix(
        args.report.suffix + ".runner.stderr.log"
    )
    stdout_log.write_text(completed.stdout, encoding="utf-8")
    stderr_log.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "generated OpenVLA runner failed with exit code "
            f"{completed.returncode}; stdout={stdout_log}; "
            f"stderr={stderr_log}"
        )
    parsed = _parse_runner(completed.stdout)
    reference_action = l3["correctness"]["pipelines"][0]["action"]
    maximum_error = max(
        abs(expected - actual)
        for expected, actual in zip(
            reference_action,
            parsed["action"],
            strict=True,
        )
    )
    if maximum_error > 1e-12 or not parsed["typed_generic_equal"]:
        raise RuntimeError(
            "generated OpenVLA action parity failed: "
            f"max_abs={maximum_error}"
        )
    linked = subprocess.run(
        ["ldd", str(runner)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "libpython" in linked.lower():
        raise RuntimeError("OpenVLA generated runner links libpython")
    manifest.verify_files(args.bundle_root)
    output_payload_bytes = sum(
        storage_size_bytes(port.payload) for port in module.outputs
    )
    report = {
        "schema": _REPORT_SCHEMA,
        "status": "passed",
        "passed": True,
        "evidence_level": "L4",
        "evidence_kind": "real-checkpoint-weight-paged-generated-session",
        "model": "OpenVLA-7B",
        "checkpoint": l3["checkpoint"],
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
            "target": args.target,
        },
        "semantic_ir": {
            "schema_version": module.schema_version,
            "logical_control": "bounded autoregressive decode",
            "physical_schedule": "six-step unrolled weight paging",
            "regions": len(module.regions),
            "persistent_state_slots": len(module.states),
            "core_op_delta": 0,
        },
        "artifacts": {
            "records": artifact_records,
            "invocation_resident": sum(
                item["residency"] == "invocation"
                for item in artifact_records
            ),
            "session_resident": sum(
                item["residency"] == "session"
                for item in artifact_records
            ),
            "total_bytes": sum(
                int(item["artifact_size_bytes"])
                for item in artifact_records
            ),
            "stable_runtime_auxiliary_files": len(auxiliary_files),
            "stable_runtime_auxiliary_bytes": sum(
                path.stat().st_size for path in auxiliary_files.values()
            ),
        },
        "bundle": {
            "root": str(args.bundle_root),
            "digest": manifest.digest(),
            "schema_digest": io_schema_digest(module),
            "runner": str(runner),
            "runner_sha256": _sha256(runner),
            "invalid_python_environment": True,
            "links_libpython": False,
            "verified": True,
        },
        "inputs": inputs,
        "correctness": {
            "reference_tokens": _EXPECTED_TOKENS,
            "reference_action": reference_action,
            "generated_action": parsed["action"],
            "action_maximum_absolute_error": maximum_error,
            "typed_generic_equal": parsed["typed_generic_equal"],
            "runs": args.runs,
        },
        "trace": parsed["trace"],
        "transaction": parsed["backend_failure"],
        "timing": {
            "bundle_build_seconds": build_seconds,
            "runner_seconds": run_seconds,
            "classification": "correctness audit, not benchmark",
        },
        "memory_semantics": {
            "authoritative_state_bytes": 0,
            "derived_fixed_kv_bytes": l3["memory"][
                "derived_fixed_kv_bytes"
            ],
            "transactional_output_payload_bytes": output_payload_bytes,
            "transactional_output_slots": 2,
            "transactional_output_storage_bytes": 2
            * output_payload_bytes,
            "artifact_residency": (
                "model raw wrappers load from stable bundle paths for one "
                "invocation and release CUDA constants; support packages "
                "remain session resident"
            ),
        },
        "l3_source": {
            "path": str(args.l3_report),
            "sha256": _sha256(args.l3_report),
        },
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
