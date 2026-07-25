#!/usr/bin/env python3
"""Benchmark an existing real-model generated no-Python C++ Session.

The harness reuses the exact generated Session source and artifacts from a
verified L4 bundle.  It compiles only a measurement runner, then executes that
runner with an invalid Python environment.  Timed intervals contain
``ModelSession::Run`` and its backend synchronizations; input upload, output
checksum copies, trace processing, and report generation stay outside them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.deployment import load_bundle_manifest  # noqa: E402


_SCHEMA = "vlaforge.generated_l4_benchmark/1"
_MODELS = ("smolvla", "diffusiondrive")
_MODES = ("full", "same", "new", "missing")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        raise ValueError("latency samples must be non-empty")
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            math.ceil(percentile * len(ordered)) - 1,
        ),
    )
    return ordered[index]


def _summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("latency samples must be non-empty")
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "mean_ns": mean,
        "p50_ns": _percentile(values, 0.50),
        "p90_ns": _percentile(values, 0.90),
        "p99_ns": _percentile(values, 0.99),
        "minimum_ns": min(values),
        "maximum_ns": max(values),
        "throughput_runs_per_second": 1e9 / mean,
    }


def _cpp_common(audit_runner: Path) -> str:
    return f"""#define main vlaforge_correctness_audit_main
#include "{audit_runner.as_posix()}"
#undef main

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <sys/resource.h>

namespace benchmark {{

struct Counts final {{
  std::uint64_t regions = 0u;
  std::uint64_t cache_hits = 0u;
  std::uint64_t cache_misses = 0u;
  std::uint64_t state_commits = 0u;
  std::uint64_t transaction_commits = 0u;
  std::uint64_t transaction_aborts = 0u;
  std::uint64_t output_commits = 0u;
  std::uint64_t resets = 0u;
  std::array<std::uint64_t, 8> state_versions{{}};
}};

void Count(void* context, const vlaforge::runtime::TraceEvent* event) {{
  auto* counts = static_cast<Counts*>(context);
  using vlaforge::runtime::TraceKind;
  switch (event->kind) {{
    case TraceKind::kRegion:
      ++counts->regions;
      break;
    case TraceKind::kCacheHit:
      ++counts->cache_hits;
      break;
    case TraceKind::kCacheMiss:
      ++counts->cache_misses;
      break;
    case TraceKind::kStateCommit:
      ++counts->state_commits;
      if (event->subject_id < counts->state_versions.size()) {{
        counts->state_versions[event->subject_id] =
            event->logical_version;
      }}
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
  }}
}}

std::uint64_t RssKiB() {{
  std::ifstream stream("/proc/self/status");
  std::string line;
  while (std::getline(stream, line)) {{
    if (line.rfind("VmRSS:", 0u) == 0u) {{
      std::istringstream parser(line.substr(6u));
      std::uint64_t value = 0u;
      parser >> value;
      return value;
    }}
  }}
  return 0u;
}}

std::uint64_t MaxRssKiB() {{
  struct rusage usage {{}};
  return getrusage(RUSAGE_SELF, &usage) == 0
      ? static_cast<std::uint64_t>(usage.ru_maxrss)
      : 0u;
}}

bool CudaUsed(std::uint64_t* used) {{
  std::size_t free_bytes = 0u;
  std::size_t total_bytes = 0u;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {{
    return false;
  }}
  *used = static_cast<std::uint64_t>(total_bytes - free_bytes);
  return true;
}}

std::uint64_t Revision(
    const std::string& mode, std::uint64_t iteration) {{
  return mode == "new" || mode == "full"
      ? 1000u + iteration
      : 1000u;
}}

bool RevisionPresent(const std::string& mode) {{
  return mode != "missing";
}}

void PrintSummary(
    std::uint64_t initialization_ns,
    std::uint64_t rss_initialized_kib,
    std::uint64_t rss_start_kib,
    std::uint64_t rss_end_kib,
    std::uint64_t cuda_initialized_bytes,
    std::uint64_t cuda_start_bytes,
    std::uint64_t cuda_end_bytes,
    std::uint64_t cuda_peak_sampled_bytes,
    double checksum,
    const Counts& before,
    const Counts& after) {{
  std::printf(
      "SUMMARY,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.17g,"
      "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\\n",
      static_cast<unsigned long long>(initialization_ns),
      static_cast<unsigned long long>(rss_initialized_kib),
      static_cast<unsigned long long>(rss_start_kib),
      static_cast<unsigned long long>(rss_end_kib),
      static_cast<unsigned long long>(MaxRssKiB()),
      static_cast<unsigned long long>(cuda_initialized_bytes),
      static_cast<unsigned long long>(cuda_start_bytes),
      static_cast<unsigned long long>(cuda_end_bytes),
      static_cast<unsigned long long>(cuda_peak_sampled_bytes),
      checksum,
      static_cast<unsigned long long>(after.regions - before.regions),
      static_cast<unsigned long long>(
          after.cache_hits - before.cache_hits),
      static_cast<unsigned long long>(
          after.cache_misses - before.cache_misses),
      static_cast<unsigned long long>(
          after.state_commits - before.state_commits),
      static_cast<unsigned long long>(
          after.transaction_commits - before.transaction_commits),
      static_cast<unsigned long long>(
          after.transaction_aborts - before.transaction_aborts),
      static_cast<unsigned long long>(
          after.output_commits - before.output_commits),
      static_cast<unsigned long long>(after.resets - before.resets),
      static_cast<unsigned long long>(after.state_versions[0]),
      static_cast<unsigned long long>(after.state_versions[1]));
}}

}}  // namespace benchmark
"""


def _smolvla_source(audit_runner: Path) -> str:
    return (
        _cpp_common(audit_runner)
        + r"""
int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(
        stderr,
        "usage: %s BUNDLE INPUT WARMUP SAMPLES MODE\n",
        argv[0]);
    return 2;
  }
  const std::uint64_t warmup =
      static_cast<std::uint64_t>(std::strtoull(argv[3], nullptr, 10));
  const std::uint64_t samples =
      static_cast<std::uint64_t>(std::strtoull(argv[4], nullptr, 10));
  const std::string mode(argv[5]);
  if (samples == 0u ||
      (mode != "full" && mode != "same" &&
       mode != "new" && mode != "missing")) {
    return 2;
  }

  Inputs inputs;
  if (!inputs.Initialize(argv[2])) {
    return 3;
  }
  const auto initialize_started = std::chrono::steady_clock::now();
  vlaforge_generated::ModelSession session(argv[1]);
  const auto initialize_finished = std::chrono::steady_clock::now();
  if (!session.initialization_status().ok()) {
    std::fprintf(
        stderr, "Session initialization failed: %s\n",
        session.initialization_status().message);
    return 4;
  }
  benchmark::Counts counts{};
  session.SetTraceSink({&counts, &benchmark::Count});

  const std::uint64_t total = warmup + samples;
  std::uint64_t cuda_initialized = 0u;
  if (!benchmark::CudaUsed(&cuda_initialized)) {
    return 5;
  }
  const auto rss_initialized = benchmark::RssKiB();
  std::uint64_t cuda_start = cuda_initialized;
  std::uint64_t cuda_peak = cuda_initialized;
  std::uint64_t rss_start = rss_initialized;
  benchmark::Counts measured_before{};
  double checksum = 0.0;
  for (std::uint64_t iteration = 0u; iteration < total; ++iteration) {
    if (iteration == warmup) {
      measured_before = counts;
      if (!benchmark::CudaUsed(&cuda_start)) {
        return 5;
      }
      cuda_peak = cuda_start;
      rss_start = benchmark::RssKiB();
    }
    if (mode == "full" &&
        !session.ResetEpisode(iteration + 1u).ok()) {
      return 6;
    }
    const auto revision = benchmark::Revision(mode, iteration);
    const auto bound = inputs.Bind(
        revision, benchmark::RevisionPresent(mode));
    vlaforge_generated::ModelOutputs outputs{};
    const auto started = std::chrono::steady_clock::now();
    const auto status = session.Run(bound, &outputs);
    const auto finished = std::chrono::steady_clock::now();
    if (!status.ok()) {
      std::fprintf(stderr, "Run failed: %s\n", status.message);
      return 7;
    }
    float first = 0.0f;
    if (cudaMemcpy(
            &first, outputs.action.tensor.data, sizeof(first),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
      return 8;
    }
    std::uint64_t cuda_used = 0u;
    if (!benchmark::CudaUsed(&cuda_used)) {
      return 9;
    }
    cuda_peak = std::max(cuda_peak, cuda_used);
    if (iteration >= warmup) {
      checksum += static_cast<double>(first);
      const auto latency = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              finished - started).count());
      std::printf(
          "SAMPLE,%llu,%llu,%llu,%u,%.17g\n",
          static_cast<unsigned long long>(iteration - warmup),
          static_cast<unsigned long long>(latency),
          static_cast<unsigned long long>(revision),
          benchmark::RevisionPresent(mode) ? 1u : 0u,
          static_cast<double>(first));
    }
  }
  std::uint64_t cuda_end = 0u;
  if (!benchmark::CudaUsed(&cuda_end)) {
    return 10;
  }
  benchmark::PrintSummary(
      static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              initialize_finished - initialize_started).count()),
      rss_initialized, rss_start, benchmark::RssKiB(),
      cuda_initialized, cuda_start, cuda_end, cuda_peak,
      checksum, measured_before, counts);
  return counts.transaction_aborts == 0u ? 0 : 11;
}
"""
    )


def _diffusiondrive_source(audit_runner: Path) -> str:
    return (
        _cpp_common(audit_runner)
        + r"""
int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(
        stderr,
        "usage: %s BUNDLE INPUT WARMUP SAMPLES MODE\n",
        argv[0]);
    return 2;
  }
  const std::uint64_t warmup =
      static_cast<std::uint64_t>(std::strtoull(argv[3], nullptr, 10));
  const std::uint64_t samples =
      static_cast<std::uint64_t>(std::strtoull(argv[4], nullptr, 10));
  const std::string mode(argv[5]);
  if (samples == 0u ||
      (mode != "full" && mode != "same" &&
       mode != "new" && mode != "missing")) {
    return 2;
  }

  Inputs inputs;
  if (!inputs.Initialize(argv[2])) {
    return 3;
  }
  const auto initialize_started = std::chrono::steady_clock::now();
  vlaforge_generated::ModelSession session(argv[1]);
  const auto initialize_finished = std::chrono::steady_clock::now();
  if (!session.initialization_status().ok()) {
    std::fprintf(
        stderr, "Session initialization failed: %s\n",
        session.initialization_status().message);
    return 4;
  }
  benchmark::Counts counts{};
  session.SetTraceSink({&counts, &benchmark::Count});

  const std::uint64_t total = warmup + samples;
  std::uint64_t cuda_initialized = 0u;
  if (!benchmark::CudaUsed(&cuda_initialized)) {
    return 5;
  }
  const auto rss_initialized = benchmark::RssKiB();
  std::uint64_t cuda_start = cuda_initialized;
  std::uint64_t cuda_peak = cuda_initialized;
  std::uint64_t rss_start = rss_initialized;
  benchmark::Counts measured_before{};
  double checksum = 0.0;
  for (std::uint64_t iteration = 0u; iteration < total; ++iteration) {
    if (iteration == warmup) {
      measured_before = counts;
      if (!benchmark::CudaUsed(&cuda_start)) {
        return 5;
      }
      cuda_peak = cuda_start;
      rss_start = benchmark::RssKiB();
    }
    const auto revision = benchmark::Revision(mode, iteration);
    const auto stamp = Stamp(
        revision, benchmark::RevisionPresent(mode));
    vlaforge_generated::ModelInputs bound{};
    bound.camera_feature = inputs.camera_view;
    bound.camera_feature_stamp = stamp;
    bound.lidar_feature = inputs.lidar_view;
    bound.lidar_feature_stamp = stamp;
    bound.status_feature = inputs.status_view;
    bound.status_feature_stamp = stamp;
    bound.noise = inputs.noise_view;
    bound.noise_stamp = stamp;
    vlaforge_generated::ModelOutputs outputs{};
    const auto started = std::chrono::steady_clock::now();
    const auto status = session.Run(bound, &outputs);
    const auto finished = std::chrono::steady_clock::now();
    if (!status.ok()) {
      std::fprintf(stderr, "Run failed: %s\n", status.message);
      return 6;
    }
    float first = 0.0f;
    if (cudaMemcpy(
            &first, outputs.trajectory.tensor.data, sizeof(first),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
      return 7;
    }
    std::uint64_t cuda_used = 0u;
    if (!benchmark::CudaUsed(&cuda_used)) {
      return 8;
    }
    cuda_peak = std::max(cuda_peak, cuda_used);
    if (iteration >= warmup) {
      checksum += static_cast<double>(first);
      const auto latency = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              finished - started).count());
      std::printf(
          "SAMPLE,%llu,%llu,%llu,%u,%.17g\n",
          static_cast<unsigned long long>(iteration - warmup),
          static_cast<unsigned long long>(latency),
          static_cast<unsigned long long>(revision),
          benchmark::RevisionPresent(mode) ? 1u : 0u,
          static_cast<double>(first));
    }
  }
  std::uint64_t cuda_end = 0u;
  if (!benchmark::CudaUsed(&cuda_end)) {
    return 9;
  }
  benchmark::PrintSummary(
      static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              initialize_finished - initialize_started).count()),
      rss_initialized, rss_start, benchmark::RssKiB(),
      cuda_initialized, cuda_start, cuda_end, cuda_peak,
      checksum, measured_before, counts);
  return counts.transaction_aborts == 0u ? 0 : 10;
}
"""
    )


def _cmake(bundle_root: Path) -> str:
    generated = bundle_root / "generated"
    return f"""cmake_minimum_required(VERSION 3.18)
project(vlaforge_generated_benchmark LANGUAGES C CXX)

if(NOT DEFINED VLAFORGE_RUNTIME_ROOT)
  message(FATAL_ERROR "set VLAFORGE_RUNTIME_ROOT")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(VLAFORGE_BUILD_AOTI_BACKEND ON CACHE BOOL "" FORCE)
find_package(Torch REQUIRED CONFIG)
add_subdirectory(
    "${{VLAFORGE_RUNTIME_ROOT}}"
    "${{CMAKE_CURRENT_BINARY_DIR}}/vlaforge_runtime")
add_library(
    vlaforge_benchmark_session STATIC
    "{(generated / 'session_generated.cpp').as_posix()}")
target_include_directories(
    vlaforge_benchmark_session PUBLIC "{generated.as_posix()}")
target_link_libraries(
    vlaforge_benchmark_session PUBLIC
    vlaforge_runtime vlaforge_aoti_backend)
target_compile_options(
    vlaforge_benchmark_session PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
add_executable(vlaforge_generated_benchmark benchmark.cpp)
target_link_libraries(
    vlaforge_generated_benchmark PRIVATE
    vlaforge_benchmark_session)
"""


def _build(
    *,
    model: str,
    bundle_root: Path,
    output_root: Path,
    cmake_prefix_path: Path,
) -> tuple[Path, float]:
    source = output_root / "source"
    build = output_root / "build"
    binary = output_root / "bin" / "vlaforge_generated_benchmark"
    source.mkdir(parents=True, exist_ok=True)
    binary.parent.mkdir(parents=True, exist_ok=True)
    audit_runner = (bundle_root / "generated" / "runner.cpp").resolve()
    if not audit_runner.is_file():
        raise FileNotFoundError(audit_runner)
    text = (
        _smolvla_source(audit_runner)
        if model == "smolvla"
        else _diffusiondrive_source(audit_runner)
    )
    (source / "benchmark.cpp").write_text(text, encoding="utf-8")
    (source / "CMakeLists.txt").write_text(
        _cmake(bundle_root.resolve()),
        encoding="utf-8",
    )
    started = time.perf_counter()
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            f"-DVLAFORGE_RUNTIME_ROOT={_SOURCE_ROOT}",
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--parallel", "4"],
        check=True,
    )
    build_seconds = time.perf_counter() - started
    compiled = build / "vlaforge_generated_benchmark"
    if not compiled.is_file():
        raise FileNotFoundError(compiled)
    binary.write_bytes(compiled.read_bytes())
    binary.chmod(0o755)
    return binary, build_seconds


def _parse_output(text: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    samples: list[dict[str, object]] = []
    summary: dict[str, object] | None = None
    for line in text.splitlines():
        fields = line.split(",")
        if fields[0] == "SAMPLE" and len(fields) == 6:
            samples.append(
                {
                    "index": int(fields[1]),
                    "latency_ns": int(fields[2]),
                    "revision": int(fields[3]),
                    "revision_present": fields[4] == "1",
                    "output_probe": float(fields[5]),
                }
            )
        elif fields[0] == "SUMMARY" and len(fields) == 21:
            summary = {
                "initialization_ns": int(fields[1]),
                "rss_initialized_kib": int(fields[2]),
                "rss_start_kib": int(fields[3]),
                "rss_end_kib": int(fields[4]),
                "maximum_rss_kib": int(fields[5]),
                "cuda_used_initialized_bytes": int(fields[6]),
                "cuda_used_start_bytes": int(fields[7]),
                "cuda_used_end_bytes": int(fields[8]),
                "cuda_used_peak_sampled_bytes": int(fields[9]),
                "checksum": float(fields[10]),
                "regions": int(fields[11]),
                "cache_hits": int(fields[12]),
                "cache_misses": int(fields[13]),
                "state_commits": int(fields[14]),
                "transaction_commits": int(fields[15]),
                "transaction_aborts": int(fields[16]),
                "output_commits": int(fields[17]),
                "resets": int(fields[18]),
                "state_0_version": int(fields[19]),
                "state_1_version": int(fields[20]),
            }
    if summary is None or not samples:
        raise RuntimeError(f"benchmark runner output is incomplete: {text}")
    if [item["index"] for item in samples] != list(range(len(samples))):
        raise RuntimeError("benchmark sample indices are not contiguous")
    return samples, summary


def _write_csv(path: Path, samples: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)


def _validate_runtime(
    *,
    model: str,
    mode: str,
    sample_count: int,
    samples: list[dict[str, object]],
    runtime: dict[str, object],
) -> None:
    for item in samples:
        if not math.isfinite(float(item["output_probe"])):
            raise RuntimeError("benchmark produced a non-finite output probe")
    for name in ("transaction_commits", "output_commits"):
        if int(runtime[name]) != sample_count:
            raise RuntimeError(
                f"{name}={runtime[name]} does not match {sample_count} samples"
            )
    if int(runtime["transaction_aborts"]) != 0:
        raise RuntimeError("benchmark observed a transaction abort")
    if model == "diffusiondrive":
        expected_hits = sample_count if mode == "same" else 0
        expected_misses = 0 if mode == "same" else sample_count
        if (
            int(runtime["cache_hits"]) != expected_hits
            or int(runtime["cache_misses"]) != expected_misses
        ):
            raise RuntimeError(
                "DiffusionDrive revision/cache accounting disagrees with mode"
            )
    elif mode == "full":
        if int(runtime["resets"]) != sample_count:
            raise RuntimeError(
                "SmolVLA full mode must reset before every measured Run"
            )
        if int(runtime["state_commits"]) != 2 * sample_count:
            raise RuntimeError(
                "SmolVLA full mode must commit queue and cursor per Run"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=_MODELS, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--mode", choices=_MODES, default="full")
    parser.add_argument("--reuse-binary", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup must be non-negative and samples positive")
    for required in (
        args.bundle_root / "bundle.json",
        args.bundle_root / "generated" / "session_generated.cpp",
        args.bundle_root / "generated" / "runner.cpp",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    import torch

    binary = args.output_root / "bin" / "vlaforge_generated_benchmark"
    if args.reuse_binary:
        if not binary.is_file():
            raise FileNotFoundError(binary)
        build_seconds = 0.0
    else:
        binary, build_seconds = _build(
            model=args.model,
            bundle_root=args.bundle_root,
            output_root=args.output_root,
            cmake_prefix_path=Path(torch.utils.cmake_prefix_path),
        )

    command = [
        str(binary),
        str(args.bundle_root),
        str(args.input_root),
        str(args.warmup),
        str(args.samples),
        args.mode,
    ]
    environment = {
        **dict(os.environ),
        "PYTHONHOME": "/definitely/not/a/python/home",
        "PYTHONPATH": "/definitely/not/a/python/path",
    }
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    elapsed_seconds = time.perf_counter() - started
    samples, runtime = _parse_output(completed.stdout)
    if len(samples) != args.samples:
        raise RuntimeError("benchmark sample count mismatch")
    _validate_runtime(
        model=args.model,
        mode=args.mode,
        sample_count=args.samples,
        samples=samples,
        runtime=runtime,
    )
    linked = subprocess.run(
        ["ldd", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "libpython" in linked.lower():
        raise RuntimeError("benchmark runner links libpython")
    manifest = load_bundle_manifest(args.bundle_root / "bundle.json")
    latencies = [int(item["latency_ns"]) for item in samples]
    report = {
        "schema": _SCHEMA,
        "status": "passed",
        "model": args.model,
        "path": "generated_no_python_cpp_session",
        "mode": args.mode,
        "warmup": args.warmup,
        "samples": args.samples,
        "latency": _summary(latencies),
        "runtime": runtime,
        "memory": {
            "warmup_rss_residency_kib": (
                int(runtime["rss_start_kib"])
                - int(runtime["rss_initialized_kib"])
            ),
            "rss_drift_kib": (
                int(runtime["rss_end_kib"])
                - int(runtime["rss_start_kib"])
            ),
            "warmup_cuda_residency_bytes": (
                int(runtime["cuda_used_start_bytes"])
                - int(runtime["cuda_used_initialized_bytes"])
            ),
            "cuda_used_drift_bytes": (
                int(runtime["cuda_used_end_bytes"])
                - int(runtime["cuda_used_start_bytes"])
            ),
            "static_arena": (
                manifest.compilation_certificate.arena.to_dict()
            ),
        },
        "bundle": {
            "root": str(args.bundle_root.resolve()),
            "digest": manifest.digest(),
            "source_revision": manifest.reproducibility.source_revision,
            "source_dirty": manifest.reproducibility.source_dirty,
            "io_schema_digest": manifest.io_schema_digest,
            "artifact_bytes": sum(
                item.artifact_size_bytes
                for item in manifest.region_artifacts
            ),
        },
        "runner": {
            "path": str(binary.resolve()),
            "sha256": _sha256(binary),
            "links_libpython": False,
            "invalid_python_environment": True,
            "build_seconds": build_seconds,
            "wall_seconds": elapsed_seconds,
        },
        "environment": {
            "host": platform.platform(),
            "python_orchestrator": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "driver": subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
        "reproduction": {
            "command": command,
            "orchestrator": [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ],
        },
        "classification": (
            "Host-CUDA generated Session benchmark; no Orin claim and "
            "no model-kernel optimization attribution"
        ),
    }
    stem = f"{args.model}_{args.mode}_{args.samples}"
    json_path = args.output_root / f"{stem}.json"
    csv_path = args.output_root / f"{stem}.csv"
    stdout_path = args.output_root / f"{stem}.stdout"
    ldd_path = args.output_root / f"{stem}.ldd"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, samples)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    ldd_path.write_text(linked, encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
