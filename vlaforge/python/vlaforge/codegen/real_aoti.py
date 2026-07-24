"""Generated no-Python C++ runners for real AOTInductor VLA artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from vlaforge.adapters import build_real_smolvla_action_program
from vlaforge.codegen.model import GeneratedSources
from vlaforge.compiler import (
    CompilationCertificate,
    CompilerProfile,
    compile_module,
)
from vlaforge.ir.program import Block, Module
from vlaforge.transforms import synthesize_epoch_memoization


@dataclass(frozen=True, slots=True)
class AotiTensorSpec:
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("AOTI tensor shape must be static and non-negative")
        if self.dtype not in {
            "bool",
            "i32",
            "i64",
            "f16",
            "bf16",
            "f32",
            "f64",
        }:
            raise ValueError(f"unsupported AOTI tensor dtype: {self.dtype}")


@dataclass(frozen=True, slots=True)
class TemporalCacheDependencySpec:
    kind: str
    subject_id: int
    max_age_ns: int | None


@dataclass(frozen=True, slots=True)
class SmolVLAAotiSpec:
    image: AotiTensorSpec
    state: AotiTensorSpec
    token_ids: tuple[int, ...]
    token_mask: tuple[bool, ...]
    prefix_outputs: tuple[AotiTensorSpec, ...]
    solver_output: AotiTensorSpec
    action_chunk: AotiTensorSpec
    num_steps: int = 10

    def __post_init__(self) -> None:
        if self.image != AotiTensorSpec((1, 3, 256, 256), "f32"):
            raise ValueError("SmolVLA deployment profile requires 1x3x256x256 f32")
        if self.state != AotiTensorSpec((1, 6), "f32"):
            raise ValueError("SmolVLA deployment profile requires 1x6 f32 state")
        if not self.token_ids or len(self.token_ids) != len(self.token_mask):
            raise ValueError("SmolVLA tokens and mask must be non-empty and aligned")
        if not self.prefix_outputs or self.prefix_outputs[0].dtype != "bool":
            raise ValueError("SmolVLA prefix output zero must be the pad mask")
        if self.solver_output != AotiTensorSpec((1, 50, 32), "f32"):
            raise ValueError("SmolVLA solver profile must be 1x50x32 f32")
        if self.action_chunk != AotiTensorSpec((1, 50, 6), "f32"):
            raise ValueError("SmolVLA action profile must be 1x50x6 f32")
        if self.num_steps < 1:
            raise ValueError("SmolVLA solver must have a positive bound")


def smolvla_spec_from_exported_programs(
    prefix_program: object,
    solver_program: object,
    trim_program: object,
) -> SmolVLAAotiSpec:
    import torch

    prefix_args = tuple(prefix_program.example_inputs[0])
    if len(prefix_args) != 4:
        raise ValueError("SmolVLA prefix export must have four user inputs")
    with torch.inference_mode():
        prefix_outputs = _as_tuple(prefix_program.module()(*prefix_args))
        solver_outputs = _as_tuple(
            solver_program.module()(*solver_program.example_inputs[0])
        )
        trim_outputs = _as_tuple(
            trim_program.module()(*trim_program.example_inputs[0])
        )
    if len(solver_outputs) != 1 or len(trim_outputs) != 1:
        raise ValueError("SmolVLA solver and trim exports require one output")
    return SmolVLAAotiSpec(
        image=_tensor_spec(prefix_args[0]),
        state=_tensor_spec(prefix_args[1]),
        token_ids=tuple(int(value) for value in prefix_args[2].cpu().flatten()),
        token_mask=tuple(
            bool(value) for value in prefix_args[3].cpu().flatten()
        ),
        prefix_outputs=tuple(_tensor_spec(value) for value in prefix_outputs),
        solver_output=_tensor_spec(solver_outputs[0]),
        action_chunk=_tensor_spec(trim_outputs[0]),
    )


def generate_real_smolvla_aoti_runner(
    spec: SmolVLAAotiSpec,
    *,
    compiler_profile: CompilerProfile | str = CompilerProfile.VERIFIED,
    allow_test_profile: bool = False,
    optimization_benchmark: bool = False,
) -> GeneratedSources:
    module = build_real_smolvla_action_program(
        chunk_size=spec.action_chunk.shape[1],
        max_action_dim=spec.solver_output.shape[2],
        output_action_dim=spec.action_chunk.shape[2],
        num_steps=spec.num_steps,
    )
    compilation = compile_module(
        module,
        profile=compiler_profile,
        allow_test_profile=allow_test_profile,
    )
    plan = compilation.plan
    tasks = {
        "input_batch": _task_id(plan, "vla.sample_input", stream="batch"),
        "input_noise": _task_id(plan, "vla.sample_input", stream="noise"),
        "begin": _task_id(plan, "vla.txn.begin"),
        "read_queue": _task_id(
            plan, "vla.state.read", state="action_queue"
        ),
        "read_cursor": _task_id(
            plan, "vla.state.read", state="queue_cursor"
        ),
        "queue_empty": _task_id(
            plan, "vla.invoke", region="queue_is_empty"
        ),
        "queue_zero": _task_id(plan, "vla.invoke", region="queue_zero"),
        "prefix": _task_id(plan, "vla.invoke", region="prepare_prefix"),
        "solver": _task_id(plan, "vla.invoke", region="solver_step"),
        "trim": _task_id(
            plan, "vla.invoke", region="trim_action_chunk"
        ),
        "select_refill": _task_id(
            plan, "vla.invoke", region="queue_select", occurrence=0
        ),
        "advance_refill": _task_id(
            plan, "vla.invoke", region="queue_advance", occurrence=0
        ),
        "select_reuse": _task_id(
            plan, "vla.invoke", region="queue_select", occurrence=1
        ),
        "advance_reuse": _task_id(
            plan, "vla.invoke", region="queue_advance", occurrence=1
        ),
        "stage_queue": _task_id(
            plan, "vla.state.stage_write", state="action_queue"
        ),
        "stage_cursor": _task_id(
            plan, "vla.state.stage_write", state="queue_cursor"
        ),
        "validate": _task_id(plan, "vla.validate"),
        "action": _task_id(plan, "vla.action.create"),
        "commit": _task_id(plan, "vla.txn.commit"),
        "publish": _task_id(plan, "vla.action.publish"),
    }
    cache_dependencies = certified_cache_dependencies(
        compilation.certificate,
        "prepare_prefix",
    )
    source = _smolvla_source(
        spec,
        tasks,
        optimization_enabled=bool(cache_dependencies),
        optimization_benchmark=optimization_benchmark,
        cache_dependencies=cache_dependencies,
        certificate_digest=compilation.certificate.digest(),
    )
    return GeneratedSources(
        (
            ("CMakeLists.txt", _real_aoti_cmake()),
            (
                "compilation_certificate.json",
                compilation.certificate.canonical_json(indent=2) + "\n",
            ),
            ("runner.cpp", source),
        )
    )


def _smolvla_source(
    spec: SmolVLAAotiSpec,
    tasks: dict[str, int],
    *,
    optimization_enabled: bool,
    optimization_benchmark: bool,
    cache_dependencies: tuple[TemporalCacheDependencySpec, ...],
    certificate_digest: str,
) -> str:
    token_values = ", ".join(str(value) for value in spec.token_ids)
    mask_values = ", ".join("1u" if value else "0u" for value in spec.token_mask)
    prefix_allocations = "\n".join(
        f"  prefix_outputs.push_back(at::empty({_shape(specification.shape)}, "
        f"options.dtype({_at_dtype(specification.dtype)})));"
        for specification in spec.prefix_outputs
    )
    prefix_pointers = "\n".join(
        f"  prefix_output_ptrs.push_back(&prefix_outputs[{index}]);"
        for index in range(len(spec.prefix_outputs))
    )
    cache_pointers = "\n".join(
        f"  solver_inputs.push_back(&prefix_outputs[{index}]);"
        for index in range(1, len(spec.prefix_outputs))
    )
    task_constants = "\n".join(
        f"constexpr std::uint32_t kTask{_camel(name)} = {task_id}u;"
        for name, task_id in tasks.items()
    )
    benchmark_runtime_header = (
        '#include "vlaforge/runtime/epoch_cache.h"\n'
        if optimization_enabled or optimization_benchmark
        else ""
    )
    benchmark_standard_headers = (
        "#include <chrono>\n#include <cstdlib>\n"
        if optimization_benchmark
        else ""
    )
    benchmark_setup = (
        """  const char* benchmark_mode =
      std::getenv("VLAFORGE_OPT_BENCHMARK");
  const bool benchmark_enabled = benchmark_mode != nullptr;
  const bool benchmark_cache =
      benchmark_enabled &&
      (std::strcmp(benchmark_mode, "cache") == 0 ||
       std::strcmp(benchmark_mode, "combined") == 0);
  const bool benchmark_licm_disabled =
      benchmark_enabled &&
      (std::strcmp(benchmark_mode, "off") == 0 ||
       std::strcmp(benchmark_mode, "cache") == 0 ||
       std::strcmp(benchmark_mode, "licm_off") == 0);
  const char* workload_value = std::getenv("VLAFORGE_BENCH_WORKLOAD");
  const std::string benchmark_workload =
      workload_value == nullptr ? "nominal" : workload_value;
  const char* iteration_value = std::getenv("VLAFORGE_BENCH_ITERATIONS");
  const std::uint64_t benchmark_iterations =
      iteration_value == nullptr
          ? 3u
          : std::max<std::uint64_t>(
                3u, std::strtoull(iteration_value, nullptr, 10));
  vlaforge::runtime::EpochVersionCacheGuard prefix_cache;
"""
        if optimization_benchmark
        else (
            """  constexpr bool benchmark_cache = true;
  vlaforge::runtime::EpochVersionCacheGuard prefix_cache;
"""
            if optimization_enabled
            else ""
        )
    )
    timer_start = (
        """    const auto benchmark_started =
        std::chrono::steady_clock::now();
"""
        if optimization_benchmark
        else ""
    )
    dependency_qualifier = "" if optimization_benchmark else "const "
    benchmark_tick_start = (
        timer_start
        + f"""    {dependency_qualifier}vlaforge::runtime::TemporalDependency cache_dependencies[] = {{
"""
        + render_cpp_cache_dependencies(cache_dependencies)
        + """
    };
"""
        + (
            """    const std::uint64_t dependency_sequence =
        benchmark_workload == "all-miss"
            ? sequence
            : (benchmark_workload == "nominal" ? sequence / 2u : 0u);
    const std::uint64_t dependency_timestamp =
        benchmark_workload == "all-miss"
            ? tick.timestamp_ns
            : (benchmark_workload == "nominal"
                   ? dependency_sequence * 2000000u
                   : 0u);
    for (auto& dependency : cache_dependencies) {
      dependency.logical_version = dependency_sequence;
      dependency.epoch.sequence = dependency_sequence;
      dependency.epoch.timestamp_ns = dependency_timestamp;
      dependency.epoch.episode = tick.episode;
    }
"""
            if optimization_benchmark
            else ""
        )
        + """
    bool benchmark_cache_hit = false;
"""
        if optimization_enabled or optimization_benchmark
        else ""
    )
    force_refill = " || benchmark_enabled" if optimization_benchmark else ""
    iteration_limit = (
        "benchmark_iterations" if optimization_benchmark else "3u"
    )
    tick_timestamp = (
        '(benchmark_workload == "stale" '
        "? sequence * 100000000u : sequence * 1000000u)"
        if optimization_benchmark
        else "sequence * kControlPeriodNs"
    )
    trace_capacity = 4096 if optimization_benchmark else 1024
    benchmark_cache_lookup = (
        """      benchmark_cache_hit =
          benchmark_cache &&
          prefix_cache.Lookup(
              cache_dependencies, """
        + f"{len(cache_dependencies)}u"
        + """, tick);
      if (!benchmark_cache_hit) {
"""
        if optimization_enabled or optimization_benchmark
        else ""
    )
    benchmark_preheader_open = (
        "        if (!benchmark_licm_disabled) {\n"
        if optimization_benchmark
        else ""
    )
    benchmark_preheader_close = (
        "        }\n"
        if optimization_enabled or optimization_benchmark
        else ""
    )
    benchmark_cache_update = (
        """        if (benchmark_cache &&
            !Check(prefix_cache.Update(
                       cache_dependencies,
                       """
        + f"{len(cache_dependencies)}u"
        + """),
                   "update prefix cache")) {
          return 19;
        }
      }
"""
        if optimization_benchmark
        else ""
    )
    benchmark_loop_prefix = (
        """        if (benchmark_licm_disabled &&
            !benchmark_cache_hit) {
          if (!prefix_region.Run(
                  prefix_inputs, prefix_output_ptrs, true)) {
            return 20;
          }
          EmitRegion(trace, kTaskPrefix, transaction.id(), tick);
          if (!refill_recorded && step == 0) {
            for (const auto& output : prefix_outputs) {
              if (!DumpTensor(evidence, output)) {
                return 21;
              }
            }
          }
        }
"""
        if optimization_benchmark
        else ""
    )
    benchmark_tick_end = (
        """    if (benchmark_enabled) {
      const auto benchmark_finished =
          std::chrono::steady_clock::now();
      const double elapsed_us =
          std::chrono::duration<double, std::micro>(
              benchmark_finished - benchmark_started)
              .count();
      std::printf("BENCH_TICK_US,%s,%llu,%.9f,%u\\n",
                  benchmark_mode,
                  static_cast<unsigned long long>(sequence),
                  elapsed_us,
                  benchmark_cache_hit ? 1u : 0u);
    }
"""
        if optimization_benchmark
        else ""
    )
    return f"""#include "vlaforge/backends/aoti_region_executable.h"
#include "vlaforge/runtime/action_queue.h"
{benchmark_runtime_header}#include "vlaforge/runtime/state_store.h"
#include "vlaforge/runtime/static_arena.h"
#include "vlaforge/runtime/transaction.h"

#include <ATen/ATen.h>
#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <array>
{benchmark_standard_headers}#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {{

{task_constants}
[[maybe_unused]] constexpr char kCompilationCertificateDigest[] =
    "{certificate_digest}";
constexpr std::uint32_t kControlClock = 1u;
constexpr std::uint64_t kControlPeriodNs = 20000000u;
constexpr std::size_t kQueueElements = 300u;
constexpr std::size_t kActionElements = 6u;

struct TraceCollector {{
  std::array<vlaforge::runtime::TraceEvent, {trace_capacity}> events{{}};
  std::size_t count = 0u;
}};

void CollectTrace(void* context,
                  const vlaforge::runtime::TraceEvent* event) {{
  auto* collector = static_cast<TraceCollector*>(context);
  if (collector->count < collector->events.size()) {{
    collector->events[collector->count++] = *event;
  }}
}}

bool Check(VLAForgeStatus status, const char* operation) {{
  if (status.code == VLAFORGE_STATUS_OK) {{
    return true;
  }}
  std::fprintf(stderr, "%s failed: %.*s\\n", operation,
               static_cast<int>(status.message_size), status.message);
  return false;
}}

bool Check(const vlaforge::runtime::Status& status,
           const char* operation) {{
  if (status.ok()) {{
    return true;
  }}
  std::fprintf(stderr, "%s failed: %s (subject=%u)\\n", operation,
               status.message, status.subject_id);
  return false;
}}

VLAForgeDType DType(const at::Tensor& tensor) {{
  switch (tensor.scalar_type()) {{
    case at::kBool: return VLAFORGE_DTYPE_BOOL;
    case at::kInt: return VLAFORGE_DTYPE_I32;
    case at::kLong: return VLAFORGE_DTYPE_I64;
    case at::kHalf: return VLAFORGE_DTYPE_F16;
    case at::kBFloat16: return VLAFORGE_DTYPE_BF16;
    case at::kFloat: return VLAFORGE_DTYPE_F32;
    case at::kDouble: return VLAFORGE_DTYPE_F64;
    default: return VLAFORGE_DTYPE_INVALID;
  }}
}}

VLAForgeTensorView View(at::Tensor& tensor) {{
  return VLAForgeTensorView{{
      tensor.data_ptr(),
      static_cast<std::uint64_t>(tensor.nbytes()),
      tensor.sizes().data(),
      static_cast<std::uint32_t>(tensor.dim()),
      DType(tensor),
      {{tensor.is_cuda() ? VLAFORGE_DEVICE_CUDA : VLAFORGE_DEVICE_CPU,
        tensor.is_cuda() ? tensor.get_device() : 0}},
  }};
}}

class Region final {{
 public:
  Region() = default;
  ~Region() {{
    if (executable_ != nullptr) {{
      api_->destroy(executable_);
    }}
  }}
  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;

  bool Load(std::uint32_t region_id, const char* path) {{
    api_ = vlaforge_aoti_region_executable_api();
    if (!Check(vlaforge_region_executable_api_validate(api_),
               "validate AOTI API")) {{
      return false;
    }}
    const VLAForgeRegionCreateOptions options{{
        sizeof(VLAForgeRegionCreateOptions),
        VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
        region_id,
        {{VLAFORGE_DEVICE_CUDA, 0}},
    }};
    if (!Check(api_->create(&options, &executable_), "create region")) {{
      return false;
    }}
    const std::string artifact_path(path);
    const VLAForgeArtifactDescriptor artifact{{
        sizeof(VLAForgeArtifactDescriptor),
        VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
        artifact_path.data(),
        artifact_path.size(),
        nullptr,
        0u,
    }};
    return Check(api_->load(executable_, &artifact), "load region");
  }}

  bool Run(const std::vector<at::Tensor*>& inputs,
           const std::vector<at::Tensor*>& outputs,
           bool synchronize) {{
    for (std::size_t index = 0; index < inputs.size(); ++index) {{
      auto view = View(*inputs[index]);
      if (!Check(api_->bind_input(
                     executable_, static_cast<std::uint32_t>(index), &view),
                 "bind region input")) {{
        return false;
      }}
    }}
    for (std::size_t index = 0; index < outputs.size(); ++index) {{
      auto view = View(*outputs[index]);
      if (!Check(api_->bind_output(
                     executable_, static_cast<std::uint32_t>(index), &view),
                 "bind region output")) {{
        return false;
      }}
    }}
    if (!Check(api_->run(executable_), "run region")) {{
      return false;
    }}
    return !synchronize ||
           Check(api_->synchronize(executable_), "synchronize region");
  }}

 private:
  const VLAForgeRegionExecutableApi* api_ = nullptr;
  VLAForgeRegionExecutable* executable_ = nullptr;
}};

void EmitRegion(vlaforge::runtime::TraceSink trace,
                std::uint32_t task_id,
                std::uint64_t transaction_id,
                const vlaforge::runtime::Epoch& tick) {{
  vlaforge::runtime::EmitTrace(
      trace,
      vlaforge::runtime::TraceEvent{{
          vlaforge::runtime::TraceKind::kRegion,
          task_id, 0u, 0u, transaction_id, tick}});
}}

void EmitSimple(vlaforge::runtime::TraceSink trace,
                vlaforge::runtime::TraceKind kind,
                std::uint32_t task_id,
                std::uint64_t transaction_id,
                const vlaforge::runtime::Epoch& tick) {{
  vlaforge::runtime::EmitTrace(
      trace,
      vlaforge::runtime::TraceEvent{{
          kind, task_id, 0u, 0u, transaction_id, tick}});
}}

bool DumpTensor(std::ofstream& output, const at::Tensor& tensor) {{
  const at::Tensor cpu = tensor.detach().to(at::kCPU).contiguous();
  output.write(static_cast<const char*>(cpu.const_data_ptr()),
               static_cast<std::streamsize>(cpu.nbytes()));
  return output.good();
}}

}}  // namespace

int main(int argc, char** argv) {{
  if (argc != 5) {{
    std::fprintf(
        stderr,
        "usage: %s PREFIX.pt2 SOLVER.pt2 TRIM.pt2 EVIDENCE.bin\\n",
        argv[0]);
    return 2;
  }}
  const c10::cuda::CUDAGuard guard(0);
  const auto options =
      at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  at::Tensor image =
      at::linspace(0.0, 1.0, 3 * 256 * 256, options).reshape({{1, 3, 256, 256}});
  at::Tensor state = at::linspace(-0.2, 0.3, 6, options).reshape({{1, 6}});
  constexpr std::int64_t kTokenIds[] = {{{token_values}}};
  constexpr std::uint8_t kTokenMask[] = {{{mask_values}}};
  at::Tensor token_ids =
      at::from_blob(
          const_cast<std::int64_t*>(kTokenIds),
          {{1, {len(spec.token_ids)}}},
          at::TensorOptions().dtype(at::kLong))
          .clone()
          .to(at::kCUDA);
  at::Tensor token_mask =
      at::from_blob(
          const_cast<std::uint8_t*>(kTokenMask),
          {{1, {len(spec.token_mask)}}},
          at::TensorOptions().dtype(at::kByte))
          .to(at::kBool)
          .to(at::kCUDA);
  at::Tensor noise =
      at::linspace(-1.0, 1.0, 50 * 32, options).reshape({{1, 50, 32}});
  at::Tensor timestep = at::ones({{1}}, options);
  at::Tensor sample = at::empty({{1, 50, 32}}, options);
  at::Tensor next_sample = at::empty_like(sample);
  at::Tensor action_chunk = at::empty({{1, 50, 6}}, options);

  std::vector<at::Tensor> prefix_outputs;
  prefix_outputs.reserve({len(spec.prefix_outputs)}u);
{prefix_allocations}
  std::vector<at::Tensor*> prefix_inputs{{
      &image, &state, &token_ids, &token_mask}};
  std::vector<at::Tensor*> prefix_output_ptrs;
  prefix_output_ptrs.reserve(prefix_outputs.size());
{prefix_pointers}
  std::vector<at::Tensor*> solver_inputs{{
      &prefix_outputs[0], &sample, &timestep}};
{cache_pointers}
  std::vector<at::Tensor*> solver_outputs{{&next_sample}};
  std::vector<at::Tensor*> trim_inputs{{&sample}};
  std::vector<at::Tensor*> trim_outputs{{&action_chunk}};

  Region prefix_region;
  Region solver_region;
  Region trim_region;
  if (!prefix_region.Load(0u, argv[1]) ||
      !solver_region.Load(1u, argv[2]) ||
      !trim_region.Load(2u, argv[3])) {{
    return 3;
  }}
{benchmark_setup}  std::ofstream evidence(argv[4], std::ios::binary | std::ios::trunc);
  if (!evidence) {{
    return 4;
  }}

  TraceCollector trace_collector;
  const vlaforge::runtime::TraceSink trace{{
      &trace_collector, &CollectTrace}};
  vlaforge::runtime::StaticArena state_arena(6100u, 64u);
  const vlaforge::runtime::StateSlotDescriptor state_slots[] = {{
      {{0u, 5u, 1216u, 64u, 0u, true}},
      {{1u, 5u, 4u, 4u, 6080u, true}},
  }};
  vlaforge::runtime::StateStore state_store(
      state_arena, state_slots, 2u, trace);
  vlaforge::runtime::Transaction transaction(2u);
  vlaforge::runtime::ActionQueue action_queue(nullptr, nullptr, trace);
  std::array<float, kQueueElements> initial_queue{{}};
  std::int32_t initial_cursor = 50;
  const vlaforge::runtime::Epoch initial_epoch{{
      kControlClock, 0u, 0u, 0u}};
  if (!Check(state_store.Initialize(
                 0u, initial_epoch, initial_queue.data(),
                 initial_queue.size() * sizeof(float)),
             "initialize action queue") ||
      !Check(state_store.Initialize(
                 1u, initial_epoch, &initial_cursor,
                 sizeof(initial_cursor)),
             "initialize cursor")) {{
    return 5;
  }}

  bool refill_recorded = false;
  for (std::uint64_t sequence = 0u;
       sequence < {iteration_limit}; ++sequence) {{
    const vlaforge::runtime::Epoch tick{{
        kControlClock, sequence, {tick_timestamp}, 0u}};
{benchmark_tick_start}    EmitSimple(trace, vlaforge::runtime::TraceKind::kInput,
               kTaskInputBatch, 0u, tick);
    EmitSimple(trace, vlaforge::runtime::TraceKind::kInput,
               kTaskInputNoise, 0u, tick);
    if (!Check(state_store.Begin(&transaction, tick, kTaskBegin),
               "begin transaction")) {{
      return 6;
    }}
    vlaforge::runtime::StateSnapshot queue_snapshot;
    vlaforge::runtime::StateSnapshot cursor_snapshot;
    if (!Check(state_store.ReadLatest(
                   0u, 0u, sequence, false, kTaskReadQueue,
                   &queue_snapshot),
               "read queue") ||
        !Check(state_store.ReadLatest(
                   1u, 0u, sequence, false, kTaskReadCursor,
                   &cursor_snapshot),
               "read cursor")) {{
      return 7;
    }}
    std::array<float, kQueueElements> queue_next{{}};
    std::memcpy(queue_next.data(), queue_snapshot.data,
                queue_next.size() * sizeof(float));
    std::int32_t cursor =
        *static_cast<const std::int32_t*>(cursor_snapshot.data);
    const bool queue_empty = cursor >= 50{force_refill};
    EmitRegion(trace, kTaskQueueEmpty, transaction.id(), tick);
    EmitRegion(trace, kTaskQueueZero, transaction.id(), tick);
    std::uint32_t select_task = kTaskSelectReuse;
    std::uint32_t advance_task = kTaskAdvanceReuse;

    if (queue_empty) {{
{benchmark_cache_lookup}{benchmark_preheader_open}      if (!prefix_region.Run(
              prefix_inputs, prefix_output_ptrs, true)) {{
        return 8;
      }}
      EmitRegion(trace, kTaskPrefix, transaction.id(), tick);
      if (!refill_recorded) {{
        for (const auto& output : prefix_outputs) {{
          if (!DumpTensor(evidence, output)) {{
            return 9;
          }}
        }}
      }}
{benchmark_preheader_close}{benchmark_cache_update}      sample.copy_(noise);
      for (std::int64_t step = 0; step < {spec.num_steps}; ++step) {{
{benchmark_loop_prefix}        timestep.fill_(1.0 - static_cast<double>(step) /
                                 static_cast<double>({spec.num_steps}));
        solver_inputs[1] = &sample;
        solver_outputs[0] = &next_sample;
        if (!solver_region.Run(solver_inputs, solver_outputs, true)) {{
          return 10;
        }}
        EmitRegion(trace, kTaskSolver, transaction.id(), tick);
        if (!refill_recorded && !DumpTensor(evidence, next_sample)) {{
          return 11;
        }}
        std::swap(sample, next_sample);
      }}
      trim_inputs[0] = &sample;
      if (!trim_region.Run(trim_inputs, trim_outputs, true)) {{
        return 12;
      }}
      EmitRegion(trace, kTaskTrim, transaction.id(), tick);
      if (!refill_recorded && !DumpTensor(evidence, action_chunk)) {{
        return 13;
      }}
      const at::Tensor queue_cpu =
          action_chunk.to(at::kCPU).contiguous().view({{-1}});
      std::memcpy(queue_next.data(), queue_cpu.const_data_ptr<float>(),
                  queue_next.size() * sizeof(float));
      cursor = 0;
      select_task = kTaskSelectRefill;
      advance_task = kTaskAdvanceRefill;
      refill_recorded = true;
    }}
    EmitRegion(trace, select_task, transaction.id(), tick);
    const float* selected =
        queue_next.data() + static_cast<std::size_t>(cursor) * kActionElements;
    ++cursor;
    EmitRegion(trace, advance_task, transaction.id(), tick);
    const vlaforge::runtime::Epoch state_epoch{{
        kControlClock, sequence + 1u,
        sequence * kControlPeriodNs, 0u}};
    if (!Check(state_store.Stage(
                   &transaction, 0u, state_epoch, queue_next.data(),
                   queue_next.size() * sizeof(float), kTaskStageQueue),
               "stage queue") ||
        !Check(state_store.Stage(
                   &transaction, 1u, state_epoch, &cursor,
                   sizeof(cursor), kTaskStageCursor),
               "stage cursor")) {{
      return 14;
    }}
    bool valid = true;
    for (std::size_t index = 0; index < kActionElements; ++index) {{
      valid = valid && std::isfinite(selected[index]);
    }}
    EmitSimple(trace, vlaforge::runtime::TraceKind::kValidation,
               kTaskValidate, transaction.id(), tick);
    EmitSimple(trace, vlaforge::runtime::TraceKind::kActionPending,
               kTaskAction, transaction.id(), tick);
    vlaforge::runtime::CommittedAction committed;
    if (!Check(state_store.Commit(
                   &transaction,
                   vlaforge::runtime::PendingAction{{
                       tick, selected, kActionElements * sizeof(float)}},
                   valid, kTaskCommit, &committed),
               "commit") ||
        !Check(action_queue.Publish(committed, kTaskPublish),
               "publish")) {{
      return 15;
    }}
    evidence.write(
        reinterpret_cast<const char*>(selected),
        static_cast<std::streamsize>(kActionElements * sizeof(float)));
    if (!evidence.good()) {{
      return 16;
    }}
    std::printf("ACTION,%llu",
                static_cast<unsigned long long>(sequence));
    for (std::size_t index = 0; index < kActionElements; ++index) {{
      std::printf(",%.9g", static_cast<double>(selected[index]));
    }}
    std::printf("\\n");
{benchmark_tick_end}  }}

  if (!Check(state_store.ResetEpisode(1u, 0u), "reset episode")) {{
    return 17;
  }}
  action_queue.Reset();
  vlaforge::runtime::StateSnapshot reset_snapshot;
  if (state_store.ReadLatest(
          0u, 1u, 0u, false, 0u, &reset_snapshot).code !=
      vlaforge::runtime::StatusCode::kNotFound) {{
    return 18;
  }}
  std::printf("RESET,1\\n");

  for (std::size_t index = 0; index < trace_collector.count; ++index) {{
    const auto& event = trace_collector.events[index];
    std::printf(
        "TRACE,%u,%u,%u,%llu,%llu,%u,%llu,%llu,%llu\\n",
        static_cast<unsigned>(event.kind), event.task_id, event.state_id,
        static_cast<unsigned long long>(event.logical_version),
        static_cast<unsigned long long>(event.transaction_id),
        event.epoch.clock_id,
        static_cast<unsigned long long>(event.epoch.sequence),
        static_cast<unsigned long long>(event.epoch.timestamp_ns),
        static_cast<unsigned long long>(event.epoch.episode));
  }}
  return 0;
}}
"""


def _real_aoti_cmake() -> str:
    return """cmake_minimum_required(VERSION 3.18)
project(vlaforge_real_aoti_runner LANGUAGES C CXX)

if(NOT DEFINED VLAFORGE_RUNTIME_ROOT)
  message(FATAL_ERROR "set VLAFORGE_RUNTIME_ROOT to the VLAForge source root")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(VLAFORGE_BUILD_AOTI_BACKEND ON CACHE BOOL "" FORCE)

find_package(Torch REQUIRED CONFIG)
add_subdirectory("${VLAFORGE_RUNTIME_ROOT}"
                 "${CMAKE_CURRENT_BINARY_DIR}/vlaforge_runtime")

add_executable(vlaforge_real_aoti_runner runner.cpp)
target_link_libraries(vlaforge_real_aoti_runner PRIVATE
    vlaforge_aoti_backend)
target_compile_options(vlaforge_real_aoti_runner PRIVATE
    ${VLAFORGE_TORCH_CXX_FLAGS}
    -Wall -Wextra -Wpedantic -Werror)
"""


def _task_id(
    plan: object,
    opcode: str,
    *,
    occurrence: int = 0,
    **attributes: object,
) -> int:
    candidates = [
        task
        for task in plan.tasks
        if task.opcode == opcode
        and all(task.attributes.get(key) == value for key, value in attributes.items())
    ]
    if occurrence < 0 or occurrence >= len(candidates):
        raise ValueError(
            f"missing task {opcode} occurrence={occurrence} attributes={attributes}"
        )
    return int(candidates[occurrence].id)


def _tensor_spec(value: object) -> AotiTensorSpec:
    dtype = str(value.dtype).removeprefix("torch.")
    mapping = {
        "bool": "bool",
        "int32": "i32",
        "int64": "i64",
        "float16": "f16",
        "bfloat16": "bf16",
        "float32": "f32",
        "float64": "f64",
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported exported tensor dtype: {dtype}")
    return AotiTensorSpec(
        tuple(int(dimension) for dimension in value.shape),
        mapping[dtype],
    )


def _as_tuple(value: object) -> tuple[object, ...]:
    return value if isinstance(value, tuple) else (value,)


def _shape(shape: tuple[int, ...]) -> str:
    return "{" + ", ".join(str(dimension) for dimension in shape) + "}"


def _at_dtype(dtype: str) -> str:
    return {
        "bool": "at::kBool",
        "i32": "at::kInt",
        "i64": "at::kLong",
        "f16": "at::kHalf",
        "bf16": "at::kBFloat16",
        "f32": "at::kFloat",
        "f64": "at::kDouble",
    }[dtype]


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def temporal_cache_dependencies(
    module: Module,
    region_name: str,
) -> tuple[TemporalCacheDependencySpec, ...]:
    """Lower memoization provenance to deterministic generated-C++ IDs."""

    transformed = synthesize_epoch_memoization(module)
    invocation = next(
        operation
        for policy in transformed.policies
        for operation in _walk_operations(policy.body)
        if operation.opcode == "vla.invoke"
        and operation.attributes.get("region") == region_name
    )
    input_ids = {
        stream.name: index
        for index, stream in enumerate(transformed.inputs)
    }
    state_ids = {
        state.name: index
        for index, state in enumerate(transformed.states)
    }
    result = []
    for dependency in invocation.attributes["memoize_dependencies"]:
        kind = str(dependency["kind"])
        subject = str(dependency["subject"])
        if kind == "epoch":
            subject_id = input_ids[subject]
        elif kind == "state_version":
            subject_id = state_ids[subject]
        else:
            raise ValueError(
                f"unsupported temporal cache dependency kind: {kind}"
            )
        result.append(
            TemporalCacheDependencySpec(
                kind=kind,
                subject_id=subject_id,
                max_age_ns=(
                    None
                    if dependency["max_age_ns"] is None
                    else int(dependency["max_age_ns"])
                ),
            )
        )
    return tuple(result)


def certified_cache_dependencies(
    certificate: CompilationCertificate,
    region_name: str,
) -> tuple[TemporalCacheDependencySpec, ...]:
    """Read the exact cache signature approved for generated C++."""

    cache = next(
        (
            item
            for item in certificate.caches
            if item.region == region_name and item.enabled
        ),
        None,
    )
    if cache is None:
        return ()
    return tuple(
        TemporalCacheDependencySpec(
            kind=item.kind,
            subject_id=item.subject_id,
            max_age_ns=item.max_age_ns,
        )
        for item in cache.dependencies
    )


def _walk_operations(block: Block):
    for operation in block.operations:
        yield operation
        for region in operation.regions:
            yield from _walk_operations(region)


def render_cpp_cache_dependencies(
    dependencies: tuple[TemporalCacheDependencySpec, ...],
) -> str:
    lines = []
    for dependency in dependencies:
        if dependency.kind != "epoch":
            raise ValueError(
                "real-model benchmark codegen requires an explicit runtime "
                "StateVersion binding"
            )
        kind = "kInputEpoch"
        maximum = (
            "vlaforge::runtime::TemporalDependency::kUnboundedAge"
            if dependency.max_age_ns is None
            else f"{dependency.max_age_ns}u"
        )
        lines.append(
            "        "
            "{vlaforge::runtime::TemporalDependencyKind::"
            f"{kind}, {dependency.subject_id}u, 0u, "
            f"{{0u, 0u, 0u, 0u}}, {maximum}}},"
        )
    return "\n".join(lines)
