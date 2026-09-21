#include "vlaforge/backends/libtorch_numerical.h"

#include <ATen/Context.h>
#include <ATen/autocast_mode.h>
#include <torch/version.h>

#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace {

constexpr char kNamespace[] = "libtorch.python_context_v2/1";
constexpr char kDomain[] = "libtorch.global_and_thread_flags/1";
constexpr char kReductionApi[] = "torch-2.10.0/reduction-and-split-k";
constexpr std::array<std::string_view, 24> kNames{{
    "autocast_cache_enabled", "autocast_cpu_dtype", "autocast_cpu_enabled",
    "autocast_cuda_dtype", "autocast_cuda_enabled",
    "cuda_matmul_allow_bf16_reduced_precision_reduction",
    "cuda_matmul_allow_bf16_reduced_precision_reduction_split_k",
    "cuda_matmul_allow_fp16_reduced_precision_reduction",
    "cuda_matmul_allow_fp16_reduced_precision_reduction_split_k",
    "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark",
    "cudnn_deterministic", "cudnn_enabled", "deterministic_algorithms_enabled",
    "deterministic_algorithms_warn_only", "float32_matmul_precision", "reduction_api",
    "sdpa_cudnn_enabled", "sdpa_flash_enabled", "sdpa_math_allow_fp16_bf16_reduction",
    "sdpa_math_enabled", "sdpa_mem_efficient_enabled", "torch_release",
}};

struct Value {
  VLAForgeNumericalKind kind = VLAFORGE_NUMERICAL_BOOL;
  bool flag = false;
  std::string_view text;
  bool operator==(const Value& other) const noexcept {
    return kind == other.kind && flag == other.flag && text == other.text;
  }
};

using Policy = std::array<Value, kNames.size()>;
Value Bool(bool value) { return {VLAFORGE_NUMERICAL_BOOL, value, {}}; }
Value Text(std::string_view value) { return {VLAFORGE_NUMERICAL_STRING, false, value}; }

VLAForgeStatus Error(const char* message) noexcept {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, message);
}

std::string_view KnownText(std::size_t index, std::string_view value) {
  if (index == 1u || index == 3u) {
    for (const auto name : {"float16", "bfloat16", "float32", "float64"}) {
      if (value == name) return name;
    }
  } else if (index == 16u) {
    for (const auto name : {"highest", "high", "medium"}) {
      if (value == name) return name;
    }
  } else if (index == 17u && value == kReductionApi) {
    return kReductionApi;
  } else if (index == 23u && value == "2.10.0") {
    return "2.10.0";
  }
  throw std::invalid_argument("unsupported LibTorch numerical string value");
}

bool TextField(std::size_t index) {
  return index == 1u || index == 3u || index == 16u || index == 17u || index == 23u;
}

VLAForgeStatus Parse(const VLAForgeNumericalRequirementView* requirement, Policy* output) {
  const auto valid = vlaforge_numerical_requirement_validate(requirement);
  if (valid.code != VLAFORGE_STATUS_OK) return valid;
  if (std::string_view(requirement->policy_namespace, requirement->namespace_size) != kNamespace ||
      requirement->entry_count != kNames.size()) {
    return Error("unsupported or incomplete LibTorch numerical policy");
  }
  if (std::string_view(TORCH_VERSION) != "2.10.0") {
    return Error("LibTorch numerical provider requires release 2.10.0");
  }
  for (std::size_t index = 0; index < kNames.size(); ++index) {
    const auto& entry = requirement->entries[index];
    if (std::string_view(entry.name, entry.name_size) != kNames[index] ||
        entry.kind != (TextField(index) ? VLAFORGE_NUMERICAL_STRING : VLAFORGE_NUMERICAL_BOOL)) {
      return Error("LibTorch numerical field name or type mismatch");
    }
    (*output)[index] = TextField(index)
        ? Text(KnownText(index, std::string_view(entry.text, entry.text_size)))
        : Bool(entry.integer != 0);
  }
  if ((*output)[9].flag != ((*output)[16].text != "highest") ||
      ((*output)[5].flag && !(*output)[6].flag) ||
      ((*output)[7].flag && !(*output)[8].flag)) {
    return Error("inconsistent LibTorch numerical aliases or reduction pair");
  }
  return vlaforge_status_ok();
}

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 10 && TORCH_VERSION_PATCH == 0
std::string_view Dtype(at::ScalarType value) {
  switch (value) {
    case at::kHalf: return "float16";
    case at::kBFloat16: return "bfloat16";
    case at::kFloat: return "float32";
    case at::kDouble: return "float64";
    default: throw std::invalid_argument("unsupported observed LibTorch autocast dtype");
  }
}

std::string_view Precision(at::Float32MatmulPrecision value) {
  switch (value) {
    case at::Float32MatmulPrecision::HIGHEST: return "highest";
    case at::Float32MatmulPrecision::HIGH: return "high";
    case at::Float32MatmulPrecision::MEDIUM: return "medium";
  }
  throw std::invalid_argument("unknown observed LibTorch matmul precision");
}

std::array<bool, 2> Reduction(at::CuBLASReductionOption value) {
  switch (value) {
    case at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK: return {true, true};
    case at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK: return {false, true};
    case at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK: return {false, false};
  }
  throw std::invalid_argument("unknown observed LibTorch reduction state");
}
#endif

Policy Current() {
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 10 && TORCH_VERSION_PATCH == 0
  const auto& context = at::globalContext();
  const auto bf16 = Reduction(context.allowBF16ReductionCuBLAS());
  const auto fp16 = Reduction(context.allowFP16ReductionCuBLAS());
  return {{
      Bool(at::autocast::is_autocast_cache_enabled()),
      Text(Dtype(at::autocast::get_autocast_dtype(at::kCPU))),
      Bool(at::autocast::is_autocast_enabled(at::kCPU)),
      Text(Dtype(at::autocast::get_autocast_dtype(at::kCUDA))),
      Bool(at::autocast::is_autocast_enabled(at::kCUDA)),
      Bool(bf16[0]), Bool(bf16[1]), Bool(fp16[0]), Bool(fp16[1]),
      Bool(context.allowTF32CuBLAS()), Bool(context.allowTF32CuDNN()),
      Bool(context.benchmarkCuDNN()), Bool(context.deterministicCuDNN()),
      Bool(context.userEnabledCuDNN()), Bool(context.deterministicAlgorithms()),
      Bool(context.deterministicAlgorithmsWarnOnly()), Text(Precision(context.float32MatmulPrecision())),
      Text(kReductionApi), Bool(context.userEnabledCuDNNSDP()), Bool(context.userEnabledFlashSDP()),
      Bool(context.allowFP16BF16ReductionMathSDP()), Bool(context.userEnabledMathSDP()),
      Bool(context.userEnabledMemEfficientSDP()), Text(TORCH_VERSION),
  }};
#else
  throw std::invalid_argument("unsupported LibTorch numerical getter domain");
#endif
}

void Apply(const Policy& target) {
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 10 && TORCH_VERSION_PATCH == 0
  auto& context = at::globalContext();
  const auto before = Current();
  const auto dtype = [](std::string_view name) {
    if (name == "float16") return at::kHalf;
    if (name == "bfloat16") return at::kBFloat16;
    if (name == "float32") return at::kFloat;
    if (name == "float64") return at::kDouble;
    throw std::invalid_argument("unsupported initialization autocast dtype");
  };
  if (!(before[0] == target[0])) at::autocast::set_autocast_cache_enabled(target[0].flag);
  if (!(before[1] == target[1])) at::autocast::set_autocast_dtype(at::kCPU, dtype(target[1].text));
  if (!(before[2] == target[2])) at::autocast::set_autocast_enabled(at::kCPU, target[2].flag);
  if (!(before[3] == target[3])) at::autocast::set_autocast_dtype(at::kCUDA, dtype(target[3].text));
  if (!(before[4] == target[4])) at::autocast::set_autocast_enabled(at::kCUDA, target[4].flag);
  if (!(before[5] == target[5]) || !(before[6] == target[6])) {
    context.setAllowBF16ReductionCuBLAS(target[5].flag, target[6].flag);
  }
  if (!(before[7] == target[7]) || !(before[8] == target[8])) {
    context.setAllowFP16ReductionCuBLAS(target[7].flag, target[8].flag);
  }
  // TF32 is an alias: writing it after "medium" would silently select "high".
  if (!(before[16] == target[16])) context.setFloat32MatmulPrecision(std::string(target[16].text));
  if (!(before[10] == target[10])) context.setAllowTF32CuDNN(target[10].flag);
  if (!(before[11] == target[11])) context.setBenchmarkCuDNN(target[11].flag);
  if (!(before[12] == target[12])) context.setDeterministicCuDNN(target[12].flag);
  if (!(before[13] == target[13])) context.setUserEnabledCuDNN(target[13].flag);
  if (!(before[14] == target[14]) || !(before[15] == target[15])) {
    context.setDeterministicAlgorithms(target[14].flag, target[15].flag);
  }
  if (!(before[18] == target[18])) context.setSDPUseCuDNN(target[18].flag);
  if (!(before[19] == target[19])) context.setSDPUseFlash(target[19].flag);
  if (!(before[20] == target[20])) context.setAllowFP16BF16ReductionMathSDP(target[20].flag);
  if (!(before[21] == target[21])) context.setSDPUseMath(target[21].flag);
  if (!(before[22] == target[22])) context.setSDPUseMemEfficient(target[22].flag);
#else
  (void)target;
  throw std::invalid_argument("unsupported LibTorch numerical setter domain");
#endif
}

struct Lease {
  Policy policy;
  std::array<char, 64> digest;
};

struct Registry {
  std::mutex mutex;
  std::unordered_set<Lease*> leases;
  bool poisoned = false;
  bool acquired_before = false;
  bool initialized = false;
  Policy initial_policy;
  std::shared_ptr<const char> initial_thread;
};

const std::shared_ptr<const char>& CallingThread() {
  // Retained ownership prevents thread-exit/address reuse from authorizing a
  // different thread, unlike a reusable std::thread::id or bare TLS address.
  thread_local const auto identity = std::make_shared<const char>(0);
  return identity;
}

Registry& Process() {
  // Fatal device paths intentionally retain leases until worker exit.
  static Registry* registry = new Registry;
  return *registry;
}

VLAForgeStatus Query(const VLAForgeNumericalRequirementView* requirement) {
  try {
    Policy policy;
    return Parse(requirement, &policy);
  } catch (...) {
    return Error("unsupported LibTorch numerical requirement");
  }
}

VLAForgeStatus Acquire(const VLAForgeNumericalRequirementView* requirement, void** output) {
  if (output == nullptr) return Error("null LibTorch numerical lease output");
  *output = nullptr;
  try {
    Policy policy;
    const auto status = Parse(requirement, &policy);
    if (status.code != VLAFORGE_STATUS_OK) return status;
    auto& registry = Process();
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (registry.poisoned) return Error("LibTorch numerical worker poisoned; exit process");
    if (registry.initialized && !(registry.initial_policy == policy)) {
      return Error("LibTorch requirement conflicts with initialized worker policy");
    }
    if (!(Current() == policy)) return Error("LibTorch current numerical policy mismatch");
    for (const auto* active : registry.leases) {
      if (!(active->policy == policy) ||
          std::memcmp(active->digest.data(), requirement->policy_sha256, 64u) != 0) {
        return Error("conflicting active LibTorch numerical lease");
      }
    }
    auto lease = std::make_unique<Lease>();
    lease->policy = policy;
    std::memcpy(lease->digest.data(), requirement->policy_sha256, 64u);
    registry.leases.insert(lease.get());
    registry.acquired_before = true;
    *output = lease.release();
    return vlaforge_status_ok();
  } catch (...) {
    return Error("LibTorch numerical lease acquisition failed");
  }
}

VLAForgeStatus Validate(void* pointer, VLAForgeNumericalBoundary boundary) {
  if (pointer == nullptr || boundary < VLAFORGE_NUMERICAL_BEFORE_LOAD ||
      boundary > VLAFORGE_NUMERICAL_BEFORE_COMMIT) return Error("invalid LibTorch numerical validation");
  try {
    auto& registry = Process();
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (registry.poisoned) return Error("LibTorch numerical worker poisoned; exit process");
    auto* lease = static_cast<Lease*>(pointer);
    if (registry.leases.count(lease) == 0u || !(Current() == lease->policy)) {
      return Error("LibTorch numerical lease or current-thread policy mismatch");
    }
    return vlaforge_status_ok();
  } catch (...) {
    return Error("LibTorch numerical validation failed");
  }
}

VLAForgeStatus InitializeWorker(const VLAForgeNumericalRequirementView* requirement,
                               const VLAForgeLibTorchNumericalWorkerOptions* options) {
  constexpr uint32_t kAcknowledgements = VLAFORGE_LIBTORCH_NUMERICAL_EXCLUSIVE_PROCESS |
                                         VLAFORGE_LIBTORCH_NUMERICAL_CALLING_THREAD;
  if (options == nullptr || options->struct_size != sizeof(*options) ||
      options->abi_version != VLAFORGE_LIBTORCH_NUMERICAL_WORKER_ABI_VERSION ||
      options->acknowledgements != kAcknowledgements) {
    return Error("LibTorch worker initialization requires explicit process/thread ownership");
  }
  Policy policy;
  const auto parsed = Parse(requirement, &policy);
  if (parsed.code != VLAFORGE_STATUS_OK) return parsed;
  auto& registry = Process();
  std::lock_guard<std::mutex> lock(registry.mutex);
  if (registry.poisoned) return Error("LibTorch numerical worker poisoned; exit process");
  if (!registry.leases.empty() || registry.acquired_before) {
    return Error("LibTorch worker initialization must precede the first numerical lease");
  }
  const auto& thread = CallingThread();
  if (registry.initialized) {
    return registry.initial_policy == policy &&
                   registry.initial_thread == thread && Current() == policy
        ? vlaforge_status_ok()
        : Error("LibTorch worker already initialized; retuning or thread transfer refused");
  }
  const auto original = Current();
  try {
    Apply(policy);
    if (!(Current() == policy)) throw std::runtime_error("numerical initialization readback mismatch");
    registry.initial_policy = policy;
    registry.initial_thread = thread;
    registry.initialized = true;
    return vlaforge_status_ok();
  } catch (...) {
    try {
      Apply(original);
      if (!(Current() == original)) throw std::runtime_error("numerical rollback readback mismatch");
    } catch (...) {
      registry.poisoned = true;
      return Error("LibTorch worker initialization rollback failed; exit process");
    }
    return Error("LibTorch worker initialization failed; declared flags restored");
  }
}

VLAForgeStatus Bind(VLAForgeRegionExecutable* region, void* lease) {
  return region == nullptr ? Error("null LibTorch numerical Region")
                           : Validate(lease, VLAFORGE_NUMERICAL_BEFORE_LOAD);
}

void Release(void* pointer) {
  if (pointer == nullptr) return;
  try {
    auto& registry = Process();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto* lease = static_cast<Lease*>(pointer);
    if (registry.leases.erase(lease) != 0u) delete lease;
  } catch (...) {
    // Do not unwind across the C ABI; retain uncertain ownership until exit.
  }
}

const VLAForgeNumericalProviderApi kApi{
    sizeof(VLAForgeNumericalProviderApi), VLAFORGE_NUMERICAL_ABI_VERSION,
    kDomain, sizeof(kDomain) - 1u, &Query, &Acquire, &Validate, &Bind, &Release,
};

}  // namespace

extern "C" const VLAForgeNumericalProviderApi* vlaforge_libtorch_numerical_provider_api(void) {
  return &kApi;
}

extern "C" VLAForgeStatus vlaforge_libtorch_numerical_initialize_worker(
    const VLAForgeNumericalRequirementView* requirement,
    const VLAForgeLibTorchNumericalWorkerOptions* options) {
  try {
    return InitializeWorker(requirement, options);
  } catch (...) {
    return Error("LibTorch worker initialization rejected before applying policy");
  }
}
