#include "vlaforge/backends/libtorch_numerical.h"

#include <ATen/Context.h>
#include <ATen/autocast_mode.h>
#include <torch/version.h>

#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>
#include <unistd.h>

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 10 && TORCH_VERSION_PATCH == 0
namespace {
using Scalar = std::variant<bool, std::string>;
using State = std::map<std::string, Scalar>;
const auto* api = vlaforge_libtorch_numerical_provider_api();
std::size_t checks = 0;

void Check(bool condition, const std::string& label) {
  ++checks;
  if (!condition) throw std::runtime_error(label);
}
bool Ok(VLAForgeStatus status) { return status.code == VLAFORGE_STATUS_OK; }
bool Flag(const State& state, const char* name) { return std::get<bool>(state.at(name)); }
const std::string& Text(const State& state, const char* name) {
  return std::get<std::string>(state.at(name));
}
std::string Dtype(at::ScalarType value) {
  switch (value) {
    case at::kHalf: return "float16";
    case at::kBFloat16: return "bfloat16";
    case at::kFloat: return "float32";
    case at::kDouble: return "float64";
    default: throw std::runtime_error("unexpected autocast dtype");
  }
}
at::ScalarType Dtype(const std::string& value) {
  if (value == "float16") return at::kHalf;
  if (value == "bfloat16") return at::kBFloat16;
  if (value == "float32") return at::kFloat;
  if (value == "float64") return at::kDouble;
  throw std::runtime_error("unexpected autocast dtype string");
}

State Current() {
  const auto& c = at::globalContext();
  State state;
  state["autocast_cache_enabled"] = at::autocast::is_autocast_cache_enabled();
  state["autocast_cpu_enabled"] = at::autocast::is_autocast_enabled(at::kCPU);
  state["autocast_cuda_enabled"] = at::autocast::is_autocast_enabled(at::kCUDA);
  state["autocast_cpu_dtype"] = Dtype(at::autocast::get_autocast_dtype(at::kCPU));
  state["autocast_cuda_dtype"] = Dtype(at::autocast::get_autocast_dtype(at::kCUDA));
  const auto fp16 = c.allowFP16ReductionCuBLAS();
  const auto bf16 = c.allowBF16ReductionCuBLAS();
  state["cuda_matmul_allow_fp16_reduced_precision_reduction"] =
      fp16 == at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK;
  state["cuda_matmul_allow_fp16_reduced_precision_reduction_split_k"] =
      fp16 != at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK;
  state["cuda_matmul_allow_bf16_reduced_precision_reduction"] =
      bf16 == at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK;
  state["cuda_matmul_allow_bf16_reduced_precision_reduction_split_k"] =
      bf16 != at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK;
  state["cuda_matmul_allow_tf32"] = c.allowTF32CuBLAS();
  state["cudnn_allow_tf32"] = c.allowTF32CuDNN();
  state["cudnn_benchmark"] = c.benchmarkCuDNN();
  state["cudnn_deterministic"] = c.deterministicCuDNN();
  state["cudnn_enabled"] = c.userEnabledCuDNN();
  state["deterministic_algorithms_enabled"] = c.deterministicAlgorithms();
  state["deterministic_algorithms_warn_only"] = c.deterministicAlgorithmsWarnOnly();
  const auto precision = c.float32MatmulPrecision();
  state["float32_matmul_precision"] = std::string(
      precision == at::Float32MatmulPrecision::HIGHEST ? "highest" :
      precision == at::Float32MatmulPrecision::HIGH ? "high" : "medium");
  state["sdpa_cudnn_enabled"] = c.userEnabledCuDNNSDP();
  state["sdpa_flash_enabled"] = c.userEnabledFlashSDP();
  state["sdpa_math_allow_fp16_bf16_reduction"] = c.allowFP16BF16ReductionMathSDP();
  state["sdpa_math_enabled"] = c.userEnabledMathSDP();
  state["sdpa_mem_efficient_enabled"] = c.userEnabledMemEfficientSDP();
  state["torch_release"] = std::string(TORCH_VERSION);
  state["reduction_api"] = std::string("torch-2.10.0/reduction-and-split-k");
  return state;
}

// Setters exist only in this test harness to inject and restore external drift.
void Set(const State& state, const std::string& name) {
  auto& c = at::globalContext();
  const auto flag = [&]() { return std::get<bool>(state.at(name)); };
  if (name == "autocast_cache_enabled") at::autocast::set_autocast_cache_enabled(flag());
  else if (name == "autocast_cpu_enabled") at::autocast::set_autocast_enabled(at::kCPU, flag());
  else if (name == "autocast_cuda_enabled") at::autocast::set_autocast_enabled(at::kCUDA, flag());
  else if (name == "autocast_cpu_dtype") at::autocast::set_autocast_dtype(at::kCPU, Dtype(Text(state, name.c_str())));
  else if (name == "autocast_cuda_dtype") at::autocast::set_autocast_dtype(at::kCUDA, Dtype(Text(state, name.c_str())));
  else if (name.find("cuda_matmul_allow_fp16_reduced") == 0) c.setAllowFP16ReductionCuBLAS(
      Flag(state, "cuda_matmul_allow_fp16_reduced_precision_reduction"),
      Flag(state, "cuda_matmul_allow_fp16_reduced_precision_reduction_split_k"));
  else if (name.find("cuda_matmul_allow_bf16_reduced") == 0) c.setAllowBF16ReductionCuBLAS(
      Flag(state, "cuda_matmul_allow_bf16_reduced_precision_reduction"),
      Flag(state, "cuda_matmul_allow_bf16_reduced_precision_reduction_split_k"));
  else if (name == "cuda_matmul_allow_tf32") c.setAllowTF32CuBLAS(flag());
  else if (name == "cudnn_allow_tf32") c.setAllowTF32CuDNN(flag());
  else if (name == "cudnn_benchmark") c.setBenchmarkCuDNN(flag());
  else if (name == "cudnn_deterministic") c.setDeterministicCuDNN(flag());
  else if (name == "cudnn_enabled") c.setUserEnabledCuDNN(flag());
  else if (name.find("deterministic_algorithms_") == 0) c.setDeterministicAlgorithms(
      Flag(state, "deterministic_algorithms_enabled"), Flag(state, "deterministic_algorithms_warn_only"));
  else if (name == "float32_matmul_precision") c.setFloat32MatmulPrecision(Text(state, name.c_str()));
  else if (name == "sdpa_cudnn_enabled") c.setSDPUseCuDNN(flag());
  else if (name == "sdpa_flash_enabled") c.setSDPUseFlash(flag());
  else if (name == "sdpa_math_allow_fp16_bf16_reduction") c.setAllowFP16BF16ReductionMathSDP(flag());
  else if (name == "sdpa_math_enabled") c.setSDPUseMath(flag());
  else if (name == "sdpa_mem_efficient_enabled") c.setSDPUseMemEfficient(flag());
}
void Restore(const State& state) {
  for (const auto& item : state) Set(state, item.first);
}
struct RestoreOnExit {
  State state = Current();
  ~RestoreOnExit() { Restore(state); }
};

struct Requirement {
  State values = Current();
  std::string name_space = "libtorch.python_context_v2/1";
  // Synthetic digests exercise transport, not model compilation provenance.
  std::string policy_digest = std::string(64, '1');
  std::string requirement_digest = std::string(64, '2');
  std::vector<VLAForgeNumericalEntry> entries;
  VLAForgeNumericalRequirementView View() {
    entries.clear();
    for (const auto& [name, value] : values) {
      const bool is_bool = std::holds_alternative<bool>(value);
      const auto* text = is_bool ? nullptr : &std::get<std::string>(value);
      entries.push_back({name.data(), name.size(),
          is_bool ? VLAFORGE_NUMERICAL_BOOL : VLAFORGE_NUMERICAL_STRING,
          is_bool && std::get<bool>(value) ? 1 : 0, 0.0,
          text == nullptr ? nullptr : text->data(), text == nullptr ? 0 : text->size()});
    }
    return {sizeof(VLAForgeNumericalRequirementView), VLAFORGE_NUMERICAL_ABI_VERSION,
        name_space.data(), name_space.size(), policy_digest.data(), policy_digest.size(),
        requirement_digest.data(), requirement_digest.size(), 1, entries.data(), entries.size()};
  }
};
struct Lease {
  void* pointer = nullptr;
  ~Lease() { api->release(pointer); }
};

void Reject(VLAForgeNumericalRequirementView view, const std::string& label) {
  const auto before = Current();
  Check(!Ok(api->query_support(&view)), label + ": query rejected");
  Lease lease;
  lease.pointer = reinterpret_cast<void*>(1);
  Check(!Ok(api->acquire_current(&view, &lease.pointer)) && lease.pointer == nullptr,
        label + ": acquire rejected with null output");
  Check(Current() == before, label + ": provider did not mutate state");
}

void SchemaTests() {
  Requirement good;
  auto view = good.View();
  const auto before = Current();
  Check(view.entry_count == 24, "22 flags plus two API identity fields");
  Check(Ok(api->query_support(&view)), "full observed policy supported");
  Check(Current() == before, "query has no setters");
  for (std::size_t index = 0; index < good.entries.size(); ++index) {
    Requirement wrong;
    auto bad = wrong.View();
    auto& entry = wrong.entries[index];
    entry.kind = VLAFORGE_NUMERICAL_I64;
    entry.text = nullptr;
    entry.text_size = 0;
    Reject(bad, "wrong type " + std::to_string(index));
  }
  for (const auto& [name, value] : before) {
    Requirement wrong;
    wrong.values.erase(name);
    Reject(wrong.View(), "missing " + name);
    wrong.values = before;
    wrong.values[name] = std::holds_alternative<bool>(value)
        ? Scalar(!std::get<bool>(value)) : Scalar(std::string("unsupported"));
    auto bad = wrong.View();
    Lease lease;
    Check(!Ok(api->acquire_current(&bad, &lease.pointer)), "mismatched " + name);
    Check(lease.pointer == nullptr && Current() == before, "mismatch is nonmutating " + name);
  }
  Requirement wrong;
  wrong.values["unknown_flag"] = false;
  Reject(wrong.View(), "unknown field");
  wrong = Requirement{};
  wrong.name_space = "libtorch.python_context_v1/1";
  Reject(wrong.View(), "legacy namespace");
  wrong.name_space = "libtorch.python_context_v2/2";
  Reject(wrong.View(), "unknown namespace revision");
  for (const auto* release : {"2.7.1", "2.10.0+cu128", "2.10.1", "2.11.0"}) {
    wrong = Requirement{};
    wrong.values["torch_release"] = std::string(release);
    Reject(wrong.View(), "release " + std::string(release));
  }
  for (const auto* dtype : {"float16", "bfloat16", "float32", "float64"}) {
    wrong = Requirement{};
    wrong.values["autocast_cpu_dtype"] = std::string(dtype);
    wrong.values["autocast_cuda_dtype"] = std::string(dtype);
    auto supported = wrong.View();
    Check(Ok(api->query_support(&supported)), "known dtype capability");
  }
  wrong = Requirement{};
  wrong.values["cuda_matmul_allow_tf32"] = false;
  wrong.values["float32_matmul_precision"] = std::string("high");
  Reject(wrong.View(), "contradictory matmul alias");
  for (const auto* dtype : {"fp16", "bf16"}) {
    wrong = Requirement{};
    const std::string prefix = std::string("cuda_matmul_allow_") + dtype + "_reduced_precision_reduction";
    wrong.values[prefix] = true;
    wrong.values[prefix + "_split_k"] = false;
    Reject(wrong.View(), "impossible reduction pair");
  }
  wrong = Requirement{};
  auto bad = wrong.View();
  wrong.entries.front().integer = 2;
  Reject(bad, "noncanonical bool");
  bad = wrong.View();
  wrong.entries.front().real = 0.5;
  Reject(bad, "unused typed member");
  bad = wrong.View();
  wrong.entries[1] = wrong.entries[0];
  Reject(bad, "duplicate names");
  bad = wrong.View();
  bad.abi_version = 2;
  Reject(bad, "ABI mismatch");
  bad = wrong.View();
  bad.execution_lane = 0;
  Reject(bad, "invalid lane");
  bad = wrong.View();
  bad.policy_sha256_size = 63;
  Reject(bad, "digest size");
  Check(Current() == before, "schema suite preserved state");
}

void BoundaryTests() {
  RestoreOnExit restore;
  // Both reduction booleans can now be toggled individually into valid pairs.
  at::globalContext().setAllowFP16ReductionCuBLAS(false, true);
  at::globalContext().setAllowBF16ReductionCuBLAS(false, true);
  const auto baseline = Current();
  Requirement requirement;
  auto view = requirement.View();
  Lease lease;
  Check(Ok(api->acquire_current(&view, &lease.pointer)), "acquire boundary lease");
  const std::array<VLAForgeNumericalBoundary, 4> boundaries{{
      VLAFORGE_NUMERICAL_BEFORE_LOAD, VLAFORGE_NUMERICAL_AFTER_LOAD,
      VLAFORGE_NUMERICAL_RUN_ENTRY, VLAFORGE_NUMERICAL_BEFORE_COMMIT}};
  std::size_t domains = 0;
  for (const auto& [name, value] : baseline) {
    if (name == "torch_release" || name == "reduction_api") continue;
    State changed = baseline;
    if (std::holds_alternative<bool>(value)) changed[name] = !std::get<bool>(value);
    else if (name == "float32_matmul_precision") changed[name] = std::string(
        std::get<std::string>(value) == "highest" ? "high" : "highest");
    else changed[name] = std::string(std::get<std::string>(value) == "float16" ? "bfloat16" : "float16");
    Set(changed, name);
    const auto drifted = Current();
    Check(drifted.at(name) != baseline.at(name), "external drift actually applied " + name);
    for (const auto boundary : boundaries) {
      Check(!Ok(api->validate_current(lease.pointer, boundary)), "drift detected " + name);
      Check(Current() == drifted, "validation did not restore caller " + name);
    }
    Restore(baseline);
    for (const auto boundary : boundaries) Check(Ok(api->validate_current(lease.pointer, boundary)), "restored boundary accepted");
    ++domains;
  }
  Check(domains == 22, "all declared mathematical fields exercised");
  int opaque_region = 0;
  Check(Ok(api->bind_region(reinterpret_cast<VLAForgeRegionExecutable*>(&opaque_region), lease.pointer)),
        "bind validates lease without dereferencing opaque Region");
  Check(!Ok(api->bind_region(nullptr, lease.pointer)), "null Region rejected");
  Check(!Ok(api->validate_current(nullptr, VLAFORGE_NUMERICAL_RUN_ENTRY)), "null lease rejected");
  Check(!Ok(api->validate_current(lease.pointer, static_cast<VLAForgeNumericalBoundary>(0))), "bad boundary rejected");
  Check(!Ok(api->acquire_current(&view, nullptr)), "null output rejected");
  Check(Current() == baseline, "all boundary checks left actual state unchanged");
}

void LeaseTests() {
  RestoreOnExit restore;
  Requirement requirement;
  auto view = requirement.View();
  vlaforge::runtime::NumericalLeaseSet first, second;
  Check(first.Add(0, api, &view).ok() && first.Add(1, api, &view).ok(), "first Session stages shared policy");
  Check(first.AcquireAll().ok(), "first Session acquires");
  Check(second.Add(0, api, &view).ok() && second.AcquireAll().ok(), "second Session shares policy");
  first.Clear();
  Check(second.Validate(VLAFORGE_NUMERICAL_RUN_ENTRY).ok(), "first Session release preserves second lease");
  Requirement different_digest;
  different_digest.policy_digest.assign(64, '3');
  auto digest_view = different_digest.View();
  Lease denied;
  Check(!Ok(api->acquire_current(&digest_view, &denied.pointer)), "conflicting digest denied");
  const auto before = Current();
  at::globalContext().setBenchmarkCuDNN(!Flag(before, "cudnn_benchmark"));
  Requirement changed;
  auto changed_view = changed.View();
  Check(Ok(api->query_support(&changed_view)), "new policy capability supported");
  Check(!Ok(api->acquire_current(&changed_view, &denied.pointer)), "external drift cannot bypass active process lease");
  Check(!second.Validate(VLAFORGE_NUMERICAL_BEFORE_COMMIT).ok(), "existing Session rejects drift");
  second.Clear();
  Check(Ok(api->acquire_current(&changed_view, &denied.pointer)), "last release removes process conflict");
  api->release(denied.pointer);
  Check(!Ok(api->validate_current(denied.pointer, VLAFORGE_NUMERICAL_RUN_ENTRY)), "released lease rejected");
  api->release(denied.pointer);
  denied.pointer = nullptr;
  api->release(nullptr);
  Restore(before);
  Check(Current() == before, "release and rejection have no setter side effects");
  {
    vlaforge::runtime::NumericalLeaseSet scoped;
    Check(scoped.Add(0, api, &view).ok() && scoped.AcquireAll().ok(), "RAII acquisition");
  }
  Lease after_destruction;
  Check(Ok(api->acquire_current(&digest_view, &after_destruction.pointer)), "Session destructor releases final lease");
}

void ReductionTests() {
  RestoreOnExit restore;
  const std::array<std::array<bool, 2>, 3> pairs{{{{true, true}}, {{false, true}}, {{false, false}}}};
  for (const auto& fp16 : pairs) for (const auto& bf16 : pairs) {
    at::globalContext().setAllowFP16ReductionCuBLAS(fp16[0], fp16[1]);
    at::globalContext().setAllowBF16ReductionCuBLAS(bf16[0], bf16[1]);
    const auto before = Current();
    Requirement requirement;
    auto view = requirement.View();
    Lease lease;
    Check(Ok(api->query_support(&view)) && Ok(api->acquire_current(&view, &lease.pointer)), "all nine actual reduction combinations accepted");
    Check(Ok(api->validate_current(lease.pointer, VLAFORGE_NUMERICAL_BEFORE_COMMIT)), "three-state complete pair validated");
    Check(Current() == before, "three-state pair not collapsed by provider");
  }
}

void ThreadTests() {
  const auto before = Current();
  Requirement requirement;
  auto view = requirement.View();
  Lease lease;
  Check(Ok(api->acquire_current(&view, &lease.pointer)), "main thread lease acquired");
  bool worker_ok = false;
  std::thread worker([&]() {
    RestoreOnExit restore;
    for (const auto& [name, value] : before) {
      (void)value;
      if (name.find("autocast_") == 0) Set(before, name);
    }
    const auto aligned = Current();
    const bool accepted = Ok(api->validate_current(lease.pointer, VLAFORGE_NUMERICAL_RUN_ENTRY));
    at::autocast::set_autocast_enabled(at::kCPU, !Flag(before, "autocast_cpu_enabled"));
    const auto drifted = Current();
    const bool rejected = !Ok(api->validate_current(lease.pointer, VLAFORGE_NUMERICAL_BEFORE_COMMIT));
    const bool unmodified = Current() == drifted;
    Set(aligned, "autocast_cpu_enabled");
    worker_ok = accepted && rejected && unmodified &&
        Ok(api->validate_current(lease.pointer, VLAFORGE_NUMERICAL_RUN_ENTRY));
  });
  worker.join();
  Check(worker_ok, "actual calling-thread autocast drift checked without setter");
  Check(Current() == before && Ok(api->validate_current(lease.pointer, VLAFORGE_NUMERICAL_RUN_ENTRY)),
        "worker autocast drift did not change caller thread");
}

int BootstrapThreadReuse(bool expect_old_bug) {
  const VLAForgeLibTorchNumericalWorkerOptions options{
      sizeof(options), VLAFORGE_LIBTORCH_NUMERICAL_WORKER_ABI_VERSION,
      VLAFORGE_LIBTORCH_NUMERICAL_EXCLUSIVE_PROCESS | VLAFORGE_LIBTORCH_NUMERICAL_CALLING_THREAD};
  State policy;
  std::thread::id original_id;
  bool first_ok = false;
  std::thread original([&] {
    policy = Current();
    Requirement request;
    auto view = request.View();
    original_id = std::this_thread::get_id();
    first_ok = Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &options));
  });
  original.join();
  Check(first_ok, "first thread bootstrap accepted");
  bool id_reused = false, second_ok = false;
  for (int attempt = 0; attempt < 16 && !id_reused; ++attempt) {
    std::thread replacement([&] {
      if (std::this_thread::get_id() != original_id) return;
      id_reused = true;
      Requirement request;
      request.values = policy;
      auto view = request.View();
      second_ok = Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &options));
    });
    replacement.join();
  }
  if (!id_reused) {
    std::cerr << "inconclusive: thread ID was not reused in sixteen attempts\n";
    return 77;
  }
  Check(second_ok == expect_old_bug, "thread transfer bootstrap outcome");
  std::ifstream input("/proc/self/maps");
  const std::string maps((std::istreambuf_iterator<char>(input)), {});
  Check(!maps.empty() && maps.find("libpython") == std::string::npos, "thread probe is native");
  if (const char* path = std::getenv("VLAFORGE_TEST_NUMERICAL_MAPS")) {
    std::ofstream output(path); output << maps;
    Check(output.good(), "thread probe maps saved");
  }
  std::cout << "{\"status\":\"" << (expect_old_bug ? "bug_reproduced" : "passed")
            << "\",\"pid\":" << getpid() << ",\"thread_id_reused\":true,\"replacement_bootstrap_accepted\":"
            << (second_ok ? "true" : "false") << ",\"libpython_mapped\":false}\n";
  return 0;
}

int BootstrapCases(const std::string& mode) {
  const VLAForgeLibTorchNumericalWorkerOptions options{
      sizeof(options), VLAFORGE_LIBTORCH_NUMERICAL_WORKER_ABI_VERSION,
      VLAFORGE_LIBTORCH_NUMERICAL_EXCLUSIVE_PROCESS | VLAFORGE_LIBTORCH_NUMERICAL_CALLING_THREAD};
  at::globalContext().setAllowFP16ReductionCuBLAS(false, false);
  at::globalContext().setAllowBF16ReductionCuBLAS(false, false);
  const auto before = Current();
  Requirement request;
  for (auto& [name, value] : request.values) {
    if (std::holds_alternative<bool>(value)) value = !std::get<bool>(value);
    else if (name == "autocast_cpu_dtype" || name == "autocast_cuda_dtype") {
      value = std::string(std::get<std::string>(value) == "float16" ? "bfloat16" : "float16");
    }
  }
  request.values["float32_matmul_precision"] = std::string("medium");
  request.values["cuda_matmul_allow_tf32"] = true;
  if (mode.find("pair-") == 0) {
    const int index = std::stoi(mode.substr(5));
    Check(index >= 0 && index < 9, "valid pair test index");
    const std::array<std::array<bool, 2>, 3> pairs{{{{true, true}}, {{false, true}}, {{false, false}}}};
    const auto& fp16 = pairs[index / 3];
    const auto& bf16 = pairs[index % 3];
    request.values["cuda_matmul_allow_fp16_reduced_precision_reduction"] = fp16[0];
    request.values["cuda_matmul_allow_fp16_reduced_precision_reduction_split_k"] = fp16[1];
    request.values["cuda_matmul_allow_bf16_reduced_precision_reduction"] = bf16[0];
    request.values["cuda_matmul_allow_bf16_reduced_precision_reduction_split_k"] = bf16[1];
  }
  auto view = request.View();
  if (mode == "historical-lease") {
    Requirement original;
    auto original_view = original.View();
    Lease lease;
    Check(Ok(api->acquire_current(&original_view, &lease.pointer)), "pre-bootstrap lease acquired");
    Check(!Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &options)), "active pre-bootstrap lease prevents initialization");
    api->release(lease.pointer); lease.pointer = nullptr;
    Check(!Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &options)), "historical lease prevents initialization after release");
    Check(Current() == before, "late bootstrap rejection did not mutate actual flags");
  } else if (mode == "fault-once" || mode == "fault-poison") {
    using FaultCount = int(*)();
    auto count = reinterpret_cast<FaultCount>(dlsym(RTLD_DEFAULT, "vlaforge_test_numerical_fault_count"));
    Check(count != nullptr && count() == 0, "independent ATen setter interposer loaded");
    const auto initialized = vlaforge_libtorch_numerical_initialize_worker(&view, &options);
    Check(!Ok(initialized) && count() == 2, "forward setter failed and rollback setter was called");
    if (mode == "fault-once") {
      Check(Current() == before, "rollback restored all 22 fields and both complete false/false pairs");
      Check(Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &options)), "rolled-back worker can initialize on a later successful attempt");
      Check(Current() == request.values, "retry actual policy matches full request");
    } else {
      Check(Current() != before, "rollback fault left actual observable drift");
      Check(std::string(initialized.message).find("rollback failed") != std::string::npos, "rollback failure is explicit");
      Restore(before);
      Check(Current() == before, "test caller externally restored complete state");
      Requirement original;
      auto original_view = original.View();
      Lease denied;
      const auto acquired = api->acquire_current(&original_view, &denied.pointer);
      Check(!Ok(acquired) && denied.pointer == nullptr && std::string(acquired.message).find("poisoned") != std::string::npos,
            "external restoration cannot heal poisoned worker");
      Check(!Ok(vlaforge_libtorch_numerical_initialize_worker(&original_view, &options)), "poisoned bootstrap cannot retry");
    }
  } else {
    Check(!Ok(vlaforge_libtorch_numerical_initialize_worker(&view, nullptr)), "missing acknowledgement rejected");
    for (const auto flags : {0u, 1u, 2u, 7u}) {
      auto invalid = options; invalid.acknowledgements = flags;
      Check(!Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &invalid)), "incomplete or unknown acknowledgement rejected");
    }
    auto invalid = options; invalid.struct_size -= 1;
    Check(!Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &invalid)), "options size rejected");
    invalid = options; invalid.abi_version += 1;
    Check(!Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &invalid)), "options ABI rejected");
    auto malformed = view; malformed.entry_count -= 1;
    Check(!Ok(vlaforge_libtorch_numerical_initialize_worker(&malformed, &options)), "malformed policy rejected before setters");
    Check(Current() == before, "all preflight failures preserved actual full state");
    Check(Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &options)), "explicit bootstrap succeeded");
    Check(Current() == request.values, "all 22 requested fields observed including medium and full reduction pairs");
    Check(Text(Current(), "float32_matmul_precision") == "medium", "TF32 alias did not overwrite medium");
    Check(Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &options)), "same policy/calling thread bootstrap is idempotent before leases");
    bool transfer_rejected = false;
    std::thread worker([&] {
      const auto own = Current();
      for (const auto& [name, value] : request.values) {
        (void)value;
        if (name.find("autocast_") == 0) Set(request.values, name);
      }
      const auto aligned = Current();
      transfer_rejected = aligned == request.values &&
          !Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &options)) && Current() == aligned;
      for (const auto& [name, value] : own) {
        (void)value;
        if (name.find("autocast_") == 0) Set(own, name);
      }
    });
    worker.join();
    Check(transfer_rejected, "live different thread rejected even with identical getters");
    const auto target = Current();
    at::globalContext().setBenchmarkCuDNN(!Flag(target, "cudnn_benchmark"));
    const auto drifted = Current();
    Check(!Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &options)), "idempotent call does not heal external drift");
    Check(Current() == drifted, "drifted idempotent rejection is nonmutating");
    Requirement conflicting;
    auto conflicting_view = conflicting.View();
    Check(!Ok(vlaforge_libtorch_numerical_initialize_worker(&conflicting_view, &options)), "retuning to another policy rejected");
    Lease denied;
    Check(!Ok(api->acquire_current(&conflicting_view, &denied.pointer)) && denied.pointer == nullptr,
          "external setter cannot bypass initialized worker policy");
    Restore(target);
    Lease lease;
    Check(Ok(api->acquire_current(&view, &lease.pointer)), "initialized worker acquired numerical lease");
    Check(!Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &options)), "bootstrap refused while lease active");
    api->release(lease.pointer); lease.pointer = nullptr;
    Check(!Ok(vlaforge_libtorch_numerical_initialize_worker(&view, &options)), "bootstrap refused after historical lease release");
    Check(Current() == target, "leases and rejected bootstrap preserved initialized fields");
  }
  std::ifstream input("/proc/self/maps");
  const std::string maps((std::istreambuf_iterator<char>(input)), {});
  Check(!maps.empty() && maps.find("libpython") == std::string::npos, "bootstrap test is native");
  if (const char* path = std::getenv("VLAFORGE_TEST_NUMERICAL_MAPS")) {
    std::ofstream output(path); output << maps;
    Check(output.good(), "bootstrap maps saved");
  }
  std::cout << "{\"status\":\"passed\",\"evidence_level\":\"native_cpu_worker_bootstrap\",\"pid\":"
            << getpid() << ",\"mode\":\"" << mode << "\",\"checks\":" << checks
            << ",\"libpython_mapped\":false,\"gpu_execution\":false,\"model_output_validation\":false}\n";
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--bootstrap-thread-reuse") {
      return BootstrapThreadReuse(std::string(argv[2]) == "old");
    }
    if (argc == 3 && std::string(argv[1]) == "--bootstrap-case") return BootstrapCases(argv[2]);
    Check(std::string(TORCH_VERSION) == "2.10.0", "test requires actual LibTorch 2.10.0");
    const auto initial = Current();
    Check(Ok(vlaforge_numerical_provider_api_validate(api)), "provider C ABI valid");
    SchemaTests();
    BoundaryTests();
    LeaseTests();
    ReductionTests();
    ThreadTests();
    Check(Current() == initial, "complete original caller state restored by test harness");
    std::ifstream maps_input("/proc/self/maps");
    const std::string maps((std::istreambuf_iterator<char>(maps_input)), {});
    Check(!maps.empty() && maps.find("libpython") == std::string::npos, "native process has no mapped libpython");
    Check(maps.find("libvlaforge_libtorch_numerical_backend.so") != std::string::npos, "shared process provider mapped");
    if (argc == 2) {
      std::ofstream output(argv[1]);
      output << maps;
      Check(output.good(), "process maps persisted");
    }
    std::cout << "{\"status\":\"passed\",\"evidence_level\":\"native_cpu_provider_contract\","
              << "\"model_output_validation\":false,\"gpu_execution\":false,\"torch_release\":\""
              << TORCH_VERSION << "\",\"pid\":" << getpid() << ",\"checks\":" << checks
              << ",\"mathematical_fields\":22,\"reduction_combinations\":9,\"libpython_mapped\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "check " << checks << " failed: " << error.what() << '\n';
    return 1;
  }
}
#else
// Unsupported releases must still compile legacy backends and reject this new
// policy domain. These requested values are synthetic, not an observation.
int main(int argc, char** argv) {
  const auto* api = vlaforge_libtorch_numerical_provider_api();
  std::map<std::string, std::variant<bool, std::string>> values;
  for (const auto* name : {
      "autocast_cache_enabled", "autocast_cpu_enabled", "autocast_cuda_enabled",
      "cuda_matmul_allow_bf16_reduced_precision_reduction",
      "cuda_matmul_allow_bf16_reduced_precision_reduction_split_k",
      "cuda_matmul_allow_fp16_reduced_precision_reduction",
      "cuda_matmul_allow_fp16_reduced_precision_reduction_split_k",
      "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark",
      "cudnn_deterministic", "cudnn_enabled", "deterministic_algorithms_enabled",
      "deterministic_algorithms_warn_only", "sdpa_cudnn_enabled",
      "sdpa_flash_enabled", "sdpa_math_allow_fp16_bf16_reduction",
      "sdpa_math_enabled", "sdpa_mem_efficient_enabled"}) values[name] = false;
  values["autocast_cpu_dtype"] = std::string("bfloat16");
  values["autocast_cuda_dtype"] = std::string("float16");
  values["float32_matmul_precision"] = std::string("highest");
  values["torch_release"] = std::string("2.10.0");
  values["reduction_api"] = std::string("torch-2.10.0/reduction-and-split-k");
  std::vector<VLAForgeNumericalEntry> entries;
  for (const auto& [name, value] : values) {
    const bool is_bool = std::holds_alternative<bool>(value);
    const auto* text = is_bool ? nullptr : &std::get<std::string>(value);
    entries.push_back({name.data(), name.size(),
        is_bool ? VLAFORGE_NUMERICAL_BOOL : VLAFORGE_NUMERICAL_STRING, 0, 0.0,
        text == nullptr ? nullptr : text->data(), text == nullptr ? 0 : text->size()});
  }
  const std::string namespace_name = "libtorch.python_context_v2/1", digest(64, '1');
  const VLAForgeNumericalRequirementView view{
      sizeof(view), VLAFORGE_NUMERICAL_ABI_VERSION, namespace_name.data(), namespace_name.size(),
      digest.data(), digest.size(), digest.data(), digest.size(), 1, entries.data(), entries.size()};
  if (vlaforge_numerical_requirement_validate(&view).code != VLAFORGE_STATUS_OK) return 1;
  const auto fp16 = at::globalContext().allowFP16ReductionCuBLAS();
  const auto bf16 = at::globalContext().allowBF16ReductionCuBLAS();
  const auto precision = at::globalContext().float32MatmulPrecision();
  const auto queried = api->query_support(&view);
  if (queried.code == VLAFORGE_STATUS_OK ||
      std::string(queried.message).find("requires release 2.10.0") == std::string::npos) return 2;
  void* lease = reinterpret_cast<void*>(1);
  if (api->acquire_current(&view, &lease).code == VLAFORGE_STATUS_OK || lease != nullptr) return 3;
  if (at::globalContext().allowFP16ReductionCuBLAS() != fp16 ||
      at::globalContext().allowBF16ReductionCuBLAS() != bf16 ||
      at::globalContext().float32MatmulPrecision() != precision) return 4;
  std::ifstream input("/proc/self/maps");
  const std::string maps((std::istreambuf_iterator<char>(input)), {});
  if (maps.empty() || maps.find("libpython") != std::string::npos) return 5;
  if (argc == 2) {
    std::ofstream output(argv[1]); output << maps;
    if (!output.good()) return 6;
  }
  std::cout << "{\"status\":\"passed\",\"evidence_level\":\"unsupported_libtorch_release_rejected\","
            << "\"torch_release\":\"" << TORCH_VERSION << "\",\"pid\":" << getpid()
            << ",\"model_output_validation\":false,\"gpu_execution\":false,\"libpython_mapped\":false}\n";
  return 0;
}
#endif
