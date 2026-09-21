#include "vlaforge/runtime/execution_context.h"
#include "cuda_runtime_api.h"
#include <atomic>
#include <cassert>
#include <cstring>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

struct FakeStream { int ordinal; int serial; int failure = 0; bool destroyed = false; };
thread_local int current_device = 0;
std::atomic<int> created{0}, synchronized{0}, destroyed{0};
bool fail_device = false, fail_create = false;
cudaError_t cudaGetDevice(int* result) { *result = current_device; return cudaSuccess; }
cudaError_t cudaSetDevice(int ordinal) { if (fail_device) return cudaErrorUnknown; current_device = ordinal; return cudaSuccess; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* result, unsigned flags) {
  assert(flags == cudaStreamNonBlocking);
  if (fail_create) return cudaErrorUnknown;
  *result = new FakeStream{current_device, ++created};
  return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  ++synchronized;
  assert(stream && !stream->destroyed && stream->ordinal == current_device);
  return stream->failure;
}
cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  ++destroyed;
  stream->destroyed = true;
  return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, int, cudaStream_t stream) {
  assert(stream && !stream->destroyed && stream->ordinal == current_device);
  std::memcpy(dst, src, count);
  return cudaSuccess;
}
const char* cudaGetErrorString(cudaError_t) { return "injected CUDA failure"; }

struct Lease {
  VLAForgeExecutionContext* owner = nullptr;
  cudaStream_t stream = nullptr;
  explicit Lease(int ordinal) {
    const VLAForgeExecutionContextOptions options{sizeof(options), VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION, {VLAFORGE_DEVICE_CUDA, ordinal}};
    assert(vlaforge_execution_context_create(&options, &owner).code == VLAFORGE_STATUS_OK);
    VLAForgeExecutionContextView view{};
    view.struct_size = sizeof(view);
    assert(vlaforge_execution_context_get_view(owner, &view).code == VLAFORGE_STATUS_OK);
    assert(view.device.ordinal == ordinal);
    stream = static_cast<cudaStream_t>(view.native_stream);
  }
  void release() { vlaforge_execution_context_destroy(owner); owner = nullptr; }
  ~Lease() { if (owner) release(); }
};

struct LateOwner {
  Lease* lease = nullptr;
  ~LateOwner() {
    if (lease != nullptr) {
      auto* original = lease->stream;
      delete lease;
      Lease final_lease(0);
      assert(final_lease.stream == original);
      std::cout << "late-global-release: passed\n";
    }
  }
};
LateOwner late_owner;

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string mode(argv[1]);
  if (mode == "late-global-release") {
    late_owner.lease = new Lease(0);
  } else if (mode == "reuse") {
    Lease first(0);
    auto* original = first.stream;
    first.release();
    for (int index = 0; index < 16; ++index) {
      Lease next(0);
      assert(next.stream == original && !original->destroyed);
    }
    assert(created == 1 && synchronized == 17 && destroyed == 0);
  } else if (mode == "live-isolation") {
    Lease first(0), second(0);
    assert(first.stream != second.stream);
    auto* reusable = first.stream;
    first.release();
    Lease third(0);
    assert(third.stream == reusable && third.stream != second.stream);
  } else if (mode == "device-isolation") {
    current_device = 3;
    Lease first(0), second(1);
    assert(current_device == 3 && first.stream != second.stream);
    auto* zero = first.stream;
    auto* one = second.stream;
    first.release(); second.release();
    assert(current_device == 3);
    Lease next_one(1), next_zero(0);
    assert(next_one.stream == one && next_zero.stream == zero && current_device == 3);
  } else if (mode == "poison" || mode == "drain-failure" || mode == "capture-active" || mode == "device-failure") {
    Lease first(1);
    auto* unsafe = first.stream;
    if (mode == "poison") vlaforge_execution_context_poison(first.owner);
    if (mode == "drain-failure") unsafe->failure = cudaErrorUnknown;
    if (mode == "capture-active") unsafe->failure = cudaErrorStreamCaptureUnsupported;
    if (mode == "device-failure") fail_device = true;
    first.release();
    fail_device = false;
    assert(!unsafe->destroyed && destroyed == 0);
    Lease replacement(1);
    assert(replacement.stream != unsafe);
  } else if (mode == "sticky-drain-failure" || mode == "sticky-device-failure") {
    Lease first(1);
    auto* unsafe = first.stream;
    if (mode == "sticky-drain-failure") unsafe->failure = cudaErrorUnknown;
    else fail_device = true;
    assert(vlaforge_execution_context_synchronize(first.owner).code == VLAFORGE_STATUS_BACKEND_ERROR);
    unsafe->failure = cudaSuccess;
    fail_device = false;
    const int calls = synchronized;
    assert(vlaforge_execution_context_status(first.owner).code == VLAFORGE_STATUS_FAILED_PRECONDITION);
    assert(vlaforge_execution_context_synchronize(first.owner).code == VLAFORGE_STATUS_FAILED_PRECONDITION);
    VLAForgeExecutionContextView view{};
    view.struct_size = sizeof(view);
    assert(vlaforge_execution_context_get_view(first.owner, &view).code == VLAFORGE_STATUS_FAILED_PRECONDITION);
    assert(vlaforge_execution_context_copy(first.owner, nullptr, nullptr, 0).code == VLAFORGE_STATUS_FAILED_PRECONDITION);
    first.release();
    assert(synchronized == calls && destroyed == 0);
    Lease replacement(1);
    assert(replacement.stream != unsafe);
  } else if (mode == "invalid-input-does-not-poison") {
    Lease valid(0);
    assert(vlaforge_execution_context_synchronize(nullptr).code == VLAFORGE_STATUS_INVALID_ARGUMENT);
    assert(vlaforge_execution_context_get_view(valid.owner, nullptr).code == VLAFORGE_STATUS_INVALID_ARGUMENT);
    assert(vlaforge_execution_context_copy(valid.owner, nullptr, nullptr, 0).code == VLAFORGE_STATUS_INVALID_ARGUMENT);
    assert(vlaforge_execution_context_status(valid.owner).code == VLAFORGE_STATUS_OK);
    assert(vlaforge_execution_context_synchronize(valid.owner).code == VLAFORGE_STATUS_OK);
  } else if (mode == "creation-failure") {
    fail_create = true;
    const VLAForgeExecutionContextOptions options{sizeof(options), VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION, {VLAFORGE_DEVICE_CUDA, 0}};
    auto* owner = reinterpret_cast<VLAForgeExecutionContext*>(1);
    assert(vlaforge_execution_context_create(&options, &owner).code == VLAFORGE_STATUS_BACKEND_ERROR && owner == nullptr);
    fail_create = false;
    Lease valid(0);
  } else if (mode == "concurrent-leases") {
    std::mutex mutex;
    std::set<cudaStream_t> live;
    std::atomic<int> entered{0};
    std::vector<std::thread> workers;
    for (int index = 0; index < 8; ++index) workers.emplace_back([&]() {
      for (int round = 0; round < 25; ++round) {
        Lease lease(0);
        {
          std::lock_guard<std::mutex> lock(mutex);
          assert(live.insert(lease.stream).second);
        }
        if (round == 0) {
          ++entered;
          while (entered < 8) std::this_thread::yield();
        }
        {
          std::lock_guard<std::mutex> lock(mutex);
          assert(live.erase(lease.stream) == 1);
        }
      }
    });
    for (auto& thread : workers) thread.join();
    assert(live.empty() && created == 8 && destroyed == 0 && synchronized == 200);
  } else return 2;
  std::cout << mode << ": passed\n";
}
