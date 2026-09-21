#include <ATen/Context.h>

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>

namespace {
int calls = 0;
}

extern "C" int vlaforge_test_numerical_fault_count() { return calls; }

// LD_PRELOAD-only test interposition. Production provider has no fault hook.
void at::Context::setBenchmarkCuDNN(bool value) {
  using Setter = void(*)(at::Context*, bool);
  static const auto next = reinterpret_cast<Setter>(
      dlsym(RTLD_NEXT, "_ZN2at7Context17setBenchmarkCuDNNEb"));
  if (next == nullptr) std::abort();
  const char* mode = std::getenv("VLAFORGE_TEST_NUMERICAL_SETTER_FAULT");
  if (mode != nullptr) {
    ++calls;
    if (std::strcmp(mode, "poison") == 0 && calls == 2) {
      throw std::runtime_error("test injected rollback setter failure before mutation");
    }
  }
  next(this, value);
  if (mode != nullptr && calls == 1) {
    throw std::runtime_error("test injected forward setter failure after mutation");
  }
}
