#include "vlaforge/runtime/external_region_plugin.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace {

constexpr std::size_t kErrorCapacity = 512u;
thread_local std::array<char, kErrorCapacity> g_error{};

VLAForgeStatus Error(VLAForgeStatusCode code, const char *message) noexcept {
  const char *text =
      message == nullptr ? "external Region plugin error" : message;
  std::snprintf(g_error.data(), g_error.size(), "%s", text);
  return vlaforge_status_error(code, g_error.data());
}

} // namespace

struct VLAForgeExternalRegionPlugin {
  void *handle = nullptr;
  const VLAForgeRegionExecutableValueApi *api = nullptr;
};

extern "C" VLAForgeStatus
vlaforge_external_region_plugin_open(const char *path, std::size_t path_size,
                                     VLAForgeExternalRegionPlugin **output) {
  if (path == nullptr || path_size == 0u || output == nullptr) {
    return Error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                 "external Region plugin path/output is invalid");
  }
  *output = nullptr;
#if defined(__unix__) || defined(__APPLE__)
  if (path_size == std::numeric_limits<std::size_t>::max()) {
    return Error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                 "external Region plugin path is too long");
  }
  std::unique_ptr<char[]> owned_path(new (std::nothrow) char[path_size + 1u]);
  if (owned_path == nullptr) {
    return Error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                 "external Region plugin path allocation failed");
  }
  std::memcpy(owned_path.get(), path, path_size);
  owned_path[path_size] = '\0';
  void *handle = dlopen(owned_path.get(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    return Error(VLAFORGE_STATUS_BACKEND_ERROR, dlerror());
  }
  (void)dlerror();
  void *symbol = dlsym(handle, VLAFORGE_REGION_EXECUTABLE_VALUE_API_SYMBOL);
  const char *symbol_error = dlerror();
  if (symbol_error != nullptr || symbol == nullptr) {
    dlclose(handle);
    return Error(VLAFORGE_STATUS_NOT_FOUND,
                 symbol_error != nullptr
                     ? symbol_error
                     : "external Region plugin value API symbol is missing");
  }
  VLAForgeRegionExecutableValueApiProviderFn provider = nullptr;
  static_assert(sizeof(provider) == sizeof(symbol),
                "function and data pointers must have equal size");
  std::memcpy(&provider, &symbol, sizeof(provider));
  const VLAForgeRegionExecutableValueApi *api = nullptr;
  try {
    api = provider();
  } catch (...) {
    dlclose(handle);
    return Error(VLAFORGE_STATUS_BACKEND_ERROR,
                 "external Region plugin API provider threw an exception");
  }
  const auto validation = vlaforge_region_executable_value_api_validate(api);
  if (validation.code != VLAFORGE_STATUS_OK) {
    dlclose(handle);
    return Error(VLAFORGE_STATUS_UNSUPPORTED_ABI,
                 "external Region plugin value ABI is invalid");
  }
  auto *plugin = new (std::nothrow) VLAForgeExternalRegionPlugin();
  if (plugin == nullptr) {
    dlclose(handle);
    return Error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                 "external Region plugin allocation failed");
  }
  plugin->handle = handle;
  plugin->api = api;
  *output = plugin;
  return vlaforge_status_ok();
#else
  (void)path;
  (void)path_size;
  return Error(VLAFORGE_STATUS_UNSUPPORTED_ABI,
               "external Region plugins require dlopen support");
#endif
}

extern "C" const VLAForgeRegionExecutableValueApi *
vlaforge_external_region_plugin_api(
    const VLAForgeExternalRegionPlugin *plugin) {
  return plugin == nullptr ? nullptr : plugin->api;
}

extern "C" void
vlaforge_external_region_plugin_close(VLAForgeExternalRegionPlugin *plugin) {
  if (plugin == nullptr) {
    return;
  }
#if defined(__unix__) || defined(__APPLE__)
  if (plugin->handle != nullptr) {
    dlclose(plugin->handle);
  }
#endif
  delete plugin;
}
