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
  const VLAForgeRegionExecutionExtensionApi *execution_extension = nullptr;
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
    const auto status =
        Error(VLAFORGE_STATUS_NOT_FOUND,
              symbol_error != nullptr
                  ? symbol_error
                  : "external Region plugin value API symbol is missing");
    dlclose(handle);
    return status;
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
  const VLAForgeRegionExecutionExtensionApi *extension = nullptr;
  (void)dlerror();
  void *extension_symbol =
      dlsym(handle, VLAFORGE_REGION_EXECUTION_EXTENSION_API_SYMBOL);
  const char *extension_error = dlerror();
  if (extension_error == nullptr && extension_symbol != nullptr) {
    VLAForgeRegionExecutionExtensionApiProviderFn extension_provider = nullptr;
    static_assert(sizeof(extension_provider) == sizeof(extension_symbol),
                  "function and data pointers must have equal size");
    std::memcpy(&extension_provider, &extension_symbol,
                sizeof(extension_provider));
    try {
      extension = extension_provider();
    } catch (...) {
      dlclose(handle);
      return Error(VLAFORGE_STATUS_BACKEND_ERROR,
                   "external Region execution extension provider threw");
    }
    const auto extension_validation =
        vlaforge_region_execution_extension_api_validate(extension);
    if (extension_validation.code != VLAFORGE_STATUS_OK) {
      dlclose(handle);
      return Error(VLAFORGE_STATUS_UNSUPPORTED_ABI,
                   "external Region execution extension ABI is invalid");
    }
  }
  auto *plugin = new (std::nothrow) VLAForgeExternalRegionPlugin();
  if (plugin == nullptr) {
    dlclose(handle);
    return Error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                 "external Region plugin allocation failed");
  }
  plugin->handle = handle;
  plugin->api = api;
  plugin->execution_extension = extension;
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

extern "C" const VLAForgeRegionExecutionExtensionApi *
vlaforge_external_region_plugin_execution_extension_api(
    const VLAForgeExternalRegionPlugin *plugin) {
  return plugin == nullptr ? nullptr : plugin->execution_extension;
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

extern "C" VLAForgeStatus vlaforge_external_region_plugin_numerical_provider(
    const VLAForgeExternalRegionPlugin* plugin,
    const VLAForgeNumericalProviderApi** output) {
  if (output == nullptr || plugin == nullptr) {
    return Error(VLAFORGE_STATUS_INVALID_ARGUMENT, "invalid numerical plugin query");
  }
  *output = nullptr;
#if defined(__unix__) || defined(__APPLE__)
  (void)dlerror();
  void* symbol = dlsym(plugin->handle, VLAFORGE_NUMERICAL_PROVIDER_API_SYMBOL);
  const char* error = dlerror();
  if (error != nullptr || symbol == nullptr) {
    return Error(VLAFORGE_STATUS_NOT_FOUND, "numerical provider sidecar is absent");
  }
  VLAForgeNumericalProviderApiFn provider = nullptr;
  static_assert(sizeof(provider) == sizeof(symbol));
  std::memcpy(&provider, &symbol, sizeof(provider));
  try {
    const auto* api = provider();
    const auto status = vlaforge_numerical_provider_api_validate(api);
    if (status.code != VLAFORGE_STATUS_OK) { return status; }
    *output = api;
    return vlaforge_status_ok();
  } catch (...) {
    return Error(VLAFORGE_STATUS_BACKEND_ERROR, "numerical provider query threw");
  }
#else
  return Error(VLAFORGE_STATUS_UNSUPPORTED_ABI, "numerical plugin requires dlopen");
#endif
}
