#include "vlaforge/runtime/external_region_plugin.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

bool Open(const char *path, VLAForgeExternalRegionPlugin **plugin,
          VLAForgeStatusCode expected) {
  const auto status =
      vlaforge_external_region_plugin_open(path, std::strlen(path), plugin);
  return status.code == expected;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    return 1;
  }
  VLAForgeExternalRegionPlugin *first = nullptr;
  VLAForgeExternalRegionPlugin *second = nullptr;
  if (!Open(argv[1], &first, VLAFORGE_STATUS_OK) ||
      !Open(argv[1], &second, VLAFORGE_STATUS_OK) || first == nullptr ||
      second == nullptr ||
      vlaforge_external_region_plugin_api(first) == nullptr ||
      vlaforge_external_region_plugin_api(second) == nullptr) {
    return 2;
  }

  const auto *api = vlaforge_external_region_plugin_api(first);
  VLAForgeRegionExecutable *executable = nullptr;
  const VLAForgeRegionCreateOptions options{
      sizeof(VLAForgeRegionCreateOptions),
      VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
      0u,
      {VLAFORGE_DEVICE_CPU, 0}};
  if (api->create(&options, &executable).code != VLAFORGE_STATUS_OK ||
      executable == nullptr) {
    return 3;
  }

  constexpr char kGoodSchema[] =
      "0000000000000000000000000000000000000000000000000000000000000000";
  constexpr char kBadSchema[] =
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
  constexpr char kTarget[] = "cpu";
  constexpr char kWrongTarget[] = "cuda";
  constexpr char kVariant[] = "shared-plugin/1";
  constexpr char kWrongVariant[] = "shared-plugin/2";
  constexpr char kPath[] = "verified/plugin.so";
  std::array<std::uint8_t, 32> sha{};
  VLAForgeArtifactDescriptor descriptor{
      sizeof(VLAForgeArtifactDescriptor),
      VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
      kPath,
      sizeof(kPath) - 1u,
      sha.data(),
      1u,
      kBadSchema,
      sizeof(kBadSchema) - 1u,
      kTarget,
      sizeof(kTarget) - 1u,
      kVariant,
      sizeof(kVariant) - 1u};
  if (api->load(executable, &descriptor).code !=
      VLAFORGE_STATUS_FAILED_PRECONDITION) {
    return 4;
  }
  descriptor.io_schema_digest = kGoodSchema;
  descriptor.io_schema_digest_size = sizeof(kGoodSchema) - 1u;
  descriptor.backend_variant = kWrongVariant;
  descriptor.backend_variant_size = sizeof(kWrongVariant) - 1u;
  if (api->load(executable, &descriptor).code !=
      VLAFORGE_STATUS_FAILED_PRECONDITION) {
    return 5;
  }
  descriptor.backend_variant = kVariant;
  descriptor.backend_variant_size = sizeof(kVariant) - 1u;
  descriptor.target = kWrongTarget;
  descriptor.target_size = sizeof(kWrongTarget) - 1u;
  if (api->load(executable, &descriptor).code !=
      VLAFORGE_STATUS_FAILED_PRECONDITION) {
    return 6;
  }
  descriptor.target = kTarget;
  descriptor.target_size = sizeof(kTarget) - 1u;
  if (api->load(executable, &descriptor).code != VLAFORGE_STATUS_OK) {
    return 7;
  }
  VLAForgeWorkspaceRequirement workspace{};
  if (api->query_workspace(executable, &workspace).code != VLAFORGE_STATUS_OK ||
      workspace.size_bytes != 0u ||
      api->bind_workspace(executable, nullptr, 0u).code != VLAFORGE_STATUS_OK) {
    return 8;
  }
  api->destroy(executable);
  vlaforge_external_region_plugin_close(second);
  vlaforge_external_region_plugin_close(first);

  VLAForgeExternalRegionPlugin *rejected = nullptr;
  if (!Open(argv[2], &rejected, VLAFORGE_STATUS_UNSUPPORTED_ABI) ||
      rejected != nullptr ||
      !Open(argv[3], &rejected, VLAFORGE_STATUS_NOT_FOUND) ||
      rejected != nullptr) {
    return 9;
  }
  return 0;
}
