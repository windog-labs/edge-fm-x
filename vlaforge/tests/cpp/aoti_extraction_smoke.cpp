#include "../../backends/aoti_extracted_package.h"
#include "vlaforge/backends/aoti_region_executable.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void Ok(VLAForgeStatus status) {
  if (status.code != VLAFORGE_STATUS_OK) {
    throw std::runtime_error(std::string(status.message, status.message_size));
  }
}

int main(int argc, char** argv) {
  if (argc < 5) return 2;
  const std::string mode = argv[1], package = argv[2], root = argv[3], sha = argv[4];
  try {
    if (mode == "extract") {
      std::string first, second;
      {
        vlaforge::backends::AotiExtractedPackage a(package, root, sha, fs::file_size(package));
        vlaforge::backends::AotiExtractedPackage b(package, root, sha, fs::file_size(package));
        first = a.directory(); second = b.directory();
        Check(first != second && fs::is_directory(first) && fs::is_directory(second), "nonexclusive extraction");
        for (const auto& entry : fs::recursive_directory_iterator(first)) {
          if (entry.is_regular_file()) {
            std::cout << "MEMBER " << fs::relative(entry.path(), first).generic_string()
                      << " " << entry.file_size() << "\n";
          }
        }
        fs::create_symlink(fs::path(root) / "foreign-sentinel", fs::path(first) / "cleanup-link");
      }
      Check(!fs::exists(first) && !fs::exists(second), "owned extraction not cleaned");
    } else {
      const auto* api = vlaforge_aoti_region_executable_api();
      VLAForgeRegionCreateOptions options{sizeof(options), 1u, 0u, {VLAFORGE_DEVICE_CPU, 0}};
      VLAForgeRegionExecutable* raw = nullptr;
      Ok(api->create(&options, &raw));
      struct Owner {
        const VLAForgeRegionExecutableApi* api;
        VLAForgeRegionExecutable* value;
        ~Owner() { api->destroy(value); }
      } owner{api, raw};
      if (mode != "default" && mode != "materialized") {
        Ok(vlaforge_aoti_set_package_extraction_root(raw, root.data(), root.size()));
      }
      std::array<std::uint8_t, 32> digest{};
      Check(sha.size() == 64u, "invalid test SHA");
      for (std::size_t i = 0; i < 32u; ++i) digest[i] = std::stoul(sha.substr(i * 2u, 2u), nullptr, 16);
      VLAForgeArtifactDescriptor artifact{};
      artifact.struct_size = sizeof(artifact); artifact.callable_abi_version = 1u;
      artifact.path = package.data(); artifact.path_size = package.size();
      artifact.sha256 = mode == "nohash" ? nullptr : digest.data();
      artifact.size_bytes = fs::file_size(package);
      artifact.target = "cpu"; artifact.target_size = 3u;
      if (mode == "sequence") {
        artifact.backend_variant = "aoti-sequence/1";
        artifact.backend_variant_size = 15u;
      }
      Ok(api->load(raw, &artifact));
      Check(vlaforge_aoti_set_package_extraction_root(raw, root.data(), root.size()).code ==
                VLAFORGE_STATUS_FAILED_PRECONDITION, "late root override accepted");
      Check(argc == 6, "missing expected output");
      std::ifstream reference(argv[5], std::ios::binary);
      std::vector<float> expected(4u), output(4u, -100.0f);
      reference.read(reinterpret_cast<char*>(expected.data()), 16u);
      Check(reference.gcount() == 16 && reference.peek() == EOF, "wrong reference size");
      std::array<float, 16> input{};
      for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i);
      const std::array<std::int64_t, 2> input_shape{4, 4};
      const std::array<std::int64_t, 1> output_shape{4};
      VLAForgeTensorView in{input.data(), 64u, input_shape.data(), 2u, VLAFORGE_DTYPE_F32, {VLAFORGE_DEVICE_CPU, 0}};
      VLAForgeTensorView out{output.data(), 16u, output_shape.data(), 1u, VLAFORGE_DTYPE_F32, {VLAFORGE_DEVICE_CPU, 0}};
      Ok(api->bind_input(raw, 0u, &in)); Ok(api->bind_output(raw, 0u, &out));
      for (int i = 0; i < 3; ++i) {
        Ok(api->run(raw)); Ok(api->synchronize(raw));
        Check(std::memcmp(output.data(), expected.data(), 16u) == 0, "real CPU AOTI output mismatch");
      }
    }
    std::cout << "EXTRACTION passed " << mode << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 42;
  }
}
