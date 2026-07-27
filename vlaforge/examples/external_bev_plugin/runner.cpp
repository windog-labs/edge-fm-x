#include "session_generated.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

VLAForgeBoundTensor Tensor(void *data, std::uint64_t bytes,
                           const std::int64_t *shape, std::uint32_t rank,
                           VLAForgeDType dtype) {
  return VLAForgeBoundTensor{
      sizeof(VLAForgeBoundTensor),
      {data, bytes, shape, rank, dtype, {VLAFORGE_DEVICE_CPU, 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS,
      1u};
}

VLAForgeScalarValue I32(std::int32_t value) {
  VLAForgeScalarValue scalar{};
  scalar.struct_size = sizeof(VLAForgeScalarValue);
  scalar.dtype = VLAFORGE_DTYPE_I32;
  scalar.value.i32 = value;
  return scalar;
}

VLAForgeInputStamp Stamp(std::uint64_t revision) {
  VLAForgeInputStamp stamp{};
  stamp.struct_size = sizeof(VLAForgeInputStamp);
  stamp.has_revision = 1u;
  stamp.revision = revision;
  return stamp;
}

struct TraceCounts {
  std::uint32_t hits = 0u;
  std::uint32_t misses = 0u;
};

void CountTrace(void *context, const vlaforge::runtime::TraceEvent *event) {
  auto *counts = static_cast<TraceCounts *>(context);
  if (event->kind == vlaforge::runtime::TraceKind::kCacheHit) {
    ++counts->hits;
  } else if (event->kind == vlaforge::runtime::TraceKind::kCacheMiss) {
    ++counts->misses;
  }
}

struct FixtureValues {
  const std::int64_t bev_shape[2] = {4, 4};
  const std::int64_t agent_shape[2] = {6, 3};
  const std::int64_t route_shape[1] = {3};
  float bev[16]{};
  float agents[18]{0.5f, 0.1f, 0.0f, 1.5f, -0.2f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 0.0f, 0.0f};
  float route[3]{1.0f, 0.3f, 0.0f};

  FixtureValues() {
    for (std::size_t row = 0; row < 4u; ++row) {
      for (std::size_t column = 0; column < 4u; ++column) {
        bev[row * 4u + column] = static_cast<float>(row + column) * 0.1f;
      }
    }
  }

  vlaforge_generated::ModelInputs Bind(std::uint64_t revision,
                                       bool with_agents) {
    const auto stamp = Stamp(revision);
    vlaforge_generated::ModelInputs inputs{};
    inputs.external_bev =
        Tensor(bev, sizeof(bev), bev_shape, 2u, VLAFORGE_DTYPE_F32);
    inputs.external_bev_stamp = stamp;
    inputs.route_command =
        Tensor(route, sizeof(route), route_shape, 1u, VLAFORGE_DTYPE_F32);
    inputs.route_command_stamp = stamp;
    if (with_agents) {
      inputs.has_agent_features = true;
      inputs.agent_features =
          Tensor(agents, sizeof(agents), agent_shape, 2u, VLAFORGE_DTYPE_F32);
      inputs.agent_features_stamp = stamp;
      inputs.has_agent_valid_count = true;
      inputs.agent_valid_count = I32(2);
      inputs.agent_valid_count_stamp = stamp;
    }
    return inputs;
  }
};

bool CopyTrajectory(const vlaforge_generated::ModelOutputs &outputs,
                    std::array<float, 12> *trajectory) {
  if (outputs.trajectory.tensor.data == nullptr ||
      outputs.trajectory.tensor.size_bytes !=
          trajectory->size() * sizeof(float)) {
    return false;
  }
  const auto *data = static_cast<const float *>(outputs.trajectory.tensor.data);
  std::copy(data, data + trajectory->size(), trajectory->begin());
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    return 1;
  }
  vlaforge_generated::ModelSession typed(argv[1]);
  if (!typed.initialization_status().ok()) {
    return 2;
  }
  TraceCounts trace{};
  typed.SetTraceSink({&trace, &CountTrace});
  FixtureValues values;
  vlaforge_generated::ModelOutputs outputs{};
  if (!typed.Run(values.Bind(40u, false), &outputs).ok() ||
      !typed.Run(values.Bind(40u, false), &outputs).ok() ||
      !typed.Run(values.Bind(41u, true), &outputs).ok()) {
    return 3;
  }
  std::array<float, 12> committed{};
  if (!CopyTrajectory(outputs, &committed)) {
    return 4;
  }

  values.bev[0] = -999.0f;
  const auto failed = typed.Run(values.Bind(42u, true), &outputs);
  VLAForgeBoundTensor preserved{};
  if (failed.code != vlaforge::runtime::StatusCode::kInternal ||
      !typed.ReadOutputTensor(0u, &preserved).ok() ||
      !std::equal(committed.begin(), committed.end(),
                  static_cast<const float *>(preserved.tensor.data))) {
    return 5;
  }
  values.bev[0] = 0.0f;
  if (!typed.Run(values.Bind(42u, true), &outputs).ok()) {
    return 6;
  }
  std::array<float, 12> recovered{};
  if (!CopyTrajectory(outputs, &recovered) || recovered != committed) {
    return 7;
  }
  if (!typed.ResetEpisode(9u).ok() ||
      typed.ReadOutputTensor(0u, &preserved).code !=
          vlaforge::runtime::StatusCode::kNotFound ||
      !typed.Run(values.Bind(42u, true), &outputs).ok()) {
    return 8;
  }

  VLAForgeSession *generic = nullptr;
  if (vlaforge_model_session_create_from_bundle(argv[1], std::strlen(argv[1]),
                                                &generic)
          .code != VLAFORGE_STATUS_OK) {
    return 9;
  }
  const auto *api = vlaforge_model_session_api();
  std::array<char, VLAFORGE_SCHEMA_DIGEST_HEX_SIZE> wrong_schema{};
  wrong_schema.fill('f');
  if (vlaforge_session_api_validate(api, wrong_schema.data(),
                                    wrong_schema.size())
          .code != VLAFORGE_STATUS_FAILED_PRECONDITION) {
    return 10;
  }
  const auto stamp = Stamp(50u);
  auto bev = Tensor(values.bev, sizeof(values.bev), values.bev_shape, 2u,
                    VLAFORGE_DTYPE_F32);
  auto agents = Tensor(values.agents, sizeof(values.agents), values.agent_shape,
                       2u, VLAFORGE_DTYPE_F32);
  auto route = Tensor(values.route, sizeof(values.route), values.route_shape,
                      1u, VLAFORGE_DTYPE_F32);
  auto count = I32(2);
  if (api->bind_tensor(generic, 0u, &bev, &stamp).code != VLAFORGE_STATUS_OK ||
      api->bind_tensor(generic, 1u, &agents, &stamp).code !=
          VLAFORGE_STATUS_OK ||
      api->bind_scalar(generic, 2u, &count, &stamp).code !=
          VLAFORGE_STATUS_OK ||
      api->bind_tensor(generic, 3u, &route, &stamp).code !=
          VLAFORGE_STATUS_OK ||
      api->run(generic).code != VLAFORGE_STATUS_OK) {
    return 11;
  }
  VLAForgeBoundTensor generic_trajectory{};
  VLAForgeScalarValue generic_token{};
  if (api->read_output_tensor(generic, 0u, &generic_trajectory).code !=
          VLAFORGE_STATUS_OK ||
      api->read_output_scalar(generic, 2u, &generic_token).code !=
          VLAFORGE_STATUS_OK ||
      !std::equal(recovered.begin(), recovered.end(),
                  static_cast<const float *>(generic_trajectory.tensor.data)) ||
      generic_token.value.i64 != outputs.vqa_token.value.i64 ||
      api->run(generic).code != VLAFORGE_STATUS_FAILED_PRECONDITION) {
    return 12;
  }
  api->destroy(generic);

  if (trace.hits != 1u || trace.misses != 5u) {
    return 13;
  }
  const auto *trajectory =
      static_cast<const float *>(outputs.trajectory.tensor.data);
  const auto *prediction =
      static_cast<const float *>(outputs.agent_prediction.tensor.data);
  std::printf(
      "EXTERNAL_PLUGIN,%.9g,%.9g,%.9g,%.9g,%lld\n",
      static_cast<double>(trajectory[0]), static_cast<double>(trajectory[1]),
      static_cast<double>(prediction[0]), static_cast<double>(prediction[1]),
      static_cast<long long>(outputs.vqa_token.value.i64));
  std::printf("PLUGIN_CACHE,%u,%u\n", trace.hits, trace.misses);
  std::printf("PLUGIN_FAILURE_RETRY,1\n");
  std::printf("PLUGIN_TYPED_GENERIC,1\n");
  return 0;
}
