#ifndef VLAFORGE_RUNTIME_STATUS_H_
#define VLAFORGE_RUNTIME_STATUS_H_

#include <cstdint>

namespace vlaforge::runtime {

enum class StatusCode : std::uint32_t {
  kOk = 0,
  kInvalidArgument = 1,
  kOutOfRange = 2,
  kNotFound = 3,
  kAlreadyExists = 4,
  kFailedPrecondition = 5,
  kResourceExhausted = 6,
  kValidationFailed = 7,
  kInternal = 8,
};

struct Status final {
  StatusCode code = StatusCode::kOk;
  std::uint32_t subject_id = 0;
  const char* message = "ok";

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == StatusCode::kOk;
  }

  [[nodiscard]] static constexpr Status Ok() noexcept { return {}; }

  [[nodiscard]] static constexpr Status Error(StatusCode error_code,
                                              std::uint32_t subject,
                                              const char* text) noexcept {
    return Status{error_code, subject, text};
  }
};

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_STATUS_H_
