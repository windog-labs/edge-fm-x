#include "vlaforge/runtime/numerical.h"

#include <cmath>
#include <cstring>
#include <new>
#include <string_view>

namespace {

bool Text(const char* data, std::size_t size) noexcept {
  return data != nullptr && size != 0u && std::memchr(data, '\0', size) == nullptr;
}

bool Digest(const char* data, std::size_t size) noexcept {
  if (data == nullptr || size != 64u) { return false; }
  for (std::size_t i = 0; i < size; ++i) {
    if (!((data[i] >= '0' && data[i] <= '9') ||
          (data[i] >= 'a' && data[i] <= 'f'))) { return false; }
  }
  return true;
}

vlaforge::runtime::Status Failure(std::uint32_t id, const char* message) noexcept {
  return vlaforge::runtime::Status::Error(
      vlaforge::runtime::StatusCode::kFailedPrecondition, id, message);
}

bool Same(const char* a, std::size_t na, const char* b, std::size_t nb) noexcept {
  return na == nb && (na == 0u || std::memcmp(a, b, na) == 0);
}

bool SameFields(const VLAForgeNumericalRequirementView& a,
                const VLAForgeNumericalRequirementView& b) noexcept {
  if (a.entry_count != b.entry_count) { return false; }
  for (std::size_t i = 0; i < a.entry_count; ++i) {
    const auto& x = a.entries[i];
    const auto& y = b.entries[i];
    if (!Same(x.name, x.name_size, y.name, y.name_size) || x.kind != y.kind ||
        x.integer != y.integer || std::memcmp(&x.real, &y.real, sizeof(x.real)) != 0 ||
        !Same(x.text, x.text_size, y.text, y.text_size)) { return false; }
  }
  return true;
}

}  // namespace

extern "C" VLAForgeStatus vlaforge_numerical_provider_api_validate(
    const VLAForgeNumericalProviderApi* api) {
  if (api == nullptr || api->struct_size < sizeof(*api) ||
      api->abi_version != VLAFORGE_NUMERICAL_ABI_VERSION ||
      !Text(api->process_domain, api->process_domain_size) ||
      api->query_support == nullptr || api->acquire_current == nullptr ||
      api->validate_current == nullptr || api->bind_region == nullptr ||
      api->release == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_UNSUPPORTED_ABI,
                                "numerical provider is absent or invalid");
  }
  return vlaforge_status_ok();
}

extern "C" VLAForgeStatus vlaforge_numerical_requirement_validate(
    const VLAForgeNumericalRequirementView* requirement) {
  if (requirement == nullptr || requirement->struct_size < sizeof(*requirement) ||
      requirement->abi_version != VLAFORGE_NUMERICAL_ABI_VERSION ||
      !Text(requirement->policy_namespace, requirement->namespace_size) ||
      !Digest(requirement->policy_sha256, requirement->policy_sha256_size) ||
      !Digest(requirement->requirement_sha256, requirement->requirement_sha256_size) ||
      (requirement->execution_lane != 1u && requirement->execution_lane != 2u) ||
      requirement->entries == nullptr || requirement->entry_count == 0u) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                "invalid numerical requirement");
  }
  std::string_view previous;
  for (std::size_t i = 0; i < requirement->entry_count; ++i) {
    const auto& entry = requirement->entries[i];
    if (!Text(entry.name, entry.name_size)) {
      return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                  "invalid numerical field name");
    }
    const std::string_view name(entry.name, entry.name_size);
    bool valid = i == 0u || previous < name;
    previous = name;
    switch (entry.kind) {
      case VLAFORGE_NUMERICAL_BOOL:
        valid &= entry.integer == 0 || entry.integer == 1;
        [[fallthrough]];
      case VLAFORGE_NUMERICAL_I64:
        valid &= entry.real == 0.0 && entry.text == nullptr && entry.text_size == 0u;
        break;
      case VLAFORGE_NUMERICAL_F64:
        valid &= std::isfinite(entry.real) && entry.integer == 0 &&
                 entry.text == nullptr && entry.text_size == 0u;
        break;
      case VLAFORGE_NUMERICAL_STRING:
        valid &= entry.integer == 0 && entry.real == 0.0 &&
                 (entry.text_size == 0u || entry.text != nullptr);
        break;
      default: valid = false;
    }
    if (!valid) {
      return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                  "invalid or duplicate typed numerical field");
    }
  }
  return vlaforge_status_ok();
}

namespace vlaforge::runtime {

Status NumericalLeaseSet::Add(
    std::uint32_t region_id, const VLAForgeNumericalProviderApi* provider,
    const VLAForgeNumericalRequirementView* requirement) noexcept {
  if (acquired_ ||
      vlaforge_numerical_provider_api_validate(provider).code != VLAFORGE_STATUS_OK ||
      vlaforge_numerical_requirement_validate(requirement).code != VLAFORGE_STATUS_OK) {
    return Failure(region_id, "numerical provider/requirement preflight failed");
  }
  for (const auto& entry : entries_) {
    if (entry.region_id == region_id) {
      return Failure(region_id, "duplicate numerical Region");
    }
    const auto* prior = entry.requirement;
    const auto* api = entry.provider;
    if (Same(api->process_domain, api->process_domain_size,
             provider->process_domain, provider->process_domain_size) &&
        (!Same(prior->policy_namespace, prior->namespace_size,
               requirement->policy_namespace, requirement->namespace_size) ||
         !Same(prior->policy_sha256, prior->policy_sha256_size,
               requirement->policy_sha256, requirement->policy_sha256_size) ||
         !SameFields(*prior, *requirement))) {
      return Failure(region_id, "conflicting requirements in numerical process domain");
    }
  }
  try {
    if (provider->query_support(requirement).code != VLAFORGE_STATUS_OK) {
      return Failure(region_id, "numerical policy unsupported by provider");
    }
    entries_.push_back({region_id, provider, requirement, nullptr});
    return Status::Ok();
  } catch (...) {
    return Failure(region_id, "numerical provider query or staging failed");
  }
}

Status NumericalLeaseSet::AcquireAll() noexcept {
  if (acquired_) { return Failure(0u, "numerical leases already acquired"); }
  for (auto& entry : entries_) {
    try {
      const auto status = entry.provider->acquire_current(entry.requirement, &entry.lease);
      if (status.code == VLAFORGE_STATUS_OK && entry.lease != nullptr) { continue; }
    } catch (...) {
    }
    const auto id = entry.region_id;
    Clear();
    return Failure(id, "numerical current-policy acquisition failed");
  }
  acquired_ = true;
  return Status::Ok();
}

Status NumericalLeaseSet::Bind(
    std::uint32_t region_id, VLAForgeRegionExecutable* region) noexcept {
  if (!acquired_ || region == nullptr) {
    return Failure(region_id, "numerical lease is not acquired before load");
  }
  for (const auto& entry : entries_) {
    if (entry.region_id != region_id) { continue; }
    try {
      if (entry.provider->bind_region(region, entry.lease).code == VLAFORGE_STATUS_OK) {
        return Status::Ok();
      }
    } catch (...) {
    }
    return Failure(region_id, "numerical Region lease binding failed");
  }
  return Failure(region_id, "numerical Region lease is absent");
}

Status NumericalLeaseSet::Validate(VLAForgeNumericalBoundary boundary) noexcept {
  if (!acquired_ || boundary < VLAFORGE_NUMERICAL_BEFORE_LOAD ||
      boundary > VLAFORGE_NUMERICAL_BEFORE_COMMIT) {
    return Failure(0u, "invalid numerical validation boundary");
  }
  for (const auto& entry : entries_) {
    try {
      if (entry.provider->validate_current(entry.lease, boundary).code == VLAFORGE_STATUS_OK) {
        continue;
      }
    } catch (...) {
    }
    return Failure(entry.region_id, "numerical current policy changed or validation failed");
  }
  return Status::Ok();
}

void NumericalLeaseSet::Clear() noexcept {
  for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
    if (it->lease != nullptr) {
      try { it->provider->release(it->lease); } catch (...) {}
      it->lease = nullptr;
    }
  }
  entries_.clear();
  acquired_ = false;
}

void NumericalLeaseSet::Abandon() noexcept {
  entries_.clear();
  acquired_ = false;
}

}  // namespace vlaforge::runtime
