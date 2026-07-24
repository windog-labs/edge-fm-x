#include "vlaforge/runtime/static_arena.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <utility>

namespace vlaforge::runtime {
namespace {

bool IsPowerOfTwo(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const std::size_t increment = alignment - remainder;
  if (value > static_cast<std::size_t>(-1) - increment) {
    throw std::overflow_error("static arena size overflow");
  }
  return value + increment;
}

}  // namespace

StaticArena::StaticArena(std::size_t size_bytes, std::size_t alignment)
    : size_bytes_(size_bytes), alignment_(alignment) {
  if (!IsPowerOfTwo(alignment)) {
    throw std::invalid_argument(
        "static arena alignment must be a non-zero power of two");
  }
  const std::size_t allocation_alignment =
      std::max(alignment, alignof(std::max_align_t));
  allocated_bytes_ =
      AlignUp(std::max<std::size_t>(size_bytes, 1), allocation_alignment);
  data_ = std::aligned_alloc(allocation_alignment, allocated_bytes_);
  if (data_ == nullptr) {
    throw std::bad_alloc();
  }
}

StaticArena::~StaticArena() { Release(); }

StaticArena::StaticArena(StaticArena&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_bytes_(std::exchange(other.size_bytes_, 0)),
      allocated_bytes_(std::exchange(other.allocated_bytes_, 0)),
      alignment_(std::exchange(other.alignment_, 1)) {}

StaticArena& StaticArena::operator=(StaticArena&& other) noexcept {
  if (this != &other) {
    Release();
    data_ = std::exchange(other.data_, nullptr);
    size_bytes_ = std::exchange(other.size_bytes_, 0);
    allocated_bytes_ = std::exchange(other.allocated_bytes_, 0);
    alignment_ = std::exchange(other.alignment_, 1);
  }
  return *this;
}

void* StaticArena::Resolve(std::size_t offset, std::size_t size_bytes,
                           std::size_t alignment) noexcept {
  if (data_ == nullptr || !IsPowerOfTwo(alignment) || offset > size_bytes_ ||
      size_bytes > size_bytes_ - offset) {
    return nullptr;
  }
  auto* result = static_cast<std::byte*>(data_) + offset;
  if (reinterpret_cast<std::uintptr_t>(result) % alignment != 0) {
    return nullptr;
  }
  return result;
}

const void* StaticArena::Resolve(std::size_t offset, std::size_t size_bytes,
                                 std::size_t alignment) const noexcept {
  return const_cast<StaticArena*>(this)->Resolve(offset, size_bytes, alignment);
}

void StaticArena::Release() noexcept {
  std::free(data_);
  data_ = nullptr;
  size_bytes_ = 0;
  allocated_bytes_ = 0;
  alignment_ = 1;
}

}  // namespace vlaforge::runtime
