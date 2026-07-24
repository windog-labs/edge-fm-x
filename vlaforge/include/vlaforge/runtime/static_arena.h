#ifndef VLAFORGE_RUNTIME_STATIC_ARENA_H_
#define VLAFORGE_RUNTIME_STATIC_ARENA_H_

#include <cstddef>

namespace vlaforge::runtime {

class StaticArena final {
 public:
  StaticArena(std::size_t size_bytes, std::size_t alignment);
  ~StaticArena();

  StaticArena(const StaticArena&) = delete;
  StaticArena& operator=(const StaticArena&) = delete;
  StaticArena(StaticArena&& other) noexcept;
  StaticArena& operator=(StaticArena&& other) noexcept;

  [[nodiscard]] void* Resolve(std::size_t offset, std::size_t size_bytes,
                              std::size_t alignment = 1) noexcept;
  [[nodiscard]] const void* Resolve(std::size_t offset,
                                    std::size_t size_bytes,
                                    std::size_t alignment = 1) const noexcept;

  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size_bytes() const noexcept { return size_bytes_; }
  [[nodiscard]] std::size_t alignment() const noexcept { return alignment_; }

 private:
  void Release() noexcept;

  void* data_ = nullptr;
  std::size_t size_bytes_ = 0;
  std::size_t allocated_bytes_ = 0;
  std::size_t alignment_ = 1;
};

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_STATIC_ARENA_H_
