#ifndef VLAFORGE_RUNTIME_TRANSACTION_H_
#define VLAFORGE_RUNTIME_TRANSACTION_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vlaforge/runtime/epoch.h"
#include "vlaforge/runtime/status.h"

namespace vlaforge::runtime {

struct PendingAction final {
  Epoch epoch{};
  const void* data = nullptr;
  std::size_t size_bytes = 0;
};

struct CommittedAction final {
  Epoch epoch{};
  const void* data = nullptr;
  std::size_t size_bytes = 0;
  std::uint64_t transaction_id = 0;
  bool valid = false;
};

enum class TransactionState : std::uint32_t {
  kIdle = 0,
  kActive = 1,
  kCommitted = 2,
  kAborted = 3,
};

struct PendingWrite final {
  std::uint32_t state_id = 0;
  Epoch epoch{};
  const std::byte* data = nullptr;
  std::size_t size_bytes = 0;
};

class Transaction final {
 public:
  explicit Transaction(std::size_t maximum_staged_states);

  Transaction(const Transaction&) = delete;
  Transaction& operator=(const Transaction&) = delete;
  Transaction(Transaction&&) = default;
  Transaction& operator=(Transaction&&) = default;

  [[nodiscard]] std::uint64_t id() const noexcept { return id_; }
  [[nodiscard]] const Epoch& tick() const noexcept { return tick_; }
  [[nodiscard]] TransactionState state() const noexcept { return state_; }
  [[nodiscard]] bool active() const noexcept {
    return state_ == TransactionState::kActive;
  }
  [[nodiscard]] std::size_t pending_count() const noexcept {
    return pending_count_;
  }
  [[nodiscard]] const PendingWrite& pending(std::size_t index) const noexcept {
    return pending_[index];
  }
  [[nodiscard]] std::size_t capacity() const noexcept {
    return pending_.size();
  }

 private:
  friend class StateStore;

  Status Begin(std::uint64_t id, const Epoch& tick) noexcept;
  Status Add(const PendingWrite& write) noexcept;
  void Close(TransactionState state) noexcept;

  std::uint64_t id_ = 0;
  Epoch tick_{};
  TransactionState state_ = TransactionState::kIdle;
  std::vector<PendingWrite> pending_;
  std::size_t pending_count_ = 0;
};

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_TRANSACTION_H_
