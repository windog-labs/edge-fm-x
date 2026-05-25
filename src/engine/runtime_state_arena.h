#pragma once

#include <edge-fm/core.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace edge_fm {

class RuntimeStateArena {
public:
    RuntimeStateArena() = default;
    RuntimeStateArena(RuntimeStateArena&&) noexcept = default;
    RuntimeStateArena& operator=(RuntimeStateArena&&) noexcept = default;
    RuntimeStateArena(const RuntimeStateArena&) = delete;
    RuntimeStateArena& operator=(const RuntimeStateArena&) = delete;
    ~RuntimeStateArena() = default;

    Tensor& get_or_create(const std::string& name,
                          const std::vector<int64_t>& shape,
                          DType dtype,
                          Device device,
                          int32_t device_id = 0,
                          MemoryOwnership ownership = MemoryOwnership::ViewExternal,
                          void* stream_handle = nullptr);

    Tensor& bind_external_view(const std::string& name,
                               void* data_ptr,
                               const std::vector<int64_t>& shape,
                               DType dtype,
                               Device device,
                               int32_t device_id = 0);

    Tensor view(const std::string& name, const std::vector<int64_t>& shape) const;

    Tensor* find(const std::string& name);
    const Tensor* find(const std::string& name) const;
    bool contains(const std::string& name) const;
    std::vector<std::string> names() const;
    void clear();

private:
    std::unordered_map<std::string, Tensor> tensors_;
};

} // namespace edge_fm
