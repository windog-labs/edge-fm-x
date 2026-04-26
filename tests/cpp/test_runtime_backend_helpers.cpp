#include "backends/runtime_backend.h"

#include <edge-fm/core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_contiguous_copy() {
    std::vector<uint8_t> src(24);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<uint8_t>(i + 1);
    }
    std::vector<uint8_t> dst(src.size(), 0);

    edge_fm::RuntimeTensorView view;
    view.data = dst.data();
    view.shape = {2, 3, 4};
    view.dtype = edge_fm::RuntimeDType::UInt8;
    view.device = edge_fm::RuntimeDevice::CPU;

    require(edge_fm::runtime_tensor_is_contiguous(view), "empty stride should be contiguous");
    require(edge_fm::runtime_tensor_byte_size(view) == src.size(), "contiguous byte size mismatch");

    edge_fm::copy_contiguous_to_runtime_buffer(src.data(), view);
    require(dst == src, "contiguous copy-in mismatch");

    std::vector<uint8_t> roundtrip(src.size(), 0);
    edge_fm::copy_runtime_buffer_to_contiguous(view, roundtrip.data());
    require(roundtrip == src, "contiguous copy-out mismatch");
}

void test_strided_copy() {
    std::vector<uint8_t> src(2 * 3 * 4);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<uint8_t>(i + 1);
    }
    std::vector<uint8_t> padded(2 * 32, 0xEE);

    edge_fm::RuntimeTensorView view;
    view.data = padded.data();
    view.shape = {2, 3, 4};
    view.dtype = edge_fm::RuntimeDType::UInt8;
    view.device = edge_fm::RuntimeDevice::CPU;
    view.stride = {32, 8, 1};

    require(!edge_fm::runtime_tensor_is_contiguous(view), "explicit padded stride should not be contiguous");
    require(edge_fm::runtime_tensor_byte_size(view) == 52, "strided byte size mismatch");

    edge_fm::copy_contiguous_to_runtime_buffer(src.data(), view);
    for (int batch = 0; batch < 2; ++batch) {
        for (int row = 0; row < 3; ++row) {
            const size_t src_offset = static_cast<size_t>((batch * 3 + row) * 4);
            const size_t dst_offset = static_cast<size_t>(batch * 32 + row * 8);
            require(
                std::equal(src.begin() + static_cast<std::ptrdiff_t>(src_offset),
                           src.begin() + static_cast<std::ptrdiff_t>(src_offset + 4),
                           padded.begin() + static_cast<std::ptrdiff_t>(dst_offset)),
                "strided copy-in logical row mismatch");
            require(padded[dst_offset + 4] == 0xEE, "strided copy-in overwrote padding");
        }
    }

    std::vector<uint8_t> roundtrip(src.size(), 0);
    edge_fm::copy_runtime_buffer_to_contiguous(view, roundtrip.data());
    require(roundtrip == src, "strided copy-out mismatch");
}

void test_invalid_stride_rejected() {
    uint8_t storage[8] = {};
    edge_fm::RuntimeTensorView view;
    view.data = storage;
    view.shape = {2, 4};
    view.dtype = edge_fm::RuntimeDType::UInt8;
    view.device = edge_fm::RuntimeDevice::CPU;
    view.stride = {8, 2};

    bool threw = false;
    try {
        uint8_t src[8] = {};
        edge_fm::copy_contiguous_to_runtime_buffer(src, view);
    } catch (const edge_fm::InvalidRequestError&) {
        threw = true;
    }
    require(threw, "invalid innermost stride should throw InvalidRequestError");
}

void test_runtime_backend_factory_boundaries() {
    bool cuda_threw = false;
    try {
        (void)edge_fm::create_runtime_backend(edge_fm::BackendTarget::Cuda);
    } catch (const edge_fm::ConfigurationError& exc) {
        cuda_threw = std::string(exc.what()).find("StandardEngine") != std::string::npos;
    }
    require(cuda_threw, "CUDA must not use the whole-graph runtime backend interface");

#ifndef EDGE_FM_ENABLE_HORIZON_RUNTIME
    bool horizon_threw = false;
    try {
        (void)edge_fm::create_runtime_backend(edge_fm::BackendTarget::Horizon);
    } catch (const edge_fm::ConfigurationError& exc) {
        horizon_threw = std::string(exc.what()).find("Horizon runtime backend is not compiled") !=
            std::string::npos;
    }
    require(horizon_threw, "Horizon runtime factory should report not-compiled when SDK support is off");
#else
    require(
        static_cast<bool>(edge_fm::create_runtime_backend(edge_fm::BackendTarget::Horizon)),
        "Horizon runtime factory should create a backend when SDK support is on");
#endif
}

} // namespace

int main() {
    try {
        test_contiguous_copy();
        test_strided_copy();
        test_invalid_stride_rejected();
        test_runtime_backend_factory_boundaries();
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
    return 0;
}
