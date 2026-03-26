#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <cuda_runtime.h>
#include <edge-fm/core.h>
#include "cuda_utils.h"
#include "utils/non_copyable.h"

namespace edge_fm {

/**
 * @brief 静态缓冲区管理器（单例模式）
 * 
 * 用于管理命名的 CUDA 缓冲区，支持按名称缓存和重用缓冲区
 */
class StaticBufferManager : public NonCopyableNonMovable {
public:
    /**
     * @brief 获取或分配缓存的 CUDA 缓冲区
     * 
     * @param name 缓冲区名称，用于标识不同的缓冲区
     * @param bytes 需要的缓冲区大小（字节数）
     * @param device_id CUDA 设备 ID
     * @return void* 指向缓冲区的指针
     * 
     * @note 这个函数会缓存缓冲区，如果缓冲区不存在或大小不够，会重新分配
     *       缓冲区在程序结束时自动释放
     */
    static void* get_cache_buf(const std::string& name, size_t bytes, int device_id = 0) {
        return instance().get_cache_buf_impl(name, bytes, device_id);
    }

private:
    StaticBufferManager() = default;
    ~StaticBufferManager() = default;
    
    /**
     * @brief 获取单例实例（内部使用）
     */
    static StaticBufferManager& instance() {
        static StaticBufferManager inst;
        return inst;
    }
    
    // 缓存缓冲区结构（RAII 管理 CUDA 内存）
    struct StaticBuffer: public NonCopyable {
        void* ptr = nullptr;
        size_t size = 0;
        
        // 构造函数：分配 CUDA 内存
        StaticBuffer(int device_id, size_t bytes) : size(bytes) {
            cudaSetDevice(device_id);
            CUDA_CHECK(cudaMalloc(&ptr, bytes));
        }
        
        // 析构函数：释放 CUDA 内存
        ~StaticBuffer() {
            if (ptr != nullptr) {
                cudaFree(ptr);
                ptr = nullptr;
            }
        }
        
        // 禁止拷贝，允许移动
        StaticBuffer(StaticBuffer&& other) noexcept 
            : ptr(other.ptr), size(other.size) {
            other.ptr = nullptr;
            other.size = 0;
        }

        StaticBuffer& operator=(StaticBuffer&& other) noexcept {
            if (this != &other) {
                // 释放当前资源
                if (ptr != nullptr) {
                    cudaFree(ptr);
                }
                // 移动资源
                ptr = other.ptr;
                size = other.size;
                other.ptr = nullptr;
                other.size = 0;
            }
            return *this;
        }
    };

    void* get_cache_buf_impl(const std::string& name, size_t bytes, int device_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 构造 key: "name_device_id"
        std::string key = name + "_" + std::to_string(device_id);
        
        auto it = cache_buf_.find(key);
        
        // 如果缓冲区不存在或大小不够，重新分配
        if (it == cache_buf_.end() || it->second.size < bytes) {
            if (it != cache_buf_.end()) {
                it->second = StaticBuffer(device_id, bytes);
            } else {
                cache_buf_.emplace(key, StaticBuffer(device_id, bytes));
                it = cache_buf_.find(key);
            }
        }
        
        // 返回缓冲区的指针
        return it->second.ptr;
    }
    
    std::unordered_map<std::string, StaticBuffer> cache_buf_;
    std::mutex mutex_;
};

// 动态内存池管理器（单例模式）
class MemoryPool : public NonCopyableNonMovable {
public:
    static MemoryPool& instance() {
        static MemoryPool inst;
        return inst;
    }

    void* allocate(size_t bytes, cudaStream_t stream, int device_id = 0) {
        CUDA_CHECK_THROW(cudaSetDevice(device_id), "Failed to set device");
        void* ptr = nullptr;
        // Use CUDA default memory pool via stream-ordered allocator
        CUDA_CHECK_THROW_EX(cudaMallocAsync(&ptr, bytes, stream),
            "Failed to allocate async memory (cudaMallocAsync)", OutOfMemoryError);

        return ptr;
    }

private:
    MemoryPool() = default;
    ~MemoryPool() = default;
};

} // namespace edge_fm

