#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <edge-fm/core.h>

// 检查 CUDA 错误（直接退出程序，用于初始化等场景）
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

/**
 * @brief 检查 CUDA 错误并抛出 DeviceError 异常（宏定义）
 * 
 * @param err CUDA 错误代码或表达式
 * @param msg 错误消息（可以是字符串字面量或表达式）
 * @throws DeviceError 如果 err != cudaSuccess
 * 
 * @example
 * CUDA_CHECK_THROW(kernel_launch(), "Kernel launch failed");
 */
#define CUDA_CHECK_THROW(err, msg) \
    do { \
        cudaError_t _err = (err); \
        if (_err != cudaSuccess) { \
            throw DeviceError(std::string(msg) + ": " + std::string(cudaGetErrorString(_err))); \
        } \
    } while(0)

/**
 * @brief 检查 CUDA 错误并抛出自定义异常类型（宏定义）
 * 
 * @param err CUDA 错误代码或表达式
 * @param msg 错误消息（可以是字符串字面量或表达式）
 * @param ExceptionType 异常类型（如 OutOfMemoryError, DeviceError 等）
 * @throws ExceptionType 如果 err != cudaSuccess
 * 
 * @example
 * CUDA_CHECK_THROW_EX(cudaMalloc(&ptr, size), "Failed to allocate memory", OutOfMemoryError);
 */
#define CUDA_CHECK_THROW_EX(err, msg, ExceptionType) \
    do { \
        cudaError_t _err = (err); \
        if (_err != cudaSuccess) { \
            throw ExceptionType(std::string(msg) + ": " + std::string(cudaGetErrorString(_err))); \
        } \
    } while (0)

