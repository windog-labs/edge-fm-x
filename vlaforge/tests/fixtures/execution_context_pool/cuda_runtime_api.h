#pragma once
#include <stddef.h>
typedef struct FakeStream* cudaStream_t;
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorUnknown = 30, cudaErrorStreamCaptureUnsupported = 900 };
enum { cudaStreamNonBlocking = 1, cudaMemcpyDeviceToDevice = 3 };
cudaError_t cudaGetDevice(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, int, cudaStream_t);
const char* cudaGetErrorString(cudaError_t);
