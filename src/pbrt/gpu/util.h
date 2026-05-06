// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_UTIL_H
#define PBRT_GPU_UTIL_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/log.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/progressreporter.h>

#include <map>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef NVTX
#ifdef UNICODE
#undef UNICODE
#endif
#include <nvtx3/nvToolsExt.h>

#ifdef RGB
#undef RGB
#endif  // RGB
#endif

#define CUDA_CHECK(EXPR)                                        \
    if (EXPR != cudaSuccess) {                                  \
        cudaError_t error = cudaGetLastError();                 \
        LOG_FATAL("CUDA error: %s", cudaGetErrorString(error)); \
    } else /* eat semicolon */

#define CU_CHECK(EXPR)                                              \
    do {                                                            \
        CUresult result = EXPR;                                     \
        if (result != CUDA_SUCCESS) {                               \
            const char *str;                                        \
            CHECK_EQ(CUDA_SUCCESS, cuGetErrorString(result, &str)); \
            LOG_FATAL("CUDA error: %s", str);                       \
        }                                                           \
    } while (false) /* eat semicolon */

namespace pbrt {

enum class GPUMemoryKind {
    Device,
    // CPU 访问时，页面可能迁移到主机内存, GPU kernel 访问时，页面可能迁移到 GPU 显存。
    // 由cudaruntime管理，可以通过prefetch来预先传输到显存
    Managed,
    HostPinned,
    MipmappedArray
};

std::pair<cudaEvent_t, cudaEvent_t> GetProfilerEvents(const char *description);

cudaError_t TrackedCudaMalloc(void **ptr, size_t size, const char *label, const char *file,
                              int line);
cudaError_t TrackedCudaMallocManaged(void **ptr, size_t size, const char *label,
                                     const char *file, int line);
cudaError_t TrackedCudaMallocHost(void **ptr, size_t size, const char *label,
                                  const char *file, int line);
cudaError_t TrackedCudaFree(void *ptr, const char *label, const char *file, int line);
cudaError_t TrackedCudaFreeHost(void *ptr, const char *label, const char *file, int line);
cudaError_t TrackedCudaMallocMipmappedArray(cudaMipmappedArray_t *mipArray,
                                            const cudaChannelFormatDesc *desc,
                                            cudaExtent extent, unsigned int numLevels,
                                            unsigned int flags, const char *label,
                                            const char *file, int line);
cudaError_t TrackedCudaFreeMipmappedArray(cudaMipmappedArray_t mipArray, const char *label,
                                          const char *file, int line);

template <typename T>
inline cudaError_t TrackedCudaMalloc(T **ptr, size_t size, const char *label,
                                     const char *file, int line) {
    return TrackedCudaMalloc(reinterpret_cast<void **>(ptr), size, label, file, line);
}

template <typename T>
inline cudaError_t TrackedCudaMallocManaged(T **ptr, size_t size, const char *label,
                                            const char *file, int line) {
    return TrackedCudaMallocManaged(reinterpret_cast<void **>(ptr), size, label, file,
                                    line);
}

template <typename T>
inline cudaError_t TrackedCudaMallocHost(T **ptr, size_t size, const char *label,
                                         const char *file, int line) {
    return TrackedCudaMallocHost(reinterpret_cast<void **>(ptr), size, label, file, line);
}

template <typename F>
inline int GetBlockSize(const char *description, F kernel) {
    // Note: this isn't reentrant, but that's fine for our purposes...
    static std::map<std::type_index, int> kernelBlockSizes;

    std::type_index index = std::type_index(typeid(F));

    auto iter = kernelBlockSizes.find(index);
    if (iter != kernelBlockSizes.end())
        return iter->second;

    int minGridSize, blockSize;
    CUDA_CHECK(
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0));
    kernelBlockSizes[index] = blockSize;
    LOG_VERBOSE("[%s]: block size %d", description, blockSize);

    return blockSize;
}

#ifdef __NVCC__
template <typename F>
__global__ void Kernel(F func, int nItems) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nItems)
        return;

    func(tid);
}

// GPU Launch Function Declarations
template <typename F>
void GPUParallelFor(const char *description, int nItems, F func);

template <typename F>
void GPUParallelFor(const char *description, int nItems, F func) {
#ifdef NVTX
    nvtxRangePush(description);
#endif
    auto kernel = &Kernel<F>;

    int blockSize = GetBlockSize(description, kernel);
    std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(description);

#ifdef PBRT_DEBUG_BUILD
    LOG_VERBOSE("Launching %s", description);
#endif
    cudaEventRecord(events.first);
    int gridSize = (nItems + blockSize - 1) / blockSize;
    kernel<<<gridSize, blockSize>>>(func, nItems);
    cudaEventRecord(events.second);

#ifdef PBRT_DEBUG_BUILD
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG_VERBOSE("Post-sync %s", description);
#endif
#ifdef NVTX
    nvtxRangePop();
#endif
}

#endif  // __NVCC__

// GPU Synchronization Function Declarations
void GPUWait();

void ReportKernelStats();

void GPUInit();
void GPUThreadInit();

void GPUMemset(void *ptr, int byte, size_t bytes);

void GPURegisterThread(const char *name);
void GPUNameStream(cudaStream_t stream, const char *name);

}  // namespace pbrt

#define CUDA_MALLOC(ptr, label, size) \
    CUDA_CHECK(::pbrt::TrackedCudaMalloc(ptr, size, label, __FILE__, __LINE__))
#define CUDA_MALLOC_MANAGED(ptr, label, size) \
    CUDA_CHECK(::pbrt::TrackedCudaMallocManaged(ptr, size, label, __FILE__, __LINE__))
#define CUDA_MALLOC_HOST(ptr, label, size) \
    CUDA_CHECK(::pbrt::TrackedCudaMallocHost(ptr, size, label, __FILE__, __LINE__))
#define CUDA_FREE(ptr, label) \
    CUDA_CHECK(::pbrt::TrackedCudaFree(ptr, label, __FILE__, __LINE__))
#define CUDA_FREE_HOST(ptr, label) \
    CUDA_CHECK(::pbrt::TrackedCudaFreeHost(ptr, label, __FILE__, __LINE__))
#define CUDA_MALLOC_MIPMAPPED_ARRAY(mipArray, label, desc, extent, numLevels, flags) \
    CUDA_CHECK(::pbrt::TrackedCudaMallocMipmappedArray(                              \
        mipArray, desc, extent, numLevels, flags, label, __FILE__, __LINE__))
#define CUDA_FREE_MIPMAPPED_ARRAY(mipArray, label) \
    CUDA_CHECK(::pbrt::TrackedCudaFreeMipmappedArray(mipArray, label, __FILE__, __LINE__))

#endif  // PBRT_GPU_UTIL_H
