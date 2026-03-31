// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/gpu/util.h>

#include <pbrt/options.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/log.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef NVTX
#ifdef PBRT_IS_WINDOWS
#include <windows.h>
#else
#include <sys/syscall.h>
#endif  // PBRT_IS_WINDOWS
#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"
#endif

namespace pbrt {

namespace {

struct AllocationRecord {
    size_t size = 0;
    GPUMemoryKind kind = GPUMemoryKind::Device;
    std::string label;
    const char *file = nullptr;
    int line = 0;
};

struct MemoryCounters {
    size_t current = 0;
    size_t peak = 0;
    size_t totalAllocated = 0;
    size_t allocationCount = 0;
    size_t peakAllocationCount = 0;
};

constexpr int GPUMemoryKindCount = 4;

int KindIndex(GPUMemoryKind kind) {
    return static_cast<int>(kind);
}

const char *KindName(GPUMemoryKind kind) {
    switch (kind) {
    case GPUMemoryKind::Device:
        return "Device";
    case GPUMemoryKind::Managed:
        return "Managed";
    case GPUMemoryKind::HostPinned:
        return "Pinned host";
    case GPUMemoryKind::MipmappedArray:
        return "Mipmapped array";
    }
    return "Unknown";
}

size_t ChannelDescBytesPerTexel(const cudaChannelFormatDesc &desc) {
    int bits = desc.x + desc.y + desc.z + desc.w;
    return (bits + 7) / 8;
}

size_t EstimateMipmappedArrayBytes(const cudaChannelFormatDesc &desc, cudaExtent extent,
                                   unsigned int numLevels) {
    size_t bytesPerTexel = ChannelDescBytesPerTexel(desc);
    size_t total = 0;
    size_t width = extent.width, height = std::max<size_t>(1, extent.height),
           depth = std::max<size_t>(1, extent.depth ? extent.depth : 1);
    for (unsigned int level = 0; level < numLevels; ++level) {
        total += width * height * depth * bytesPerTexel;
        width = std::max<size_t>(1, width / 2);
        height = std::max<size_t>(1, height / 2);
        depth = std::max<size_t>(1, depth / 2);
    }
    return total;
}

class GPUMemoryTracker {
  public:
    void RecordAllocation(uintptr_t key, size_t size, GPUMemoryKind kind, const char *label,
                          const char *file, int line) {
        std::lock_guard<std::mutex> lock(mutex);
        std::string labelString = label ? label : "(unlabeled)";
        allocations[key] = AllocationRecord{size, kind, labelString, file, line};
        MemoryCounters &counters = byKind[KindIndex(kind)];
        counters.current += size;
        counters.peak = std::max(counters.peak, counters.current);
        counters.totalAllocated += size;
        counters.allocationCount += 1;
        counters.peakAllocationCount =
            std::max(counters.peakAllocationCount, counters.allocationCount);
        totalTrackedAllocationCount += 1;
        peakTrackedAllocationCount =
            std::max(peakTrackedAllocationCount, totalTrackedAllocationCount);
        if (kind != GPUMemoryKind::HostPinned) {
            currentGPUTrackedBytes += size;
            peakGPUTrackedBytes = std::max(peakGPUTrackedBytes, currentGPUTrackedBytes);
        }
        MemoryCounters &labelCounters = byLabel[labelString];
        labelCounters.current += size;
        labelCounters.peak = std::max(labelCounters.peak, labelCounters.current);
        labelCounters.totalAllocated += size;
        labelCounters.allocationCount += 1;
        labelCounters.peakAllocationCount =
            std::max(labelCounters.peakAllocationCount, labelCounters.allocationCount);
    }

    void RecordFree(uintptr_t key, GPUMemoryKind expectedKind, const char *label,
                    const char *file, int line) {
        if (key == 0)
            return;

        std::lock_guard<std::mutex> lock(mutex);
        auto iter = allocations.find(key);
        if (iter == allocations.end()) {
            LOG_VERBOSE("GPU memory tracker: untracked free at %s:%d (%s)", file, line,
                        KindName(expectedKind));
            return;
        }
        AllocationRecord record = iter->second;
        allocations.erase(iter);
        if (label && record.label != label)
            LOG_VERBOSE("GPU memory tracker: label mismatch at free %s:%d (alloc \"%s\", "
                        "free \"%s\")",
                        file, line, record.label.c_str(), label);
        MemoryCounters &counters = byKind[KindIndex(record.kind)];
        counters.current -= record.size;
        counters.allocationCount -= 1;
        totalTrackedAllocationCount -= 1;
        if (record.kind != GPUMemoryKind::HostPinned)
            currentGPUTrackedBytes -= record.size;
        MemoryCounters &labelCounters = byLabel[record.label];
        labelCounters.current -= record.size;
        labelCounters.allocationCount -= 1;
    }

    MemoryCounters Counters(GPUMemoryKind kind) const {
        std::lock_guard<std::mutex> lock(mutex);
        return byKind[KindIndex(kind)];
    }

    size_t CurrentGPUTrackedBytes() const {
        std::lock_guard<std::mutex> lock(mutex);
        return currentGPUTrackedBytes;
    }

    size_t PeakGPUTrackedBytes() const {
        std::lock_guard<std::mutex> lock(mutex);
        return peakGPUTrackedBytes;
    }

    size_t CurrentTrackedAllocationCount() const {
        std::lock_guard<std::mutex> lock(mutex);
        return totalTrackedAllocationCount;
    }

    size_t PeakTrackedAllocationCount() const {
        std::lock_guard<std::mutex> lock(mutex);
        return peakTrackedAllocationCount;
    }

    std::vector<std::pair<std::string, MemoryCounters>> LabelCounters() const {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<std::pair<std::string, MemoryCounters>> result;
        result.reserve(byLabel.size());
        for (const auto &entry : byLabel)
            result.push_back(entry);
        std::sort(result.begin(), result.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
        return result;
    }

  private:
    mutable std::mutex mutex;
    std::unordered_map<uintptr_t, AllocationRecord> allocations;
    std::array<MemoryCounters, GPUMemoryKindCount> byKind{};
    std::unordered_map<std::string, MemoryCounters> byLabel;
    size_t currentGPUTrackedBytes = 0;
    size_t peakGPUTrackedBytes = 0;
    size_t totalTrackedAllocationCount = 0;
    size_t peakTrackedAllocationCount = 0;
};

GPUMemoryTracker gpuMemoryTracker;
std::atomic<size_t> gpuRuntimeBaselineBytes{0};
std::atomic<bool> gpuRuntimeBaselineValid{false};

void ReportGPUMemoryStats(StatsAccumulator &accum) {
    auto reportKind = [&](GPUMemoryKind kind, const char *currentName, const char *peakName,
                          const char *totalName, const char *countName,
                          const char *peakCountName) {
        MemoryCounters counters = gpuMemoryTracker.Counters(kind);
        accum.ReportMemoryCounter(currentName, counters.current);
        accum.ReportMemoryCounter(peakName, counters.peak);
        accum.ReportMemoryCounter(totalName, counters.totalAllocated);
        accum.ReportCounter(countName, counters.allocationCount);
        accum.ReportCounter(peakCountName, counters.peakAllocationCount);
    };

    reportKind(GPUMemoryKind::Device, "Memory/CUDA device current",
               "Memory/CUDA device peak", "Memory/CUDA device total allocated",
               "Memory/CUDA device live allocations",
               "Memory/CUDA device peak live allocations");
    reportKind(GPUMemoryKind::Managed, "Memory/CUDA managed current",
               "Memory/CUDA managed peak", "Memory/CUDA managed total allocated",
               "Memory/CUDA managed live allocations",
               "Memory/CUDA managed peak live allocations");
    reportKind(GPUMemoryKind::HostPinned, "Memory/CUDA pinned host current",
               "Memory/CUDA pinned host peak",
               "Memory/CUDA pinned host total allocated",
               "Memory/CUDA pinned host live allocations",
               "Memory/CUDA pinned host peak live allocations");
    reportKind(GPUMemoryKind::MipmappedArray, "Memory/CUDA mipmapped array current",
               "Memory/CUDA mipmapped array peak",
               "Memory/CUDA mipmapped array total allocated",
               "Memory/CUDA mipmapped array live allocations",
               "Memory/CUDA mipmapped array peak live allocations");

    size_t trackedGPUCurrent = gpuMemoryTracker.CurrentGPUTrackedBytes();
    size_t trackedGPUPeak = gpuMemoryTracker.PeakGPUTrackedBytes();
    accum.ReportMemoryCounter("Memory/CUDA tracked GPU current", trackedGPUCurrent);
    accum.ReportMemoryCounter("Memory/CUDA tracked GPU peak", trackedGPUPeak);
    accum.ReportCounter("Memory/CUDA tracked live allocations",
                        gpuMemoryTracker.CurrentTrackedAllocationCount());
    accum.ReportCounter("Memory/CUDA tracked peak live allocations",
                        gpuMemoryTracker.PeakTrackedAllocationCount());

    for (const auto &entry : gpuMemoryTracker.LabelCounters()) {
        const std::string prefix = "Memory/CUDA label " + entry.first;
        accum.ReportMemoryCounter((prefix + " current").c_str(), entry.second.current);
        accum.ReportMemoryCounter((prefix + " peak").c_str(), entry.second.peak);
        accum.ReportMemoryCounter((prefix + " total allocated").c_str(),
                                  entry.second.totalAllocated);
        accum.ReportCounter((prefix + " live allocations").c_str(),
                            entry.second.allocationCount);
        accum.ReportCounter((prefix + " peak live allocations").c_str(),
                            entry.second.peakAllocationCount);
    }

    if (Options && Options->useGPU) {
        size_t freeBytes, totalBytes;
        if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
            size_t runtimeUsed = totalBytes - freeBytes;
            accum.ReportMemoryCounter("Memory/CUDA runtime in use", runtimeUsed);
            if (gpuRuntimeBaselineValid.load()) {
                size_t baseline = gpuRuntimeBaselineBytes.load();
                size_t delta = runtimeUsed > baseline ? runtimeUsed - baseline : 0;
                accum.ReportMemoryCounter("Memory/CUDA runtime over init baseline", delta);
                size_t gap = delta > trackedGPUCurrent ? delta - trackedGPUCurrent : 0;
                accum.ReportMemoryCounter("Memory/CUDA runtime untracked estimate", gap);
            } else {
                size_t gap = runtimeUsed > trackedGPUCurrent ? runtimeUsed - trackedGPUCurrent : 0;
                accum.ReportMemoryCounter("Memory/CUDA runtime untracked estimate", gap);
            }
        }
    }
}

static StatRegisterer gpuMemoryStatsRegisterer(ReportGPUMemoryStats);

}  // namespace

cudaError_t TrackedCudaMalloc(void **ptr, size_t size, const char *label, const char *file,
                              int line) {
    cudaError_t result = cudaMalloc(ptr, size);
    if (result == cudaSuccess && ptr && *ptr)
        gpuMemoryTracker.RecordAllocation(reinterpret_cast<uintptr_t>(*ptr), size,
                                          GPUMemoryKind::Device, label, file, line);
    return result;
}

cudaError_t TrackedCudaMallocManaged(void **ptr, size_t size, const char *label,
                                     const char *file, int line) {
    cudaError_t result = cudaMallocManaged(ptr, size);
    if (result == cudaSuccess && ptr && *ptr)
        gpuMemoryTracker.RecordAllocation(reinterpret_cast<uintptr_t>(*ptr), size,
                                          GPUMemoryKind::Managed, label, file, line);
    return result;
}

cudaError_t TrackedCudaMallocHost(void **ptr, size_t size, const char *label,
                                  const char *file, int line) {
    cudaError_t result = cudaMallocHost(ptr, size);
    if (result == cudaSuccess && ptr && *ptr)
        gpuMemoryTracker.RecordAllocation(reinterpret_cast<uintptr_t>(*ptr), size,
                                          GPUMemoryKind::HostPinned, label, file, line);
    return result;
}

cudaError_t TrackedCudaFree(void *ptr, const char *label, const char *file, int line) {
    cudaError_t result = cudaFree(ptr);
    if (result == cudaSuccess)
        gpuMemoryTracker.RecordFree(reinterpret_cast<uintptr_t>(ptr), GPUMemoryKind::Device,
                                    label, file, line);
    return result;
}

cudaError_t TrackedCudaFreeHost(void *ptr, const char *label, const char *file, int line) {
    cudaError_t result = cudaFreeHost(ptr);
    if (result == cudaSuccess)
        gpuMemoryTracker.RecordFree(reinterpret_cast<uintptr_t>(ptr),
                                    GPUMemoryKind::HostPinned, label, file, line);
    return result;
}

cudaError_t TrackedCudaMallocMipmappedArray(cudaMipmappedArray_t *mipArray,
                                            const cudaChannelFormatDesc *desc,
                                            cudaExtent extent, unsigned int numLevels,
                                            unsigned int flags, const char *label,
                                            const char *file, int line) {
    cudaError_t result =
        cudaMallocMipmappedArray(mipArray, desc, extent, numLevels, flags);
    if (result == cudaSuccess && mipArray && *mipArray) {
        size_t size = EstimateMipmappedArrayBytes(*desc, extent, numLevels);
        gpuMemoryTracker.RecordAllocation(reinterpret_cast<uintptr_t>(*mipArray), size,
                                          GPUMemoryKind::MipmappedArray, label, file, line);
    }
    return result;
}

cudaError_t TrackedCudaFreeMipmappedArray(cudaMipmappedArray_t mipArray, const char *label,
                                          const char *file, int line) {
    cudaError_t result = cudaFreeMipmappedArray(mipArray);
    if (result == cudaSuccess)
        gpuMemoryTracker.RecordFree(reinterpret_cast<uintptr_t>(mipArray),
                                    GPUMemoryKind::MipmappedArray, label, file, line);
    return result;
}

void GPUInit() {
    cudaFree(nullptr);

    int driverVersion;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    int runtimeVersion;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    auto versionToString = [](int version) {
        int major = version / 1000;
        int minor = (version - major * 1000) / 10;
        return StringPrintf("%d.%d", major, minor);
    };
    LOG_VERBOSE("GPU CUDA driver %s, CUDA runtime %s", versionToString(driverVersion),
                versionToString(runtimeVersion));

    int nDevices;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    std::string devices;
    int clockRateKHz = 0;
    for (int i = 0; i < nDevices; ++i) {
        cudaDeviceProp deviceProperties;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, i));
#if CUDART_VERSION >= 13000
        CUdevice cdevice;
        cuDeviceGet(&cdevice, i);
        cuDeviceGetAttribute(&clockRateKHz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cdevice);
#else
        clockRateKHz = deviceProperties.clockRate;
#endif
        CHECK(deviceProperties.canMapHostMemory);

        std::string deviceString = StringPrintf(
            "CUDA device %d (%s) with %f MiB, %d SMs running at %f MHz "
            "with shader model %d.%d",
            i, deviceProperties.name, deviceProperties.totalGlobalMem / (1024. * 1024.),
            deviceProperties.multiProcessorCount, clockRateKHz / 1000.,
            deviceProperties.major, deviceProperties.minor);
        LOG_VERBOSE("%s", deviceString);
        devices += deviceString + "\n";
    }

#ifdef PBRT_IS_WINDOWS
    if (nDevices > 1)
        ErrorExit("Found multiple GPUs.\n"
                  "On Windows, this unfortunately causes a significant slowdown with "
                  "pbrt.\n"
                  "Please select a single GPU and use the --gpu-device command line "
                  "option to specify it.\n"
                  "Found devices:\n%s",
                  devices);
#endif

    int device = Options->gpuDevice ? *Options->gpuDevice : 0;
    LOG_VERBOSE("Selecting GPU device %d", device);
#ifdef NVTX
    nvtxNameCuDevice(device, "PBRT_GPU");
#endif
    CUDA_CHECK(cudaSetDevice(device));

    int hasUnifiedAddressing;
    CUDA_CHECK(cudaDeviceGetAttribute(&hasUnifiedAddressing, cudaDevAttrUnifiedAddressing,
                                      device));
    if (!hasUnifiedAddressing)
        LOG_FATAL("The selected GPU device (%d) does not support unified addressing.",
                  device);

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
    size_t stackSize;
    CUDA_CHECK(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
    LOG_VERBOSE("Reset stack size to %d", stackSize);

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 32 * 1024 * 1024));

    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

#ifdef NVTX
#ifdef PBRT_IS_WINDOWS
    nvtxNameOsThread(GetCurrentThreadId(), "MAIN_THREAD");
#else
    nvtxNameOsThread(syscall(SYS_gettid), "MAIN_THREAD");
#endif
#endif  // NVTX

    size_t freeBytes, totalBytes;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    gpuRuntimeBaselineBytes = totalBytes - freeBytes;
    gpuRuntimeBaselineValid = true;
}

void GPUThreadInit() {
    if (!Options->useGPU)
        return;
    int device = Options->gpuDevice ? *Options->gpuDevice : 0;
    LOG_VERBOSE("Selecting GPU device %d", device);
    CUDA_CHECK(cudaSetDevice(device));
}

void GPURegisterThread(const char *name) {
#ifdef NVTX
#ifdef PBRT_IS_WINDOWS
    nvtxNameOsThread(GetCurrentThreadId(), name);
#else
    nvtxNameOsThread(syscall(SYS_gettid), name);
#endif
#endif
}

void GPUNameStream(cudaStream_t stream, const char *name) {
#ifdef NVTX
    nvtxNameCuStream(stream, name);
#endif
}

struct KernelStats {
    KernelStats(const char *description) : description(description) {}

    std::string description;
    int numLaunches = 0;
    float sumMS = 0, minMS = 0, maxMS = 0;
};

// Store pointers so that reallocs don't mess up held KernelStats pointers
// in ProfilerEvent..
static std::vector<KernelStats *> kernelStats;

struct ProfilerEvent {
    ProfilerEvent() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    void Sync() {
        CHECK(active);
        CUDA_CHECK(cudaEventSynchronize(start));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        ++stats->numLaunches;
        if (stats->numLaunches == 1)
            stats->sumMS = stats->minMS = stats->maxMS = ms;
        else {
            stats->sumMS += ms;
            stats->minMS = std::min(stats->minMS, ms);
            stats->maxMS = std::max(stats->maxMS, ms);
        }

        active = false;
    }

    cudaEvent_t start, stop;
    bool active = false;
    KernelStats *stats = nullptr;
};

// Ring buffer
static std::vector<ProfilerEvent> eventPool;
static size_t eventPoolOffset = 0;

std::pair<cudaEvent_t, cudaEvent_t> GetProfilerEvents(const char *description) {
    if (eventPool.empty())
        eventPool.resize(1024);  // how many? This is probably more than we need...

    if (eventPoolOffset == eventPool.size())
        eventPoolOffset = 0;

    ProfilerEvent &pe = eventPool[eventPoolOffset++];
    if (pe.active)
        pe.Sync();

    pe.active = true;
    pe.stats = nullptr;

    for (size_t i = 0; i < kernelStats.size(); ++i) {
        if (kernelStats[i]->description == description) {
            pe.stats = kernelStats[i];
            break;
        }
    }
    if (!pe.stats) {
        kernelStats.push_back(new KernelStats(description));
        pe.stats = kernelStats.back();
    }

    return {pe.start, pe.stop};
}

void GPUWait() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUMemset(void *ptr, int byte, size_t bytes) {
    CUDA_CHECK(cudaMemset(ptr, byte, bytes));
}

void ReportKernelStats() {
    CUDA_CHECK(cudaDeviceSynchronize());

    // Drain active profiler events
    for (size_t i = 0; i < eventPool.size(); ++i)
        if (eventPool[i].active)
            eventPool[i].Sync();

    // Compute total milliseconds over all kernels and launches
    float totalMS = 0;
    for (size_t i = 0; i < kernelStats.size(); ++i)
        totalMS += kernelStats[i]->sumMS;

    printf("Wavefront Kernel Profile:\n");
    int otherLaunches = 0;
    float otherMS = 0;
    const float otherCutoff = 0.001f * totalMS;
    for (size_t i = 0; i < kernelStats.size(); ++i) {
        KernelStats *stats = kernelStats[i];
        if (stats->sumMS > otherCutoff)
            Printf("  %-49s %5d launches %9.2f ms / %5.1f%s (avg %6.3f, min "
                   "%6.3f, max %7.3f)\n",
                   stats->description, stats->numLaunches, stats->sumMS,
                   100.f * stats->sumMS / totalMS, "%", stats->sumMS / stats->numLaunches,
                   stats->minMS, stats->maxMS);
        else {
            otherMS += stats->sumMS;
            otherLaunches += stats->numLaunches;
        }
    }
    Printf("  %-49s %5d launches %9.2f ms / %5.1f%s (avg %6.3f)\n", "Other",
           otherLaunches, otherMS, 100.f * otherMS / totalMS, "%",
           otherMS / otherLaunches);
    Printf("\nTotal rendering time: %9.2f ms\n", totalMS);
    Printf("\n");
}

}  // namespace pbrt
