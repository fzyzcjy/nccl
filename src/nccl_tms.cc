#include "comm.h"
#include "utils.h"
#include "nccl_tms.h"
#include "json.hpp"

// NOTE MODIFIED from CUCHECK
#define CUCHECKEXIT(cmd) do {				      \
    CUresult err = pfn_##cmd;				      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				      \
      (void) pfn_cuGetErrorString(err, &errStr);	      \
      WARN("Cuda failure %d '%s'", err, errStr);	      \
      exit(1);			      \
    }							      \
} while(false)

// NOTE MODIFIED from CUDACHECK
#define CUDACHECKEXIT(cmd) do {                                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
        exit(1);                      \
    }                                                       \
} while(false)

// ref: ncclCuMemAlloc
CUmemAllocationProp getCUmemAllocationProp() {
    CUdevice currentDev;
    CUmemAllocationProp prop = {};
    // CUmemAccessDesc accessDesc = {};
    CUmemAllocationHandleType type = ncclCuMemHandleType;
    int cudaDev;
    int flag = 0;
    CUDACHECKEXIT(cudaGetDevice(&cudaDev));
    CUCHECKEXIT(cuDeviceGet(&currentDev, cudaDev));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.requestedHandleTypes = type;
    prop.location.id = currentDev;
    // Query device to see if RDMA support is available
    CUCHECKEXIT(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, currentDev));
    if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;
    return prop;
}

size_t alignSizeByGranularity(size_t size, CUmemAllocationProp prop) {
    size_t granularity = 0;
    CUCHECKEXIT(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    ALIGN_SIZE(size, granularity);
    return size;
}

NcclTms::NcclTms() {}

// 静态单例方法实现
NcclTms& NcclTms::instance() {
    static NcclTms instance;
    return instance;
}

void NcclTms::registerAlloc(void* ptr, size_t size, uint64_t rawCuDesc, NcclTmsIpcMode ipcMode) {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    WARN("NcclTms::registerAlloc ptr=%p, size=%zu, rawCuDesc=%lu, ipcMode=%d", ptr, size, rawCuDesc, static_cast<int>(ipcMode));
    records_.push_back(NcclTmsRecord{ptr, size, rawCuDesc, ipcMode});
}

void NcclTms::copyToHostAndReleaseA() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    // copy to host
    WARN("NcclTms::copyToHostAndReleaseA stage copy");
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ipcMode == NcclTmsIpcMode::EXPORTER) {
            if (records_[i].cpuBackup == nullptr) {
                CUDACHECKEXIT(cudaMallocHost(&records_[i].cpuBackup, records_[i].size));
            }
            CUDACHECKEXIT(cudaMemcpyAsync(records_[i].cpuBackup, records_[i].ptr, records_[i].size, cudaMemcpyDeviceToHost));
        }
    }

    // TODO improve all code, e.g. the `[i]
    WARN("NcclTms::copyToHostAndReleaseA stage release");
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ipcMode == NcclTmsIpcMode::IMPORTER) {
            CUmemAllocationProp prop = getCUmemAllocationProp();
            size_t size = alignSizeByGranularity(records_[i].size, prop);

            CUCHECKEXIT(cuMemUnmap((CUdeviceptr)records_[i].ptr, size));
        }
    }
}

void NcclTms::copyToHostAndReleaseB() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    WARN("NcclTms::copyToHostAndReleaseA stage release");
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ipcMode == NcclTmsIpcMode::EXPORTER) {
            CUmemAllocationProp prop = getCUmemAllocationProp();
            size_t alignedSize = alignSizeByGranularity(records_[i].size, prop);

            CUmemGenericAllocationHandle handle;
            CUCHECKEXIT(cuMemRetainAllocationHandle(&handle, records_[i].ptr));

            CUCHECKEXIT(cuMemUnmap((CUdeviceptr)records_[i].ptr, alignedSize));
            CUCHECKEXIT(cuMemRelease(handle));
        }
    }
}

const char* ipcModeToString(NcclTmsIpcMode ipc_mode) {
  switch (ipc_mode) {
    case NcclTmsIpcMode::EXPORTER: return "EXPORTER";
    case NcclTmsIpcMode::IMPORTER: return "IMPORTER";
    default: exit(1);
  }
}

char* NcclTms::getRecords() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    nlohmann::json output_json = nlohmann::json::array();

    for (size_t i = 0; i < records_.size(); ++i) {
        output_json.push_back({
            {"i", i},
            {"initialRawCuDesc", records_[i].initialRawCuDesc},
            {"ipcMode", ipcModeToString(records_[i].ipcMode)},
        });
    }

    std::string message = output_json.dump();
    char* result = new char[message.size() + 1];
    std::strcpy(result, message.c_str());
    return result;
}

char* NcclTms::resumeAndCopyToDeviceA(const char* input_str) {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    nlohmann::json input_json = nlohmann::json::parse(input_str);
    nlohmann::json output_json = nlohmann::json::array();

    WARN("NcclTms::resumeAndCopyToDeviceA stage resume");
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ipcMode == NcclTmsIpcMode::EXPORTER) {
            // ref: ncclP2pAllocateShareableBuffer,

            // ref: ncclCuMemAlloc
            CUmemGenericAllocationHandle handle;
            {
                CUmemAllocationProp prop = getCUmemAllocationProp();
                size_t alignedSize = alignSizeByGranularity(records_[i].size, prop);
                /* Allocate the physical memory on the device */
                CUCHECKEXIT(cuMemCreate(&handle, alignedSize, &prop, 0));
                CUCHECKEXIT(cuMemMap((CUdeviceptr)records_[i].ptr, alignedSize, 0, handle, 0));
            }

            // ref: proxyGetFd
            int fd_repeat_num = input_json[i]["fd_repeat_num"];
            std::vector<int> fd_arr;
            for (int fd_repeat_index = 0; fd_repeat_index < fd_repeat_num; ++fd_repeat_index) {
                int fd = -1;
                CUmemAllocationHandleType type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
                CUCHECKEXIT(cuMemExportToShareableHandle(&fd, handle, type, 0));
                fd_arr.push_back(fd);
            }
            output_json.push_back({{"fd_arr", fd_arr}});
        } else {
            output_json.push_back({});
        }
    }

    std::string message = output_json.dump();
    char* result = new char[message.size() + 1];
    std::strcpy(result, message.c_str());
    return result;
}

void NcclTms::resumeAndCopyToDeviceB(const char* input_str) {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    nlohmann::json input_json = nlohmann::json::parse(input_str);

    WARN("NcclTms::resumeAndCopyToDeviceB stage resume");
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ipcMode == NcclTmsIpcMode::IMPORTER) {
            CUmemAllocationProp prop = getCUmemAllocationProp();
            size_t alignedSize = alignSizeByGranularity(records_[i].size, prop);

            // ref: ncclP2pImportShareableBuffer
            int fd = input_json[i]["fd"];
            WARN("NcclTms::resumeAndCopyToDeviceB cuMemMap i=%d fd=%d", (int) i, fd);

            CUmemAllocationHandleType type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
            CUmemGenericAllocationHandle handle;
            CUCHECKEXIT(cuMemImportFromShareableHandle(&handle, (void *)(uintptr_t)fd, type));
            (void) close(fd);

            CUCHECKEXIT(cuMemMap((CUdeviceptr)records_[i].ptr, alignedSize, /* offset */ 0, handle, /* flags */ 0));
        }
    }

    // copy to device
    WARN("NcclTms::resumeAndCopyToDeviceB stage copy");
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ipcMode == NcclTmsIpcMode::EXPORTER) {
            CUDACHECKEXIT(cudaMemcpyAsync(records_[i].ptr, records_[i].cpuBackup, records_[i].size, cudaMemcpyHostToDevice));
            // TODO free host memory later
        }
    }
}

extern "C" {

__attribute__((visibility("default"))) void nccl_tms_copyToHostAndReleaseA() { NcclTms::instance().copyToHostAndReleaseA(); }
__attribute__((visibility("default"))) void nccl_tms_copyToHostAndReleaseB() { NcclTms::instance().copyToHostAndReleaseB(); }
__attribute__((visibility("default"))) char* nccl_tms_getRecords() { return NcclTms::instance().getRecords(); }
__attribute__((visibility("default"))) char* nccl_tms_resumeAndCopyToDeviceA(const char* input_str) { return NcclTms::instance().resumeAndCopyToDeviceA(input_str); }
__attribute__((visibility("default"))) void nccl_tms_resumeAndCopyToDeviceB(const char* input_str) { NcclTms::instance().resumeAndCopyToDeviceB(input_str); }

//__attribute__((visibility("default"))) void nccl_tms_freeDynamicString(char* ptr) {
//    WARN("nccl_tms_freeDynamicString START ptr=%p", ptr);
//    delete[] ptr;
//    WARN("nccl_tms_freeDynamicString END");
//}

}
