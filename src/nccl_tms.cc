#include "comm.h"
#include "utils.h"
#include "nccl_tms.h"
#include "json.hpp"
#include <sys/syscall.h>
#include <unistd.h>

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

CUdevice getCurrentDev() {
    CUdevice currentDev;
    int cudaDev;
    CUDACHECKEXIT(cudaGetDevice(&cudaDev));
    CUCHECKEXIT(cuDeviceGet(&currentDev, cudaDev));
    return currentDev;
}

// ref: ncclCuMemAlloc
CUmemAllocationProp getCUmemAllocationProp() {
    CUdevice currentDev = getCurrentDev();
    CUmemAllocationProp prop = {};
    CUmemAllocationHandleType type = ncclCuMemHandleType;
    int flag = 0;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.requestedHandleTypes = type;
    prop.location.id = currentDev;
    // Query device to see if RDMA support is available
    CUCHECKEXIT(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, currentDev));
    if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;
    return prop;
}

// ref: ncclCuMemAlloc
void wrappedCuMemSetAccess(void* ptr, size_t size) {
    CUdevice currentDev = getCurrentDev();
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = currentDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECKEXIT(cuMemSetAccess((CUdeviceptr)ptr, size, &accessDesc, 1));
}

// ref: ncclCuMemAlloc
size_t alignSizeByGranularity(size_t size, CUmemAllocationProp prop) {
    size_t granularity = 0;
    CUCHECKEXIT(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    ALIGN_SIZE(size, granularity);
    return size;
}

static thread_local bool nccl_tms_enable_ = true;

NcclTms::NcclTms() {}

// 静态单例方法实现
NcclTms& NcclTms::instance() {
    static NcclTms instance;
    return instance;
}

void NcclTms::setThreadLocalEnable(bool enable) {
    nccl_tms_enable_ = enable;
}

void NcclTms::registerAlloc(void* ptr, size_t size, uint64_t rawCuDesc, CUmemGenericAllocationHandle handle, NcclTmsIpcMode ipcMode) {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    WARN("NcclTms::registerAlloc enable=%d ptr=%p, size=%zu, rawCuDesc=%lu, ipcMode=%d", (int) nccl_tms_enable_, ptr, size, rawCuDesc, static_cast<int>(ipcMode));
    if (nccl_tms_enable_) {
        records_.push_back(NcclTmsRecord{ptr, size, rawCuDesc, handle, ipcMode});
    }
}

void NcclTms::registerDealloc(void* ptr) {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    WARN("NcclTms::registerDealloc ptr=%p", ptr);
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ptr == ptr) {
            WARN("NcclTms::registerDealloc find i=%d", (int) i);
            records_[i].deallocated = true;
        }
    }
}

void NcclTms::copyToHostAndReleaseA() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    // copy to host
    WARN("NcclTms::copyToHostAndReleaseA stage copy");
    for (size_t i = 0; i < records_.size(); ++i) {
        if (
            (records_[i].ipcMode == NcclTmsIpcMode::EXPORTER) ||
            (records_[i].ipcMode == NcclTmsIpcMode::LOCAL)
        ) {
            if (records_[i].cpuBackup == nullptr) {
                CUDACHECKEXIT(cudaMallocHost(&records_[i].cpuBackup, records_[i].size));
            }
            CUDACHECKEXIT(cudaMemcpyAsync(records_[i].cpuBackup, records_[i].ptr, records_[i].size, cudaMemcpyDeviceToHost));
        }
    }

    // TODO improve all code, e.g. the `[i]
    WARN("NcclTms::copyToHostAndReleaseA stage release");
    int importerSizeSum = 0;
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ipcMode == NcclTmsIpcMode::IMPORTER) {
            CUmemAllocationProp prop = getCUmemAllocationProp();
            size_t alignedSize = alignSizeByGranularity(records_[i].size, prop);

            CUCHECKEXIT(cuMemUnmap((CUdeviceptr)records_[i].ptr, alignedSize));

            WARN("NcclTms::copyToHostAndReleaseA hack also release IMPORTER physical memory");
            CUCHECKEXIT(cuMemRelease(records_[i].initialHandle));

            importerSizeSum += alignedSize;
        }
    }
    WARN("NcclTms::copyToHostAndReleaseA importerSizeSum=%d", importerSizeSum);
}

void NcclTms::copyToHostAndReleaseB() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    WARN("NcclTms::copyToHostAndReleaseA stage release");
    int exporterSizeSum = 0;
    for (size_t i = 0; i < records_.size(); ++i) {
        if (
            (records_[i].ipcMode == NcclTmsIpcMode::EXPORTER) ||
            (records_[i].ipcMode == NcclTmsIpcMode::LOCAL)
        ) {
            CUmemAllocationProp prop = getCUmemAllocationProp();
            size_t alignedSize = alignSizeByGranularity(records_[i].size, prop);

            CUCHECKEXIT(cuMemUnmap((CUdeviceptr)records_[i].ptr, alignedSize));
            CUCHECKEXIT(cuMemRelease(records_[i].initialHandle));
            exporterSizeSum += alignedSize;
        }
    }
    WARN("NcclTms::copyToHostAndReleaseB exporterSizeSum=%d", exporterSizeSum);
}

const char* ipcModeToString(NcclTmsIpcMode ipc_mode) {
  switch (ipc_mode) {
    case NcclTmsIpcMode::EXPORTER: return "EXPORTER";
    case NcclTmsIpcMode::IMPORTER: return "IMPORTER";
    case NcclTmsIpcMode::LOCAL: return "LOCAL";
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
        if (
            (records_[i].ipcMode == NcclTmsIpcMode::EXPORTER) ||
            (records_[i].ipcMode == NcclTmsIpcMode::LOCAL)
        ) {
            // ref: ncclP2pAllocateShareableBuffer,

            // ref: ncclCuMemAlloc
            CUmemGenericAllocationHandle handle;
            {
                CUmemAllocationProp prop = getCUmemAllocationProp();
                size_t alignedSize = alignSizeByGranularity(records_[i].size, prop);
                /* Allocate the physical memory on the device */
                CUCHECKEXIT(cuMemCreate(&handle, alignedSize, &prop, 0));
                CUCHECKEXIT(cuMemMap((CUdeviceptr)records_[i].ptr, alignedSize, 0, handle, 0));
                wrappedCuMemSetAccess(records_[i].ptr, alignedSize);
            }

            // ref: proxyGetFd
            if (records_[i].ipcMode == NcclTmsIpcMode::EXPORTER) {
                int fd_repeat_num = input_json[i]["fd_repeat_num"];
                std::vector<int> fd_arr;
                for (int fd_repeat_index = 0; fd_repeat_index < fd_repeat_num; ++fd_repeat_index) {
                    int fd = -1;
                    CUmemAllocationHandleType type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
                    // TODO check whether need to close this fd at sender side
                    CUCHECKEXIT(cuMemExportToShareableHandle(&fd, handle, type, 0));
                    fd_arr.push_back(fd);
                }
                output_json.push_back({{"fd_arr", fd_arr}});
            } else {
                output_json.push_back({});
            }
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

            int fdInSenderProcess = input_json[i]["fd_in_sender_process"];
            int senderPid = input_json[i]["sender_pid"];

            // https://stackoverflow.com/questions/2358684/can-i-share-a-file-descriptor-to-another-process-on-linux-or-are-they-local-to-t
            int senderPidFd = syscall(SYS_pidfd_open, senderPid, 0);
            int fdInLocalProcess = syscall(SYS_pidfd_getfd, senderPidFd, fdInSenderProcess, 0);
            WARN("NcclTms::resumeAndCopyToDeviceB cuMemMap i=%d fdInSenderProcess=%d fdInLocalProcess=%d", (int) i, fdInSenderProcess, fdInLocalProcess);

            CUmemAllocationHandleType type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
            CUmemGenericAllocationHandle handle;
            CUCHECKEXIT(cuMemImportFromShareableHandle(&handle, (void *)(uintptr_t)fdInLocalProcess, type));

            (void) close(fdInLocalProcess);
            (void) close(senderPidFd); // can optimize (not open-close every time)

            CUCHECKEXIT(cuMemMap((CUdeviceptr)records_[i].ptr, alignedSize, /* offset */ 0, handle, /* flags */ 0));
            wrappedCuMemSetAccess(records_[i].ptr, alignedSize);
        }
    }

    // copy to device
    WARN("NcclTms::resumeAndCopyToDeviceB stage copy");
    for (size_t i = 0; i < records_.size(); ++i) {
        if (
            (records_[i].ipcMode == NcclTmsIpcMode::EXPORTER) ||
            (records_[i].ipcMode == NcclTmsIpcMode::LOCAL)
        ) {
            WARN("NcclTms::resumeAndCopyToDeviceB cudaMemcpy i=%d ptr=%p cpuBackup=%p size=%d",
                (int) i, records_[i].ptr, records_[i].cpuBackup, (int) records_[i].size);
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
