#include "tms.h"

NcclTms::NcclTms() {}

// 静态单例方法实现
NcclTms& NcclTms::instance() {
    static NcclTms instance;
    return instance;
}

void NcclTms::registerAlloc(void* ptr, size_t size, uint64_t rawCuDesc, NcclTmsIpcMode ipcMode) {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    records_.push_back(NcclTmsRecord{ptr, size, rawCuDesc, ipcMode});
}

void NcclTms::copyToHostAndReleaseA() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    // copy to host
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ipcMode == NcclTmsIpcMode::EXPORTER) {
            if (records_[i].cpuBackup == nullptr) {
                CUCHECK(cudaMallocHost(&records_[i].cpuBackup, records_[i].size));
            }
            CUCHECK(cudaMemcpyAsync(records_[i].cpuBackup, ptr, records_[i].size, cudaMemcpyDeviceToHost));
        }
    }

    // TODO improve all code, e.g. the `[i]
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ipcMode == NcclTmsIpcMode::IMPORTER) {
            CUCHECK(cuMemUnmap(records_[i].ptr, records_[i].size));
        }
    }
}

void NcclTms::copyToHostAndReleaseB() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ipcMode == NcclTmsIpcMode::EXPORTER) {
            CUmemGenericAllocationHandle handle;
            CUCHECK(cuMemRetainAllocationHandle(&handle, records_[i].ptr));

            CUCHECK(cuMemUnmap(records_[i].ptr, records_[i].size));
            CUCHECK(cuMemRelease(handle));
        }
    }
}

void NcclTms::resumeAndCopyToDeviceA() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ipcMode == NcclTmsIpcMode::EXPORTER) {
            // ref: ncclP2pAllocateShareableBuffer,

            // ref: ncclCuMemAlloc
            CUmemGenericAllocationHandle handle;
            {
                size_t granularity = 0;
                CUdevice currentDev;
                CUmemAllocationProp prop = {};
                // CUmemAccessDesc accessDesc = {};
                CUmemAllocationHandleType type = ncclCuMemHandleType;
                int cudaDev;
                int flag = 0;
                CUDACHECK(cudaGetDevice(&cudaDev));
                CUCHECK(cuDeviceGet(&currentDev, cudaDev));
                prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                prop.requestedHandleTypes = type;
                prop.location.id = currentDev;
                // Query device to see if RDMA support is available
                CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, currentDev));
                if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;
                CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
                ALIGN_SIZE(size, granularity);
                /* Allocate the physical memory on the device */
                CUCHECK(cuMemCreate(&handle, size, &prop, 0));
                // /* Reserve a virtual address range */
                // CUCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, size, granularity, 0, 0));
                /* Map the virtual address range to the physical allocation */
                CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
                // /* Now allow RW access to the newly mapped memory */
                // accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                // accessDesc.location.id = currentDev;
                // accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                // CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
            }

            // ref: proxyGetFd
            {
                CUmemAllocationHandleType type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
                int fd = -1;
                CUCHECK(cuMemExportToShareableHandle(&fd, handle, type, 0));
            }
        }
    }
}

void NcclTms::resumeAndCopyToDeviceB() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ipcMode == NcclTmsIpcMode::IMPORTER) {
            TODO;
        }
    }

    // copy to device
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].ipcMode == NcclTmsIpcMode::EXPORTER) {
            CUCHECK(cudaMemcpyAsync(records_[i].ptr, records_[i].cpuBackup, records_[i].size, cudaMemcpyHostToDevice));
            // TODO free host memory later
        }
    }
}

extern "C" {

void nccl_tms_copyToHostAndReleaseA() { NcclTms::instance().copyToHostAndReleaseA(); }
void nccl_tms_copyToHostBndReleaseB() { NcclTms::instance().copyToHostBndReleaseB(); }
void nccl_tms_resumeAndCopyToDeviceA() { NcclTms::instance().resumeAndCopyToDeviceA(); }
void nccl_tms_resumeBndCopyToDeviceB() { NcclTms::instance().resumeAndCopyToDeviceB(); }

}
