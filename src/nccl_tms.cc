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
            // ref: ncclP2pAllocateShareableBuffer
            TODO;
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
