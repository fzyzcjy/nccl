#include "tms.h"

NcclTms::NcclTms() {}

// 静态单例方法实现
NcclTms& NcclTms::instance() {
    static NcclTms instance;
    return instance;
}

void NcclTms::registerAlloc(void* ptr, size_t size, uint64_t rawIpcDesc, NcclTmsIpcMode ipcMode) {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    records_.push_back(NcclTmsRecord{ptr, size, rawIpcDesc, ipcMode});
}

void NcclTms::copyToHostAndReleaseA() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    // copy to host
    TODO

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
            // ref: ncclP2pAllocateShareableBuffer
            TODO;
        }
    }

    // copy to device
    TODO;
}
