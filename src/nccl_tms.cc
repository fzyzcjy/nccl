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

    // release
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
            CUCHECK(cuMemUnmap(records_[i].ptr, records_[i].size));
            CUCHECK(cuMemRelease(TODO));
        }
    }
}

void NcclTms::resumeAndCopyToDeviceA() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    TODO;
}

void NcclTms::resumeAndCopyToDeviceB() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    TODO;
}
