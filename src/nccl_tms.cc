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

    TODO;
}

void NcclTms::copyToHostAndReleaseB() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    TODO;
}

void NcclTms::resumeAndCopyToDeviceA() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    TODO;
}

void NcclTms::resumeAndCopyToDeviceB() {
    const std::lock_guard<std::mutex> lock(primary_mutex_);

    TODO;
}
