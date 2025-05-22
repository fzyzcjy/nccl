#include "tms.h"

NcclTmsImpl::NcclTmsImpl() {}

// 静态单例方法实现
NcclTmsImpl& NcclTmsImpl::instance() {
    static NcclTmsImpl instance;
    return instance;
}

void NcclTmsImpl::registerAlloc(void* ptr, size_t size, uint64_t rawIpcDesc, NcclTmsIpcMode ipcMode) {
    TODO;
}
