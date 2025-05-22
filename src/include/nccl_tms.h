#ifndef NCCL_TMS_H_
#define NCCL_TMS_H_

enum NcclTmsIpcMode {
    EXPORT,
    IMPORT,
};

void ncclTmsRegister(void* ptr, size_t size, uint64_t rawIpcDesc, NcclTmsIpcMode ipcMode);

#endif
