#ifndef NCCL_TMS_H_
#define NCCL_TMS_H_

#include <mutex>
#include <vector>

enum NcclTmsIpcMode {
    EXPORTER,
    IMPORTER,
    LOCAL,
};

struct NcclTmsRecord {
    void* ptr;
    size_t size;
    uint64_t initialRawCuDesc;
    CUmemGenericAllocationHandle initialHandle;
    NcclTmsIpcMode ipcMode;
    void* cpuBackup;
};

class NcclTms {
public:
    NcclTms();
    static NcclTms &instance();
    void registerAlloc(void* ptr, size_t size, uint64_t rawCuDesc, CUmemGenericAllocationHandle handle, NcclTmsIpcMode ipcMode);
    void registerDealloc(void* ptr);
    void copyToHostAndReleaseA();
    void copyToHostAndReleaseB();
    char* getRecords();
    char* resumeAndCopyToDeviceA(const char* input_str);
    void resumeAndCopyToDeviceB(const char* input_str);
    void setThreadLocalEnable(bool enable);

private:
    // TODO improve
    std::mutex primary_mutex_;
    std::vector<NcclTmsRecord> records_;
};

#endif
