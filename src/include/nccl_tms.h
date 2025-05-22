#ifndef NCCL_TMS_H_
#define NCCL_TMS_H_

#include <mutex>
#include <vector>

enum NcclTmsIpcMode {
    EXPORTER,
    IMPORTER,
};

struct NcclTmsRecord {
    void* ptr;
    size_t size;
    uint64_t initialRawCuDesc;
    NcclTmsIpcMode ipcMode;
    void* cpuBackup;
};

class NcclTms {
public:
    NcclTms();
    static NcclTms &instance();
    void registerAlloc(void* ptr, size_t size, uint64_t rawCuDesc, NcclTmsIpcMode ipcMode);
    void copyToHostAndReleaseA();
    void copyToHostAndReleaseB();
    char* NcclTms::getRecords();
    char* NcclTms::resumeAndCopyToDeviceA(const char* input_str);
    void NcclTms::resumeAndCopyToDeviceB(const char* input_str);
}

private:
    // TODO improve
    std::mutex primary_mutex_;
    std::vector<NcclTmsRecord> records_;
};

#endif
