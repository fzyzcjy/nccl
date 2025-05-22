#ifndef NCCL_TMS_H_
#define NCCL_TMS_H_

#include <mutex>
#include <vector>

enum NcclTmsIpcMode {
    EXPORT,
    IMPORT,
};

struct NcclTmsRecord {

};

class NcclTms {
public:
    NcclTms();
    static NcclTms &instance();
    void registerAlloc(void* ptr, size_t size, uint64_t rawIpcDesc, NcclTmsIpcMode ipcMode);

private:
    std::mutex allocator_metadata_mutex_;
    std::vector<NcclTmsRecord> records_;
};

#endif
