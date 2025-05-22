#include "tms.h"

struct NcclTmsRecord {

};

class NcclTmsImpl {
public:
    NcclTmsImpl() {}

    static NcclTmsImpl &instance() {
        static NcclTmsImpl instance;
        return instance;
    }

private:
    std::mutex allocator_metadata_mutex_;
    std::vector<NcclTmsRecord> records_;
};

void ncclTmsRegister(void* ptr, size_t size, uint64_t rawIpcDesc, NcclTmsIpcMode ipcMode) {
    TODO;
}
