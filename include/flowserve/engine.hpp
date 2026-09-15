#pragma once

#include "flowserve/scheduler.hpp"
#include "flowserve/types.hpp"
#include <vector>

namespace flowserve {

struct WorkloadItem {
    std::vector<int> prompt;
    int max_new_tokens{16};
    uint64_t arrival_tick{0};
};

class Engine {
public:
    explicit Engine(ServingConfig cfg);

    RunReport run(const std::vector<WorkloadItem>& items);

private:
    ServingConfig cfg_;
};

} // namespace flowserve
