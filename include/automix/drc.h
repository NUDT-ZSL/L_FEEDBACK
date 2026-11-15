#pragma once

#include "automix/config.h"
#include "automix/stereo.h"

#include <cstddef>
#include <vector>

namespace automix {

struct DrcStats {
    double maxReductionDb = 0.0;
    double meanReductionDb = 0.0;
    std::uint64_t activeSamples = 0;
};

double drcStaticReductionDb(double levelDbfs, const DrcConfig& config);
DrcStats processDrc(StereoBuffer& buffer,
                    const DrcConfig& config,
                    std::vector<double>* gainReductionDb = nullptr);

}  // namespace automix
