#pragma once

#include "automix/config.h"
#include "automix/stereo.h"

#include <cstdint>

namespace automix {

struct LimiterStats {
    std::uint64_t lookaheadSamples = 0;
    std::uint64_t triggeredSamples = 0;
    std::uint64_t affectedSamples = 0;
    double maxAttenuationDb = 0.0;
    double outputPeakDbfs = 0.0;
};

LimiterStats processLimiter(StereoBuffer& buffer, const LimiterConfig& config);

}  // namespace automix
