#pragma once

#include "automix/stereo.h"

#include <cstdint>

namespace automix {

struct LoudnessResult {
    double integratedLufs = 0.0;
    double relativeGateLufs = 0.0;
    double truePeakDbtp = 0.0;
    double truePeakLinear = 0.0;
    std::uint64_t totalBlocks = 0;
    std::uint64_t gatedBlocks = 0;
    bool hasIntegratedValue = false;
};

LoudnessResult measureBs1770(const StereoBuffer& buffer);

}  // namespace automix
