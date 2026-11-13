#pragma once

#include "skinbake/snapshot.hpp"
#include "skinbake/state_machine.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace skinbake {

struct VerificationReport {
    bool passed = false;
    std::uint32_t frames = 0;
    double tolerance = 1e-5;
    double maxAbsError = 0.0;
    std::uint32_t maxErrorFrame = 0;
    bool byteIdentical = true;
    std::vector<std::uint32_t> mismatchFrames;
};

VerificationReport verify_snapshot(const TimelineEvaluator& timeline,
                                   const std::string& path,
                                   double tolerance = 1e-5,
                                   std::optional<std::uint32_t> expectedVersion =
                                       std::nullopt);

}

