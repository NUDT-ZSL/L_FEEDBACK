#pragma once

#include "renderer.hpp"
#include "types.hpp"

#include <cstdint>
#include <vector>

namespace mv {

struct ValidationSummary {
    std::vector<RuleResult> rules;
    std::uint64_t pixelCount = 0;
    std::uint64_t passedPixels = 0;
    std::uint64_t failedPixels = 0;
};

ValidationSummary evaluateRules(const MaterialSpec& material,
                                const std::vector<RawAttachment>& attachments,
                                int maxDifferences);

}  // namespace mv
