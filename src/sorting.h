#pragma once

#include "config.h"
#include "scene.h"

#include <string>
#include <vector>

namespace df {

enum class SortMode {
    Disabled,
    Exact,
    BucketApprox,
};

struct SortedTransparent {
    int id = 0;
    float depth = 0.0f;
    float opacity = 1.0f;
    Vec3 center{};
};

struct SortResult {
    SortMode mode = SortMode::Exact;
    bool approximate = false;
    std::vector<SortedTransparent> objects;
    std::int64_t comparisons = 0;
};

const char* sortModeName(SortMode mode);
SortResult sortTransparent(const Scene& scene, const TransparentConfig& config,
                           const Camera& camera);

} // namespace df

