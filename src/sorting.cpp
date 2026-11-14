#include "sorting.h"

#include <algorithm>
#include <limits>

namespace df {

const char* sortModeName(SortMode mode) {
    switch (mode) {
    case SortMode::Disabled: return "disabled";
    case SortMode::Exact: return "exact";
    case SortMode::BucketApprox: return "bucket_approx";
    }
    return "unknown";
}

SortResult sortTransparent(const Scene& scene, const TransparentConfig& config,
                           const Camera& camera) {
    SortResult result;
    result.objects.reserve(scene.transparent.size());
    for (const Primitive& primitive : scene.transparent) {
        const float depth = viewDepth(camera, primitive.center);
        if (depth <= camera.nearPlane) continue;
        result.objects.push_back(
            SortedTransparent{primitive.id, depth, primitive.opacity, primitive.center});
    }

    if (!config.sortingEnabled) {
        result.mode = SortMode::Disabled;
    } else if (static_cast<int>(result.objects.size()) > config.exactSortLimit) {
        result.mode = SortMode::BucketApprox;
        result.approximate = true;

        float minDepth = std::numeric_limits<float>::max();
        float maxDepth = -std::numeric_limits<float>::max();
        for (const auto& object : result.objects) {
            minDepth = std::min(minDepth, object.depth);
            maxDepth = std::max(maxDepth, object.depth);
        }
        const float range = std::max(maxDepth - minDepth, 1.0e-4f);
        const int bucketCount = config.bucketCount;
        std::vector<int> buckets(result.objects.size(), 0);
        std::vector<std::size_t> counts(static_cast<std::size_t>(bucketCount), 0);
        for (std::size_t i = 0; i < result.objects.size(); ++i) {
            const float normalized = (maxDepth - result.objects[i].depth) / range;
            buckets[i] = std::min(bucketCount - 1,
                                  std::max(0, static_cast<int>(normalized * bucketCount)));
            ++counts[static_cast<std::size_t>(buckets[i])];
        }
        std::vector<std::size_t> offsets(static_cast<std::size_t>(bucketCount), 0);
        for (int i = 1; i < bucketCount; ++i) {
            offsets[static_cast<std::size_t>(i)] =
                offsets[static_cast<std::size_t>(i - 1)] +
                counts[static_cast<std::size_t>(i - 1)];
        }
        std::vector<SortedTransparent> sorted;
        sorted.reserve(result.objects.size());
        sorted.resize(result.objects.size());
        for (std::size_t i = 0; i < result.objects.size(); ++i) {
            sorted[offsets[static_cast<std::size_t>(buckets[i])]++] = result.objects[i];
        }
        result.objects = std::move(sorted);
        result.comparisons = static_cast<std::int64_t>(result.objects.size() + bucketCount);
    } else {
        result.mode = SortMode::Exact;
        std::stable_sort(result.objects.begin(), result.objects.end(),
                         [](const SortedTransparent& a, const SortedTransparent& b) {
                             if (a.depth > b.depth) return true;
                             if (a.depth < b.depth) return false;
                             return a.id < b.id;
                         });
        result.comparisons = static_cast<std::int64_t>(
            result.objects.size() * (result.objects.size() > 1 ? 1 : 0));
    }
    return result;
}

} // namespace df

