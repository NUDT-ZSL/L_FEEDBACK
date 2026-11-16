#pragma once

#include "core/bvh.hpp"
#include "core/scene.hpp"

#include <cstdint>
#include <vector>

namespace uid166 {

struct DepthHierarchy {
    static constexpr uint32_t kWidth = 160;
    static constexpr uint32_t kHeight = 90;

    void clear();
    void rasterizeAABB(const Mat4& viewProj, const AABB& bounds, uint32_t ownerId);
    void buildPyramid();
    [[nodiscard]] bool mostlyOccluded(const Mat4& viewProj, const AABB& bounds,
                                      uint32_t ownerId, float depthBias,
                                      uint32_t expandPixels) const;

    [[nodiscard]] float maxDepth() const;
    [[nodiscard]] bool valid() const { return valid_; }

private:
    struct Level {
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<float> values;
    };

    std::vector<Level> levels_;
    std::vector<uint32_t> owners_;
    bool valid_ = false;
};

struct CullOptions {
    bool enableOcclusion = true;
    float fastCameraDistance = 1.25f;
    float fastCameraAngleDegrees = 0.5f;
    float depthBias = 0.0015f;
    float pixelsPerMeter = 4.0f;
    uint32_t maxExpandPixels = 28;
};

struct CullResult {
    std::vector<uint32_t> frustumVisible;
    std::vector<uint32_t> visible;
    std::vector<uint32_t> occluded;
    uint32_t totalObjects = 0;
    uint32_t frustumRejected = 0;
    uint32_t occlusionCandidates = 0;
    uint32_t delayedQueryResults = 0;
    uint32_t hierarchyTests = 0;
    bool cameraFast = false;
    BvhQueryStats bvhStats{};
};

class OcclusionCuller {
public:
    void reset(size_t objectCount);
    [[nodiscard]] CullResult cull(const std::vector<SceneObject>& objects, const Bvh& bvh,
                                  const Camera& camera, const Camera& previousCamera,
                                  const CullOptions& options);
    [[nodiscard]] const DepthHierarchy& previousDepth() const { return previousDepth_; }
    [[nodiscard]] const std::vector<uint8_t>& previousQueryResults() const {
        return previousQueries_;
    }

private:
    DepthHierarchy previousDepth_;
    Mat4 previousViewProjection_ = Mat4::identity();
    std::vector<uint8_t> previousQueries_;
    std::vector<uint8_t> currentQueries_;
    std::vector<uint32_t> queryCandidates_;
    bool hasPreviousFrame_ = false;
};

[[nodiscard]] bool cameraIsFast(const Camera& current, const Camera& previous,
                                const CullOptions& options);

} // namespace uid166

