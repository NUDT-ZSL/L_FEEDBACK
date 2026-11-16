#pragma once

#include "core/bvh.hpp"
#include "core/math.hpp"
#include "core/scene.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace uid166 {

struct CascadeConfig {
    uint32_t count = 3;
    uint32_t shadowSize = 2048;
    uint32_t stabilizationSize = 2048;
    float splitLambda = 0.75f;
    float casterDepthPadding = 45.0f;
};

struct Cascade {
    uint32_t index = 0;
    float splitNear = 0.0f;
    float splitFar = 0.0f;
    Mat4 matrix = Mat4::identity();
    Frustum frustum{};
    uint32_t visibleCasters = 0;
    bool updated = false;
};

struct CsmFrame {
    std::vector<Cascade> cascades;
    uint32_t updatedCascades = 0;
    bool matricesStable = false;
    std::array<float, 4> splitDistances{0.0f, 0.0f, 0.0f, 0.0f};
};

[[nodiscard]] std::vector<Cascade> computeCascades(const Camera& camera,
                                                   const Light& directionalLight,
                                                   const CascadeConfig& config);
[[nodiscard]] std::vector<float> cascadeSplits(float nearZ, float farZ, uint32_t count,
                                               float lambda);
[[nodiscard]] uint32_t selectCascade(const CsmFrame& frame, float viewDepth);

class CsmUpdatePolicy {
public:
    [[nodiscard]] CsmFrame update(const Camera& camera, const Light& light,
                                  const Bvh& bvh, const CascadeConfig& config,
                                  bool animationDirty, bool staticSceneDirty);
    void invalidate();

private:
    bool valid_ = false;
    Mat4 previousView_ = Mat4::identity();
    Mat4 previousProjection_ = Mat4::identity();
    Vec3 previousLightDirection_{};
    uint32_t previousShadowSize_ = 0;
    std::vector<Mat4> previousMatrices_;
};

enum ShadowUpdateSource : uint32_t {
    shadowSourceNone = 0,
    shadowSourceInvalid = 1u << 0,
    shadowSourceTransform = 1u << 1,
    shadowSourceVisibility = 1u << 2,
    shadowSourceAnimation = 1u << 3,
    shadowSourceResolution = 1u << 4,
    shadowSourceStaticScene = 1u << 5,
    shadowSourceDeferredBudget = 1u << 6,
};

struct LocalShadowConfig {
    uint32_t maxUpdatesPerFrame = 2;
    float transformEpsilon = 0.01f;
    float faceResolution = 1024.0f;
};

struct ShadowMatrixInfo {
    std::string face;
    Mat4 matrix = Mat4::identity();
};

struct LocalShadowDecision {
    uint32_t lightId = 0;
    LightType type = LightType::point;
    bool visible = false;
    bool dirty = false;
    bool updated = false;
    bool deferredByBudget = false;
    uint32_t sourceMask = shadowSourceNone;
    std::vector<ShadowMatrixInfo> matrices;
};

struct LocalShadowFrame {
    std::vector<LocalShadowDecision> lights;
    uint32_t updatedCount = 0;
    uint32_t visibleCount = 0;
    uint32_t dirtyCount = 0;
};

class LocalShadowPolicy {
public:
    void reset(size_t lightCount);
    [[nodiscard]] LocalShadowFrame update(const Camera& camera, const std::vector<Light>& lights,
                                          const std::vector<uint32_t>& visibleObjectIds,
                                          const std::vector<SceneObject>& objects,
                                          const AABB& characterBounds, bool characterVisible,
                                          bool characterAnimationDirty,
                                          bool staticSceneDirty, const LocalShadowConfig& config);

private:
    struct State {
        bool initialized = false;
        Vec3 position{};
        Vec3 direction{};
        uint32_t resolution = 0;
        uint64_t lastUpdatedFrame = 0;
    };
    std::vector<State> states_;
    uint64_t frame_ = 0;
};

[[nodiscard]] std::vector<ShadowMatrixInfo> localShadowMatrices(const Light& light,
                                                                uint32_t resolution);
[[nodiscard]] std::string shadowSourceNames(uint32_t sourceMask);
[[nodiscard]] bool matrixExactlyEqual(const Mat4& a, const Mat4& b);

} // namespace uid166

