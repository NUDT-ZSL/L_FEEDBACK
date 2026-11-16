#include "core/shadows.hpp"

#include "core/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

namespace uid166 {

namespace {

Vec3 averageCorners(const std::array<Vec3, 8>& corners) {
    Vec3 center{};
    for (const Vec3& c : corners) center += c;
    return center * (1.0f / 8.0f);
}

Mat4 orientationOnly(const Vec3& direction) {
    const Vec3 dir = normalize(direction);
    Vec3 up = std::abs(dir.y) > 0.95f ? Vec3{1.0f, 0.0f, 0.0f} : Vec3{0.0f, 1.0f, 0.0f};
    return lookAtLH({0.0f, 0.0f, 0.0f}, dir, up);
}

bool aabbIntersects(const AABB& a, const AABB& b) {
    return !(a.mx.x < b.mn.x || a.mn.x > b.mx.x || a.mx.y < b.mn.y ||
             a.mn.y > b.mx.y || a.mx.z < b.mn.z || a.mn.z > b.mx.z);
}

bool lightOverlapsRenderable(const Light& light, const AABB& bounds) {
    if (light.type == LightType::directional) return true;
    if (!aabbIntersects(lightInfluenceBounds(light), bounds)) return false;
    if (light.type == LightType::point) {
        const Vec3 closest{
            std::clamp(light.position.x, bounds.mn.x, bounds.mx.x),
            std::clamp(light.position.y, bounds.mn.y, bounds.mx.y),
            std::clamp(light.position.z, bounds.mn.z, bounds.mx.z),
        };
        return length(closest - light.position) <= light.range;
    }
    const Vec3 toCenter = bounds.center() - light.position;
    const float distance = length(toCenter);
    if (distance > light.range + bounds.radius()) return false;
    const float cosine = dot(normalize(toCenter), normalize(light.direction));
    return cosine >= light.outerCos - 0.35f;
}

Vec3 cubeFaceDirection(int face) {
    switch (face) {
    case 0: return {1.0f, 0.0f, 0.0f};
    case 1: return {-1.0f, 0.0f, 0.0f};
    case 2: return {0.0f, 1.0f, 0.0f};
    case 3: return {0.0f, -1.0f, 0.0f};
    case 4: return {0.0f, 0.0f, 1.0f};
    default: return {0.0f, 0.0f, -1.0f};
    }
}

Vec3 cubeFaceUp(int face) {
    switch (face) {
    case 0:
    case 1:
    case 4:
    case 5: return {0.0f, 1.0f, 0.0f};
    case 2: return {0.0f, 0.0f, 1.0f};
    default: return {0.0f, 0.0f, -1.0f};
    }
}

const char* cubeFaceName(int face) {
    switch (face) {
    case 0: return "+X";
    case 1: return "-X";
    case 2: return "+Y";
    case 3: return "-Y";
    case 4: return "+Z";
    default: return "-Z";
    }
}

} // namespace

std::vector<float> cascadeSplits(float nearZ, float farZ, uint32_t count, float lambda) {
    std::vector<float> splits(count + 1);
    splits[0] = nearZ;
    for (uint32_t i = 1; i <= count; ++i) {
        const float ratio = static_cast<float>(i) / static_cast<float>(count);
        const float logSplit = nearZ * std::pow(farZ / nearZ, ratio);
        const float uniformSplit = nearZ + (farZ - nearZ) * ratio;
        splits[i] = lambda * logSplit + (1.0f - lambda) * uniformSplit;
    }
    splits[count] = farZ;
    return splits;
}

std::vector<Cascade> computeCascades(const Camera& camera, const Light& directionalLight,
                                     const CascadeConfig& config) {
    if (config.count == 0 || config.count > 8 || config.shadowSize < 64 ||
        config.shadowSize > 8192 || config.stabilizationSize < 64 ||
        config.stabilizationSize > 8192 || config.splitLambda < 0.0f ||
        config.splitLambda > 1.0f) {
        throw AppError(ErrorCode::internal, "invalid cascade configuration");
    }
    const Light* light = &directionalLight;
    std::vector<float> splits = cascadeSplits(camera.nearZ, camera.farZ, config.count,
                                              config.splitLambda);
    std::vector<Cascade> cascades;
    cascades.reserve(config.count);
    const Mat4 lightRot = orientationOnly(light->direction);
    for (uint32_t i = 0; i < config.count; ++i) {
        const auto corners = frustumCorners(camera, splits[i], splits[i + 1]);
        const Vec3 center = averageCorners(corners);
        float radius = 0.0f;
        for (const Vec3& c : corners) radius = std::max(radius, length(c - center));
        radius = std::max(radius, 0.25f);
        const Vec3 lightCenter = transformPoint(lightRot, center);
        const float texel = 2.0f * radius / static_cast<float>(config.stabilizationSize);
        const float snappedX = std::floor(lightCenter.x / texel) * texel;
        const float snappedY = std::floor(lightCenter.y / texel) * texel;
        const Mat4 lightView = lightRot *
            Mat4::translation({-snappedX, -snappedY, -lightCenter.z});
        float minZ = std::numeric_limits<float>::max();
        float maxZ = -std::numeric_limits<float>::max();
        for (const Vec3& c : corners) {
            const float z = transformPoint(lightView, c).z;
            minZ = std::min(minZ, z);
            maxZ = std::max(maxZ, z);
        }
        const float extent = radius + texel * 2.0f;
        const float nearZ = std::max(0.01f, minZ - config.casterDepthPadding);
        const float farZ = maxZ + config.casterDepthPadding;
        const Mat4 projection = orthoLH(-extent, extent, -extent, extent, nearZ, farZ);

        Cascade cascade;
        cascade.index = i;
        cascade.splitNear = splits[i];
        cascade.splitFar = splits[i + 1];
        cascade.matrix = lightView * projection;
        cascade.frustum = frustumFromViewProj(cascade.matrix);
        cascades.push_back(cascade);
    }
    return cascades;
}

uint32_t selectCascade(const CsmFrame& frame, float viewDepth) {
    for (const Cascade& cascade : frame.cascades) {
        if (viewDepth <= cascade.splitFar) return cascade.index;
    }
    return frame.cascades.empty() ? 0u : static_cast<uint32_t>(frame.cascades.size() - 1);
}

bool matrixExactlyEqual(const Mat4& a, const Mat4& b) {
    return std::memcmp(a.m.data(), b.m.data(), sizeof(float) * a.m.size()) == 0;
}

void CsmUpdatePolicy::invalidate() {
    valid_ = false;
    previousMatrices_.clear();
}

CsmFrame CsmUpdatePolicy::update(const Camera& camera, const Light& light,
                                 const Bvh& bvh, const CascadeConfig& config,
                                 bool animationDirty, bool staticSceneDirty) {
    const Mat4 view = camera.view();
    const Mat4 projection = camera.projection();
    const Vec3 lightDirection = normalize(light.direction);
    const bool matrixDirty = !valid_ ||
        !matrixExactlyEqual(view, previousView_) ||
        !matrixExactlyEqual(projection, previousProjection_) ||
        dot(lightDirection, previousLightDirection_) < 0.9999999f;
    const bool resolutionDirty = previousShadowSize_ != config.shadowSize;
    const bool contentDirty = matrixDirty || resolutionDirty || animationDirty || staticSceneDirty;

    CsmFrame frame;
    frame.cascades = computeCascades(camera, light, config);
    frame.splitDistances = {camera.nearZ, 0.0f, 0.0f, 0.0f};
    for (const Cascade& cascade : frame.cascades) {
        if (cascade.index + 1 < frame.splitDistances.size()) {
            frame.splitDistances[cascade.index + 1] = cascade.splitFar;
        }
    }
    frame.matricesStable = valid_ && previousMatrices_.size() == frame.cascades.size();
    for (size_t i = 0; i < frame.cascades.size(); ++i) {
        if (frame.matricesStable && !matrixExactlyEqual(previousMatrices_[i],
                                                        frame.cascades[i].matrix)) {
            frame.matricesStable = false;
            break;
        }
    }
    if (contentDirty) {
        std::vector<uint32_t> casters;
        BvhQueryStats stats;
        for (Cascade& cascade : frame.cascades) {
            casters.clear();
            bvh.queryFrustum(cascade.frustum, casters, &stats);
            cascade.visibleCasters = static_cast<uint32_t>(casters.size());
            cascade.updated = true;
            ++frame.updatedCascades;
        }
    }
    valid_ = true;
    previousView_ = view;
    previousProjection_ = projection;
    previousLightDirection_ = lightDirection;
    previousShadowSize_ = config.shadowSize;
    previousMatrices_.clear();
    for (const Cascade& cascade : frame.cascades) previousMatrices_.push_back(cascade.matrix);
    return frame;
}

std::vector<ShadowMatrixInfo> localShadowMatrices(const Light& light, uint32_t resolution) {
    std::vector<ShadowMatrixInfo> result;
    if (light.type == LightType::point) {
        const Vec3 position = light.position;
        for (int face = 0; face < 6; ++face) {
            const Vec3 dir = cubeFaceDirection(face);
            const Mat4 view = lookAtLH(position, position + dir, cubeFaceUp(face));
            const Mat4 projection = perspectiveLH(0.5f * kPi, 1.0f, 0.05f, light.range);
            result.push_back({cubeFaceName(face), view * projection});
        }
    } else if (light.type == LightType::spot) {
        result.push_back({"spot", lightViewProjection(light)});
    }
    (void)resolution;
    return result;
}

std::string shadowSourceNames(uint32_t sourceMask) {
    std::ostringstream out;
    bool first = true;
    auto add = [&](uint32_t bit, const char* name) {
        if ((sourceMask & bit) == 0) return;
        if (!first) out << ',';
        out << name;
        first = false;
    };
    add(shadowSourceInvalid, "invalid");
    add(shadowSourceTransform, "transform");
    add(shadowSourceVisibility, "visibility");
    add(shadowSourceAnimation, "animation");
    add(shadowSourceResolution, "resolution");
    add(shadowSourceStaticScene, "staticScene");
    add(shadowSourceDeferredBudget, "deferredBudget");
    return out.str();
}

void LocalShadowPolicy::reset(size_t lightCount) {
    states_.assign(lightCount, {});
    frame_ = 0;
}

LocalShadowFrame LocalShadowPolicy::update(const Camera& camera, const std::vector<Light>& lights,
                                           const std::vector<uint32_t>& visibleObjectIds,
                                           const std::vector<SceneObject>& objects,
                                           const AABB& characterBounds, bool characterVisible,
                                           bool characterAnimationDirty,
                                           bool staticSceneDirty,
                                           const LocalShadowConfig& config) {
    if (states_.size() != lights.size()) reset(lights.size());
    ++frame_;
    const Frustum cameraFrustum = camera.frustum();
    LocalShadowFrame frame;
    frame.lights.reserve(lights.size());
    std::vector<size_t> candidates;
    for (size_t i = 0; i < lights.size(); ++i) {
        const Light& light = lights[i];
        if (light.type == LightType::directional) continue;
        State& state = states_[i];
        LocalShadowDecision decision;
        decision.lightId = light.id;
        decision.type = light.type;
        decision.matrices = localShadowMatrices(light, static_cast<uint32_t>(config.faceResolution));

        bool overlapsVisible = false;
        const AABB influence = lightInfluenceBounds(light);
        if (cameraFrustum.intersects(influence)) {
            for (uint32_t id : visibleObjectIds) {
                if (id < objects.size() && lightOverlapsRenderable(light, objects[id].bounds)) {
                    overlapsVisible = true;
                    break;
                }
            }
            if (!overlapsVisible && characterVisible &&
                lightOverlapsRenderable(light, characterBounds)) {
                overlapsVisible = true;
            }
        }
        decision.visible = overlapsVisible;
        if (overlapsVisible) ++frame.visibleCount;

        if (!state.initialized) decision.sourceMask |= shadowSourceInvalid;
        if (length(light.position - state.position) > config.transformEpsilon ||
            dot(normalize(light.direction), normalize(state.direction)) < 0.99999f) {
            decision.sourceMask |= shadowSourceTransform;
        }
        if (overlapsVisible && !state.initialized) decision.sourceMask |= shadowSourceVisibility;
        if (characterAnimationDirty && characterVisible &&
            lightOverlapsRenderable(light, characterBounds)) {
            decision.sourceMask |= shadowSourceAnimation;
        }
        if (state.resolution != static_cast<uint32_t>(config.faceResolution)) {
            decision.sourceMask |= shadowSourceResolution;
        }
        if (staticSceneDirty) decision.sourceMask |= shadowSourceStaticScene;
        decision.dirty = decision.sourceMask != shadowSourceNone;
        if (decision.dirty) ++frame.dirtyCount;
        if (decision.visible && decision.dirty) candidates.push_back(i);
        frame.lights.push_back(std::move(decision));
    }

    std::sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
        const bool aAnimation = (frame.lights[a].sourceMask & shadowSourceAnimation) != 0;
        const bool bAnimation = (frame.lights[b].sourceMask & shadowSourceAnimation) != 0;
        if (aAnimation != bAnimation) return aAnimation > bAnimation;
        if (states_[a].lastUpdatedFrame != states_[b].lastUpdatedFrame) {
            return states_[a].lastUpdatedFrame < states_[b].lastUpdatedFrame;
        }
        return lights[a].id < lights[b].id;
    });
    const uint32_t updateBudget = std::min<uint32_t>(
        config.maxUpdatesPerFrame, static_cast<uint32_t>(candidates.size()));
    for (uint32_t i = 0; i < updateBudget; ++i) {
        const size_t index = candidates[i];
        LocalShadowDecision& decision = frame.lights[index];
        decision.updated = true;
        ++frame.updatedCount;
        State& state = states_[index];
        state.initialized = true;
        state.position = lights[index].position;
        state.direction = lights[index].direction;
        state.resolution = static_cast<uint32_t>(config.faceResolution);
        state.lastUpdatedFrame = frame_;
    }
    for (size_t i = updateBudget; i < candidates.size(); ++i) {
        LocalShadowDecision& decision = frame.lights[candidates[i]];
        decision.deferredByBudget = true;
        decision.sourceMask |= shadowSourceDeferredBudget;
    }
    return frame;
}

} // namespace uid166

