#include "core/culling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace uid166 {

namespace {

struct ProjectedRect {
    bool valid = false;
    float nearDepth = 1.0f;
    int x0 = 0;
    int y0 = 0;
    int x1 = -1;
    int y1 = -1;
};

ProjectedRect projectAABB(const Mat4& viewProj, const AABB& bounds) {
    ProjectedRect rect;
    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = -std::numeric_limits<float>::max();
    float maxY = -std::numeric_limits<float>::max();
    float minZ = 1.0f;
    float maxZ = 0.0f;
    for (int i = 0; i < 8; ++i) {
        const Vec4 clip = transform({bounds.corner(i).x, bounds.corner(i).y,
                                     bounds.corner(i).z, 1.0f}, viewProj);
        if (clip.w <= 0.001f) return rect;
        const float x = clip.x / clip.w;
        const float y = clip.y / clip.w;
        const float z = clip.z / clip.w;
        minX = std::min(minX, x);
        maxX = std::max(maxX, x);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
        minZ = std::min(minZ, z);
        maxZ = std::max(maxZ, z);
    }
    if (maxZ < 0.0f || minZ > 1.0f || maxX < -1.0f || minX > 1.0f ||
        maxY < -1.0f || minY > 1.0f) {
        return rect;
    }
    const float w = static_cast<float>(DepthHierarchy::kWidth);
    const float h = static_cast<float>(DepthHierarchy::kHeight);
    rect.x0 = std::clamp(static_cast<int>(std::floor((minX * 0.5f + 0.5f) * w)), 0,
                         static_cast<int>(DepthHierarchy::kWidth) - 1);
    rect.x1 = std::clamp(static_cast<int>(std::ceil((maxX * 0.5f + 0.5f) * w)) - 1, 0,
                         static_cast<int>(DepthHierarchy::kWidth) - 1);
    rect.y0 = std::clamp(static_cast<int>(std::floor((1.0f - (maxY * 0.5f + 0.5f)) * h)), 0,
                         static_cast<int>(DepthHierarchy::kHeight) - 1);
    rect.y1 = std::clamp(static_cast<int>(std::ceil((1.0f - (minY * 0.5f + 0.5f)) * h)) - 1, 0,
                         static_cast<int>(DepthHierarchy::kHeight) - 1);
    rect.valid = rect.x0 <= rect.x1 && rect.y0 <= rect.y1;
    rect.nearDepth = std::clamp(minZ, 0.0f, 1.0f);
    return rect;
}

} // namespace

void DepthHierarchy::clear() {
    levels_.clear();
    Level base;
    base.width = kWidth;
    base.height = kHeight;
    base.values.assign(static_cast<size_t>(kWidth) * kHeight, 1.0f);
    levels_.push_back(std::move(base));
    owners_.assign(static_cast<size_t>(kWidth) * kHeight, 0xffffffffu);
    valid_ = false;
}

void DepthHierarchy::rasterizeAABB(const Mat4& viewProj, const AABB& bounds,
                                   uint32_t ownerId) {
    if (levels_.empty()) clear();
    const ProjectedRect rect = projectAABB(viewProj, bounds);
    if (!rect.valid) return;
    Level& level = levels_[0];
    for (int y = rect.y0; y <= rect.y1; ++y) {
        for (int x = rect.x0; x <= rect.x1; ++x) {
            const size_t index = static_cast<size_t>(y) * level.width + static_cast<size_t>(x);
            if (rect.nearDepth < level.values[index]) {
                level.values[index] = rect.nearDepth;
                owners_[index] = ownerId;
            }
        }
    }
}

void DepthHierarchy::buildPyramid() {
    if (levels_.empty()) clear();
    while (levels_.back().width > 1 || levels_.back().height > 1) {
        const Level& src = levels_.back();
        Level dst;
        dst.width = std::max(1u, (src.width + 1) / 2);
        dst.height = std::max(1u, (src.height + 1) / 2);
        dst.values.assign(static_cast<size_t>(dst.width) * dst.height, 0.0f);
        for (uint32_t y = 0; y < dst.height; ++y) {
            for (uint32_t x = 0; x < dst.width; ++x) {
                float value = 0.0f;
                for (uint32_t oy = 0; oy < 2; ++oy) {
                    for (uint32_t ox = 0; ox < 2; ++ox) {
                        const uint32_t sx = x * 2 + ox;
                        const uint32_t sy = y * 2 + oy;
                        if (sx < src.width && sy < src.height) {
                            value = std::max(value, src.values[static_cast<size_t>(sy) * src.width + sx]);
                        }
                    }
                }
                dst.values[static_cast<size_t>(y) * dst.width + x] = value;
            }
        }
        levels_.push_back(std::move(dst));
    }
    valid_ = true;
}

bool DepthHierarchy::mostlyOccluded(const Mat4& viewProj, const AABB& bounds,
                                    uint32_t ownerId, float depthBias,
                                    uint32_t expandPixels) const {
    if (!valid_ || levels_.empty()) return false;
    ProjectedRect rect = projectAABB(viewProj, bounds);
    if (!rect.valid) return false;
    rect.x0 -= static_cast<int>(expandPixels);
    rect.x1 += static_cast<int>(expandPixels);
    rect.y0 -= static_cast<int>(expandPixels);
    rect.y1 += static_cast<int>(expandPixels);
    if (rect.x0 < 0 || rect.y0 < 0 || rect.x1 >= static_cast<int>(kWidth) ||
        rect.y1 >= static_cast<int>(kHeight)) {
        return false;
    }
    const Level& base = levels_[0];
    uint32_t total = 0;
    uint32_t covered = 0;
    const float threshold = rect.nearDepth - depthBias;
    for (int y = rect.y0; y <= rect.y1; ++y) {
        for (int x = rect.x0; x <= rect.x1; ++x) {
            const size_t index = static_cast<size_t>(y) * base.width + static_cast<size_t>(x);
            ++total;
            if (owners_[index] != ownerId && base.values[index] < threshold) ++covered;
        }
    }
    return total != 0 && static_cast<float>(covered) / static_cast<float>(total) >= 0.78f;
}

float DepthHierarchy::maxDepth() const {
    if (levels_.empty()) return 1.0f;
    const Level& root = levels_.back();
    return root.values.empty() ? 1.0f : root.values[0];
}

bool cameraIsFast(const Camera& current, const Camera& previous, const CullOptions& options) {
    const Vec3 currentForward = normalize(current.target - current.eye);
    const Vec3 previousForward = normalize(previous.target - previous.eye);
    const float cosine = std::clamp(dot(currentForward, previousForward), -1.0f, 1.0f);
    const float angle = std::acos(cosine) * 180.0f / kPi;
    return length(current.eye - previous.eye) > options.fastCameraDistance ||
           angle > options.fastCameraAngleDegrees;
}

void OcclusionCuller::reset(size_t objectCount) {
    previousQueries_.assign(objectCount, 0);
    currentQueries_.assign(objectCount, 0);
    queryCandidates_.clear();
    previousDepth_.clear();
    previousDepth_.buildPyramid();
    previousViewProjection_ = Mat4::identity();
    hasPreviousFrame_ = false;
}

 CullResult OcclusionCuller::cull(const std::vector<SceneObject>& objects, const Bvh& bvh,
                                 const Camera& camera, const Camera& previousCamera,
                                 const CullOptions& options) {
    if (previousQueries_.size() != objects.size()) reset(objects.size());
    CullResult result;
    result.totalObjects = static_cast<uint32_t>(objects.size());
    result.cameraFast = hasPreviousFrame_ && cameraIsFast(camera, previousCamera, options);
    const Mat4 currentViewProjection = camera.viewProjection();
    bvh.queryFrustum(camera.frustum(), result.frustumVisible, &result.bvhStats);
    result.frustumRejected = result.totalObjects -
                             static_cast<uint32_t>(result.frustumVisible.size());

    const float movement = hasPreviousFrame_ ? length(camera.eye - previousCamera.eye) : 0.0f;
    const uint32_t expansion = hasPreviousFrame_
        ? static_cast<uint32_t>(std::clamp(movement * options.pixelsPerMeter, 0.0f,
                                           static_cast<float>(options.maxExpandPixels)))
        : 0u;

    for (uint32_t id : result.frustumVisible) {
        bool occluded = false;
        if (options.enableOcclusion && hasPreviousFrame_ && id < previousQueries_.size() &&
            id < objects.size() && previousDepth_.valid()) {
            const bool depthOccluded = previousDepth_.mostlyOccluded(
                previousViewProjection_, objects[id].bounds, id, options.depthBias, expansion);
            if (depthOccluded) {
                ++result.hierarchyTests;
                if (result.cameraFast) {
                    occluded = true;
                } else {
                    occluded = previousQueries_[id] != 0;
                    if (movement > 0.02f) occluded = occluded && depthOccluded;
                }
            }
            if (previousQueries_[id]) ++result.delayedQueryResults;
        }
        if (occluded) result.occluded.push_back(id);
        else result.visible.push_back(id);
    }
    std::sort(result.occluded.begin(), result.occluded.end());
    std::sort(result.visible.begin(), result.visible.end());

    DepthHierarchy currentDepth;
    currentDepth.clear();
    for (uint32_t id : result.frustumVisible) {
        if (id < objects.size()) {
            currentDepth.rasterizeAABB(currentViewProjection, objects[id].bounds, id);
        }
    }
    currentDepth.buildPyramid();
    std::fill(currentQueries_.begin(), currentQueries_.end(), 0);
    queryCandidates_.clear();
    for (uint32_t id : result.frustumVisible) {
        if (id >= objects.size()) continue;
        const bool selfMinusBias = currentDepth.mostlyOccluded(
            currentViewProjection, objects[id].bounds, id, options.depthBias, 0);
        currentQueries_[id] = selfMinusBias ? 1 : 0;
        if (selfMinusBias) queryCandidates_.push_back(id);
    }
    result.occlusionCandidates = static_cast<uint32_t>(result.frustumVisible.size());

    previousDepth_ = std::move(currentDepth);
    previousViewProjection_ = currentViewProjection;
    std::swap(previousQueries_, currentQueries_);
    hasPreviousFrame_ = true;
    return result;
}

} // namespace uid166

