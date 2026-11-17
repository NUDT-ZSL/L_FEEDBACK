#include "visibility.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cv {

UniformGrid::UniformGrid(float cellSize) : cellSize_(cellSize) {}

void UniformGrid::build(const Scene& scene) {
    faces_.clear();
    cells_.clear();
    cellBounds_.clear();

    AABB sceneBounds;
    for (const Mesh& mesh : scene.meshes) {
        for (const Vertex& v : mesh.vertices) {
            sceneBounds.expand(v.position);
        }
    }
    if (!sceneBounds.valid()) {
        sceneBounds = {{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}};
    }
    sceneBounds.expand(Vec3{-3.0f, 2.4f, 2.0f});
    sceneBounds.expand(Vec3{3.0f, 0.0f, -2.0f});
    sceneBounds.pad(0.30f);
    boundsMin_ = sceneBounds.min;
    boundsMax_ = sceneBounds.max;

    nx_ = std::max(1, static_cast<int>(std::ceil((boundsMax_.x - boundsMin_.x) / cellSize_)));
    ny_ = std::max(1, static_cast<int>(std::ceil((boundsMax_.y - boundsMin_.y) / cellSize_)));
    nz_ = std::max(1, static_cast<int>(std::ceil((boundsMax_.z - boundsMin_.z) / cellSize_)));
    const size_t totalCells =
        static_cast<size_t>(nx_) * static_cast<size_t>(ny_) * static_cast<size_t>(nz_);
    cells_.resize(totalCells);
    cellBounds_.resize(totalCells);

    for (int z = 0; z < nz_; ++z) {
        for (int y = 0; y < ny_; ++y) {
            for (int x = 0; x < nx_; ++x) {
                const int index = cellIndex(x, y, z);
                cellBounds_[static_cast<size_t>(index)] = {
                    {boundsMin_.x + static_cast<float>(x) * cellSize_,
                     boundsMin_.y + static_cast<float>(y) * cellSize_,
                     boundsMin_.z + static_cast<float>(z) * cellSize_},
                    {boundsMin_.x + static_cast<float>(x + 1) * cellSize_,
                     boundsMin_.y + static_cast<float>(y + 1) * cellSize_,
                     boundsMin_.z + static_cast<float>(z + 1) * cellSize_},
                };
            }
        }
    }

    uint32_t globalIndex = 0;
    for (size_t meshIndex = 0; meshIndex < scene.meshes.size(); ++meshIndex) {
        const Mesh& mesh = scene.meshes[meshIndex];
        const size_t triangleCount = mesh.indices.size() / 3;
        for (size_t tri = 0; tri < triangleCount; ++tri) {
            AABB bounds;
            for (int k = 0; k < 3; ++k) {
                const uint32_t index = mesh.indices[tri * 3 + static_cast<size_t>(k)];
                bounds.expand(mesh.vertices[index].position);
            }
            // Animation displaces vertices, so insertion is conservatively padded.
            bounds.pad(0.42f);
            const int minX = std::clamp(
                static_cast<int>(std::floor((bounds.min.x - boundsMin_.x) / cellSize_)), 0,
                nx_ - 1);
            const int minY = std::clamp(
                static_cast<int>(std::floor((bounds.min.y - boundsMin_.y) / cellSize_)), 0,
                ny_ - 1);
            const int minZ = std::clamp(
                static_cast<int>(std::floor((bounds.min.z - boundsMin_.z) / cellSize_)), 0,
                nz_ - 1);
            const int maxX = std::clamp(
                static_cast<int>(std::floor((bounds.max.x - boundsMin_.x) / cellSize_)), 0,
                nx_ - 1);
            const int maxY = std::clamp(
                static_cast<int>(std::floor((bounds.max.y - boundsMin_.y) / cellSize_)), 0,
                ny_ - 1);
            const int maxZ = std::clamp(
                static_cast<int>(std::floor((bounds.max.z - boundsMin_.z) / cellSize_)), 0,
                nz_ - 1);

            for (int z = minZ; z <= maxZ; ++z) {
                for (int y = minY; y <= maxY; ++y) {
                    for (int x = minX; x <= maxX; ++x) {
                        cells_[static_cast<size_t>(cellIndex(x, y, z))].push_back(globalIndex);
                    }
                }
            }

            FaceRef ref;
            ref.mesh = static_cast<uint16_t>(meshIndex);
            ref.material = mesh.triangleMaterial[tri];
            ref.triangle = static_cast<uint32_t>(tri);
            ref.globalIndex = globalIndex;
            ref.vertexIndex = {
                mesh.indices[tri * 3 + 0],
                mesh.indices[tri * 3 + 1],
                mesh.indices[tri * 3 + 2],
            };
            faces_.push_back(ref);
            ++globalIndex;
        }
    }
    faceStamp_.assign(faces_.size(), 0);
    queryStamp_ = 0;
}

std::vector<FaceRef> UniformGrid::query(
    const Frustum& frustum, const std::vector<std::vector<FrameVertex>>& frameVertices,
    VisibilityStats& stats) {
    ++queryStamp_;
    if (queryStamp_ == 0) {
        std::fill(faceStamp_.begin(), faceStamp_.end(), 0);
        queryStamp_ = 1;
    }
    visibleCells_.clear();
    for (uint32_t i = 0; i < cells_.size(); ++i) {
        if (frustum.intersects(cellBounds_[i])) {
            visibleCells_.push_back(i);
        }
    }

    std::vector<uint32_t> candidates;
    for (uint32_t cell : visibleCells_) {
        for (uint32_t face : cells_[cell]) {
            if (faceStamp_[face] != queryStamp_) {
                faceStamp_[face] = queryStamp_;
                candidates.push_back(face);
            }
        }
    }
    std::sort(candidates.begin(), candidates.end());

    std::vector<FaceRef> visible;
    visible.reserve(candidates.size());
    for (uint32_t faceIndex : candidates) {
        const FaceRef& face = faces_[faceIndex];
        const std::vector<FrameVertex>& verts = frameVertices[face.mesh];
        AABB bounds;
        for (uint32_t vertexIndex : face.vertexIndex) {
            bounds.expand(verts[vertexIndex].worldPosition);
        }
        bounds.pad(0.001f);
        if (frustum.intersects(bounds)) {
            visible.push_back(face);
        }
    }
    stats = {};
    stats.totalFaces = static_cast<uint32_t>(faces_.size());
    stats.visibleCells = static_cast<uint32_t>(visibleCells_.size());
    stats.totalCells = static_cast<uint32_t>(cells_.size());
    stats.cellCandidateFaces = static_cast<uint32_t>(candidates.size());
    stats.frustumVisibleFaces = static_cast<uint32_t>(visible.size());
    return visible;
}

}  // namespace cv

