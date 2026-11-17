#pragma once

#include "assets.h"
#include "cv_math.h"

#include <cstdint>
#include <vector>

namespace cv {

struct FaceRef {
    uint16_t mesh = 0;
    uint16_t material = 0;
    uint32_t triangle = 0;
    uint32_t globalIndex = 0;
    std::array<uint32_t, 3> vertexIndex{};
};

struct VisibilityStats {
    uint32_t totalFaces = 0;
    uint32_t visibleCells = 0;
    uint32_t totalCells = 0;
    uint32_t cellCandidateFaces = 0;
    uint32_t frustumVisibleFaces = 0;
};

class UniformGrid {
public:
    explicit UniformGrid(float cellSize = 1.25f);

    void build(const Scene& scene);

    // The visible-cell set is rebuilt from scratch every call (no sticky state).
    std::vector<FaceRef> query(const Frustum& frustum,
                               const std::vector<std::vector<FrameVertex>>& frameVertices,
                               VisibilityStats& stats);

    uint32_t cellCount() const { return static_cast<uint32_t>(cells_.size()); }
    float cellSize() const { return cellSize_; }
    Vec3 boundsMin() const { return boundsMin_; }
    Vec3 boundsMax() const { return boundsMax_; }
    int dimensionX() const { return nx_; }
    int dimensionY() const { return ny_; }
    int dimensionZ() const { return nz_; }

private:
    int cellIndex(int x, int y, int z) const { return (z * ny_ + y) * nx_ + x; }

    float cellSize_;
    Vec3 boundsMin_;
    Vec3 boundsMax_;
    int nx_ = 0;
    int ny_ = 0;
    int nz_ = 0;
    std::vector<AABB> cellBounds_;
    std::vector<std::vector<uint32_t>> cells_;
    std::vector<FaceRef> faces_;
    std::vector<uint32_t> faceStamp_;
    uint32_t queryStamp_ = 0;
    std::vector<uint32_t> visibleCells_;
};

}  // namespace cv

