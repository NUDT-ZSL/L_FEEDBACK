#pragma once

#include "core/math.hpp"

#include <cstdint>
#include <vector>

namespace uid166 {

struct BvhQueryStats {
    uint32_t nodesVisited = 0;
    uint32_t leavesVisited = 0;
    uint32_t objectsTested = 0;
};

struct BvhNode {
    AABB bounds{};
    int32_t parent = -1;
    int32_t left = -1;
    int32_t right = -1;
    uint32_t first = 0;
    uint32_t count = 0;

    [[nodiscard]] bool leaf() const { return count != 0; }
};

class Bvh {
public:
    void build(const std::vector<AABB>& objectBounds);
    void updateObject(uint32_t objectId, const AABB& newBounds);
    void refitAll(const std::vector<AABB>& objectBounds);

    void queryFrustum(const Frustum& frustum, std::vector<uint32_t>& out,
                      BvhQueryStats* stats = nullptr) const;
    void queryAABB(const AABB& query, std::vector<uint32_t>& out,
                   BvhQueryStats* stats = nullptr) const;

    [[nodiscard]] size_t nodeCount() const { return nodes_.size(); }
    [[nodiscard]] const AABB& rootBounds() const { return nodes_.at(0).bounds; }
    [[nodiscard]] bool valid() const { return !nodes_.empty(); }
    [[nodiscard]] const AABB& objectBounds(uint32_t objectId) const;

private:
    int32_t buildRecursive(uint32_t begin, uint32_t end, int32_t parent);
    void refitLeaf(int32_t leaf, int32_t parent);

    std::vector<BvhNode> nodes_;
    std::vector<uint32_t> objectOrder_;
    std::vector<AABB> objectBounds_;
    std::vector<int32_t> leafForObject_;
};

} // namespace uid166

