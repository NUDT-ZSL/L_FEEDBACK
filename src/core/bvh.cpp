#include "core/bvh.hpp"

#include "core/error.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace uid166 {

namespace {

constexpr uint32_t kLeafSize = 8;

} // namespace

void Bvh::build(const std::vector<AABB>& objectBounds) {
    if (objectBounds.empty()) throw AppError(ErrorCode::internal, "BVH requires at least one object");
    objectBounds_ = objectBounds;
    objectOrder_.resize(objectBounds.size());
    std::iota(objectOrder_.begin(), objectOrder_.end(), 0u);
    leafForObject_.assign(objectBounds.size(), -1);
    nodes_.clear();
    nodes_.reserve(objectBounds.size() * 2);
    buildRecursive(0, static_cast<uint32_t>(objectOrder_.size()), -1);
}

int32_t Bvh::buildRecursive(uint32_t begin, uint32_t end, int32_t parent) {
    const int32_t nodeIndex = static_cast<int32_t>(nodes_.size());
    nodes_.push_back({});
    nodes_.back().parent = parent;
    AABB bounds;
    AABB centroidBounds;
    for (uint32_t i = begin; i < end; ++i) {
        const AABB& b = objectBounds_[objectOrder_[i]];
        bounds.expand(b);
        centroidBounds.expand(b.center());
    }
    nodes_[static_cast<size_t>(nodeIndex)].bounds = bounds;
    const uint32_t count = end - begin;
    if (count <= kLeafSize) {
        BvhNode& node = nodes_[static_cast<size_t>(nodeIndex)];
        node.first = begin;
        node.count = count;
        for (uint32_t i = begin; i < end; ++i) {
            leafForObject_[objectOrder_[i]] = nodeIndex;
        }
        return nodeIndex;
    }

    const Vec3 extent = centroidBounds.mx - centroidBounds.mn;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if ((axis == 0 ? extent.x : extent.y) < extent.z) axis = 2;
    const uint32_t mid = begin + count / 2;
    std::sort(objectOrder_.begin() + begin, objectOrder_.begin() + end,
              [this, axis](uint32_t a, uint32_t b) {
                  const Vec3 ca = objectBounds_[a].center();
                  const Vec3 cb = objectBounds_[b].center();
                  const float va = axis == 0 ? ca.x : (axis == 1 ? ca.y : ca.z);
                  const float vb = axis == 0 ? cb.x : (axis == 1 ? cb.y : cb.z);
                  if (va != vb) return va < vb;
                  return a < b;
              });
    const int32_t left = buildRecursive(begin, mid, nodeIndex);
    const int32_t right = buildRecursive(mid, end, nodeIndex);
    nodes_[static_cast<size_t>(nodeIndex)].left = left;
    nodes_[static_cast<size_t>(nodeIndex)].right = right;
    return nodeIndex;
}

void Bvh::updateObject(uint32_t objectId, const AABB& newBounds) {
    if (objectId >= objectBounds_.size() || !newBounds.valid()) {
        throw AppError(ErrorCode::internal, "dynamic BVH update received invalid object or bounds");
    }
    objectBounds_[objectId] = newBounds;
    const int32_t leafIndex = leafForObject_[objectId];
    if (leafIndex < 0) throw AppError(ErrorCode::internal, "object is not present in BVH");
    const int32_t parent = nodes_[static_cast<size_t>(leafIndex)].parent;
    refitLeaf(leafIndex, parent);
}

void Bvh::refitLeaf(int32_t leafIndex, int32_t parent) {
    BvhNode& leaf = nodes_[static_cast<size_t>(leafIndex)];
    AABB bounds;
    for (uint32_t i = 0; i < leaf.count; ++i) {
        bounds.expand(objectBounds_[objectOrder_[leaf.first + i]]);
    }
    leaf.bounds = bounds;
    int32_t current = parent;
    while (current >= 0) {
        BvhNode& node = nodes_[static_cast<size_t>(current)];
        AABB merged = nodes_[static_cast<size_t>(node.left)].bounds;
        merged.expand(nodes_[static_cast<size_t>(node.right)].bounds);
        if (merged.mn.x == node.bounds.mn.x && merged.mn.y == node.bounds.mn.y &&
            merged.mn.z == node.bounds.mn.z && merged.mx.x == node.bounds.mx.x &&
            merged.mx.y == node.bounds.mx.y && merged.mx.z == node.bounds.mx.z) {
            break;
        }
        node.bounds = merged;
        current = node.parent;
    }
}

void Bvh::refitAll(const std::vector<AABB>& objectBounds) {
    if (objectBounds.size() != objectBounds_.size()) {
        throw AppError(ErrorCode::internal, "BVH refit size mismatch");
    }
    objectBounds_ = objectBounds;
    for (int32_t nodeIndex = static_cast<int32_t>(nodes_.size()) - 1; nodeIndex >= 0; --nodeIndex) {
        BvhNode& node = nodes_[static_cast<size_t>(nodeIndex)];
        if (!node.leaf()) {
            AABB merged = nodes_[static_cast<size_t>(node.left)].bounds;
            merged.expand(nodes_[static_cast<size_t>(node.right)].bounds);
            node.bounds = merged;
        } else {
            AABB bounds;
            for (uint32_t i = 0; i < node.count; ++i) {
                bounds.expand(objectBounds_[objectOrder_[node.first + i]]);
            }
            node.bounds = bounds;
        }
    }
}

void Bvh::queryFrustum(const Frustum& frustum, std::vector<uint32_t>& out,
                       BvhQueryStats* stats) const {
    out.clear();
    if (nodes_.empty()) return;
    std::vector<int32_t> stack;
    stack.reserve(64);
    stack.push_back(0);
    while (!stack.empty()) {
        const int32_t index = stack.back();
        stack.pop_back();
        const BvhNode& node = nodes_[static_cast<size_t>(index)];
        if (stats) ++stats->nodesVisited;
        if (!frustum.intersects(node.bounds)) continue;
        if (node.leaf()) {
            if (stats) ++stats->leavesVisited;
            for (uint32_t i = 0; i < node.count; ++i) {
                const uint32_t id = objectOrder_[node.first + i];
                if (stats) ++stats->objectsTested;
                if (frustum.intersects(objectBounds_[id])) out.push_back(id);
            }
        } else {
            stack.push_back(node.left);
            stack.push_back(node.right);
        }
    }
    std::sort(out.begin(), out.end());
}

void Bvh::queryAABB(const AABB& query, std::vector<uint32_t>& out,
                    BvhQueryStats* stats) const {
    out.clear();
    if (nodes_.empty()) return;
    std::vector<int32_t> stack;
    stack.reserve(64);
    stack.push_back(0);
    while (!stack.empty()) {
        const int32_t index = stack.back();
        stack.pop_back();
        const BvhNode& node = nodes_[static_cast<size_t>(index)];
        if (stats) ++stats->nodesVisited;
        if (node.bounds.mx.x < query.mn.x || node.bounds.mn.x > query.mx.x ||
            node.bounds.mx.y < query.mn.y || node.bounds.mn.y > query.mx.y ||
            node.bounds.mx.z < query.mn.z || node.bounds.mn.z > query.mx.z) {
            continue;
        }
        if (node.leaf()) {
            if (stats) ++stats->leavesVisited;
            for (uint32_t i = 0; i < node.count; ++i) {
                const uint32_t id = objectOrder_[node.first + i];
                if (stats) ++stats->objectsTested;
                const AABB& box = objectBounds_[id];
                if (!(box.mx.x < query.mn.x || box.mn.x > query.mx.x ||
                      box.mx.y < query.mn.y || box.mn.y > query.mx.y ||
                      box.mx.z < query.mn.z || box.mn.z > query.mx.z)) {
                    out.push_back(id);
                }
            }
        } else {
            stack.push_back(node.left);
            stack.push_back(node.right);
        }
    }
    std::sort(out.begin(), out.end());
}

const AABB& Bvh::objectBounds(uint32_t objectId) const {
    if (objectId >= objectBounds_.size()) {
        throw AppError(ErrorCode::internal, "BVH object id out of range");
    }
    return objectBounds_[objectId];
}

} // namespace uid166

