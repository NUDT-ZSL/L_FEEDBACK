#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace terrain {

struct CachedStitchOffset {
    uint16_t vertexIndex = 0;
    float offset = 0.0f;
};

struct CacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t versionInvalidations = 0;
};

struct NodeCacheEntry {
    uint64_t version = 0;
    uint64_t lastSeenFrame = 0;
    uint64_t lastStateChangeFrame = 0;
    uint8_t selectedDepth = 0;
    bool selectedAsLeaf = false;
    bool hasSelectionHistory = false;
    bool seenThisFrame = false;
    std::vector<CachedStitchOffset> stitchOffsets;
};

class NodeStateCache {
public:
    explicit NodeStateCache(std::size_t capacity = 2048);

    NodeCacheEntry& acquire(uint64_t nodeId, uint64_t version, uint64_t frame, CacheStats& stats);
    NodeCacheEntry* find(uint64_t nodeId, uint64_t version);

    void beginFrame();
    void endFrame(uint64_t frame, CacheStats& stats);

    void storeStitchOffsets(
        uint64_t nodeId,
        const std::vector<CachedStitchOffset>& offsets);

    std::size_t size() const { return entries_.size(); }
    std::size_t capacity() const { return capacity_; }

private:
    struct Entry {
        NodeCacheEntry data;
        uint64_t useSerial = 0;
    };

    std::size_t capacity_;
    uint64_t useSerial_ = 0;
    std::unordered_map<uint64_t, Entry> entries_;
};

}  // namespace terrain

