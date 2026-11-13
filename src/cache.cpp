#include "terrain/cache.hpp"

#include <algorithm>

namespace terrain {

NodeStateCache::NodeStateCache(std::size_t capacity)
    : capacity_(std::max<std::size_t>(1, capacity)) {}

void NodeStateCache::beginFrame() {
    useSerial_ = 0;
    for (auto& [id, entry] : entries_) {
        (void)id;
        entry.data.seenThisFrame = false;
    }
}

NodeCacheEntry& NodeStateCache::acquire(
    uint64_t nodeId,
    uint64_t version,
    uint64_t frame,
    CacheStats& stats) {
    auto iterator = entries_.find(nodeId);
    if (iterator == entries_.end()) {
        Entry entry;
        entry.data.version = version;
        entry.data.lastSeenFrame = frame;
        entry.data.seenThisFrame = true;
        entry.useSerial = ++useSerial_;
        ++stats.misses;
        auto [inserted, ok] = entries_.emplace(nodeId, std::move(entry));
        (void)ok;
        return inserted->second.data;
    }

    Entry& entry = iterator->second;
    entry.data.seenThisFrame = true;
    entry.data.lastSeenFrame = frame;
    entry.useSerial = ++useSerial_;
    if (entry.data.version != version) {
        ++stats.misses;
        ++stats.versionInvalidations;
        entry.data.version = version;
        entry.data.stitchOffsets.clear();
        entry.data.selectedDepth = 0;
        entry.data.selectedAsLeaf = false;
        entry.data.hasSelectionHistory = false;
        entry.data.lastStateChangeFrame = frame;
    } else {
        ++stats.hits;
    }
    return entry.data;
}

NodeCacheEntry* NodeStateCache::find(uint64_t nodeId, uint64_t version) {
    auto iterator = entries_.find(nodeId);
    if (iterator == entries_.end() || iterator->second.data.version != version) {
        return nullptr;
    }
    return &iterator->second.data;
}

void NodeStateCache::storeStitchOffsets(
    uint64_t nodeId,
    const std::vector<CachedStitchOffset>& offsets) {
    auto iterator = entries_.find(nodeId);
    if (iterator != entries_.end()) {
        iterator->second.data.stitchOffsets = offsets;
    }
}

void NodeStateCache::endFrame(uint64_t frame, CacheStats& stats) {
    (void)frame;
    while (entries_.size() > capacity_) {
        auto victim = entries_.end();
        for (auto iterator = entries_.begin(); iterator != entries_.end(); ++iterator) {
            if (iterator->second.data.seenThisFrame) {
                continue;
            }
            if (victim == entries_.end() ||
                iterator->second.useSerial < victim->second.useSerial) {
                victim = iterator;
            }
        }
        if (victim == entries_.end()) {
            break;
        }
        entries_.erase(victim);
        ++stats.evictions;
    }
}

}  // namespace terrain

