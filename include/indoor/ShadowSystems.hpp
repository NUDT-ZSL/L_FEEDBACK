#pragma once

#include "indoor/Scene.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <optional>
#include <unordered_map>
#include <vector>

namespace indoor {

class ShadowVersionLedger {
public:
    void initialize(const std::vector<Light>& lights);
    void clear();

    std::uint64_t version(std::uint32_t lightId) const;
    std::uint64_t bump(std::uint32_t lightId);
    void remove(std::uint32_t lightId);

    [[nodiscard]] std::uint64_t migrations() const { return migrations_; }
    [[nodiscard]] std::size_t trackedLights() const { return versions_.size(); }

private:
    std::unordered_map<std::uint32_t, std::uint64_t> versions_;
    std::uint64_t migrations_ = 0;
};

struct ShadowCacheEntry {
    std::uint32_t lightId = 0;
    std::uint64_t version = 0;
    std::uint64_t lastUse = 0;
};

struct CacheAcquireResult {
    bool created = false;
    std::optional<std::uint32_t> evictedLightId;
};

// Render-thread logical cache. The GL owner mirrors this policy and owns the
// actual texture handles. Keeping this class GL-free makes LRU behavior easy to
// test without a context.
class ShadowCachePolicy {
public:
    explicit ShadowCachePolicy(std::size_t capacity);

    CacheAcquireResult acquire(std::uint32_t lightId);
    bool contains(std::uint32_t lightId) const;
    bool release(std::uint32_t lightId);
    void clear();
    void setVersion(std::uint32_t lightId, std::uint64_t version);

    [[nodiscard]] std::uint64_t version(std::uint32_t lightId) const;
    [[nodiscard]] std::size_t size() const { return entries_.size(); }
    [[nodiscard]] std::size_t capacity() const { return capacity_; }
    [[nodiscard]] std::uint64_t tick() const { return tick_; }
    [[nodiscard]] std::vector<std::uint32_t> lruOrder() const;

private:
    void touch(std::uint32_t lightId);
    std::optional<std::uint32_t> evictLeastRecentlyUsed();

    std::size_t capacity_ = 0;
    std::uint64_t tick_ = 0;
    std::unordered_map<std::uint32_t, ShadowCacheEntry> entries_;
    std::list<std::uint32_t> lru_;
};

} // namespace indoor
