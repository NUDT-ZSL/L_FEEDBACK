#include "indoor/ShadowSystems.hpp"

#include <algorithm>

namespace indoor {

void ShadowVersionLedger::initialize(const std::vector<Light>& lights) {
    for (const Light& light : lights) {
        versions_.try_emplace(light.id, 1);
    }
}

void ShadowVersionLedger::clear() {
    versions_.clear();
    migrations_ = 0;
}

std::uint64_t ShadowVersionLedger::version(std::uint32_t lightId) const {
    const auto it = versions_.find(lightId);
    return it == versions_.end() ? 0 : it->second;
}

std::uint64_t ShadowVersionLedger::bump(std::uint32_t lightId) {
    auto it = versions_.find(lightId);
    if (it == versions_.end()) {
        it = versions_.emplace(lightId, 1).first;
    } else {
        ++it->second;
    }
    ++migrations_;
    return it->second;
}

void ShadowVersionLedger::remove(std::uint32_t lightId) {
    versions_.erase(lightId);
}

ShadowCachePolicy::ShadowCachePolicy(std::size_t capacity)
    : capacity_(capacity) {}

CacheAcquireResult ShadowCachePolicy::acquire(std::uint32_t lightId) {
    CacheAcquireResult result;
    const auto found = entries_.find(lightId);
    if (found != entries_.end()) {
        touch(lightId);
        return result;
    }

    if (capacity_ == 0) {
        return result;
    }
    if (entries_.size() >= capacity_) {
        result.evictedLightId = evictLeastRecentlyUsed();
    }

    ShadowCacheEntry entry;
    entry.lightId = lightId;
    entry.version = 0;
    entry.lastUse = ++tick_;
    entries_.emplace(lightId, entry);
    lru_.push_back(lightId);
    result.created = true;
    return result;
}

bool ShadowCachePolicy::contains(std::uint32_t lightId) const {
    return entries_.find(lightId) != entries_.end();
}

bool ShadowCachePolicy::release(std::uint32_t lightId) {
    const auto it = entries_.find(lightId);
    if (it == entries_.end()) {
        return false;
    }
    entries_.erase(it);
    lru_.remove(lightId);
    return true;
}

void ShadowCachePolicy::clear() {
    entries_.clear();
    lru_.clear();
    tick_ = 0;
}

void ShadowCachePolicy::setVersion(std::uint32_t lightId, std::uint64_t version) {
    const auto it = entries_.find(lightId);
    if (it != entries_.end()) {
        it->second.version = version;
        touch(lightId);
    }
}

std::uint64_t ShadowCachePolicy::version(std::uint32_t lightId) const {
    const auto it = entries_.find(lightId);
    return it == entries_.end() ? 0 : it->second.version;
}

std::vector<std::uint32_t> ShadowCachePolicy::lruOrder() const {
    return std::vector<std::uint32_t>(lru_.begin(), lru_.end());
}

void ShadowCachePolicy::touch(std::uint32_t lightId) {
    const auto it = entries_.find(lightId);
    if (it == entries_.end()) {
        return;
    }
    it->second.lastUse = ++tick_;
    lru_.remove(lightId);
    lru_.push_back(lightId);
}

std::optional<std::uint32_t> ShadowCachePolicy::evictLeastRecentlyUsed() {
    if (lru_.empty()) {
        return std::nullopt;
    }
    const std::uint32_t evicted = lru_.front();
    lru_.pop_front();
    entries_.erase(evicted);
    return evicted;
}

} // namespace indoor
