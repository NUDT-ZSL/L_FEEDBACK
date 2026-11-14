#include "lightcluster/scene.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace lightcluster {

SceneManager::SceneManager(std::vector<Light> initialLights)
    : working_(std::move(initialLights)) {
    std::uint32_t highestId = 0;
    for (Light& light : working_) {
        if (light.id == 0) {
            light.id = ++highestId;
        } else {
            highestId = std::max(highestId, light.id);
        }
    }
    nextId_ = highestId + 1;
    publishLocked();
}

std::uint32_t SceneManager::addLight(Light light) {
    std::unique_lock lock(mutex_);
    if (light.id != 0 &&
        std::any_of(working_.begin(), working_.end(), [&](const Light& existing) {
            return existing.id == light.id;
        })) {
        return 0;
    }

    if (light.id == 0) {
        light.id = nextId_++;
    } else {
        nextId_ = std::max(nextId_, light.id + 1);
    }
    working_.push_back(light);
    ++version_;
    publishLocked();
    return light.id;
}

bool SceneManager::removeLight(std::uint32_t id) {
    std::unique_lock lock(mutex_);
    const auto it = std::find_if(
        working_.begin(), working_.end(),
        [id](const Light& light) { return light.id == id; });
    if (it == working_.end()) {
        return false;
    }
    working_.erase(it);
    ++version_;
    publishLocked();
    return true;
}

bool SceneManager::updateLight(
    std::uint32_t id,
    const std::function<void(Light&)>& update) {
    std::unique_lock lock(mutex_);
    const auto it = std::find_if(
        working_.begin(), working_.end(),
        [id](const Light& light) { return light.id == id; });
    if (it == working_.end()) {
        return false;
    }
    update(*it);
    ++version_;
    publishLocked();
    return true;
}

bool SceneManager::contains(std::uint32_t id) const {
    std::shared_lock lock(mutex_);
    return std::any_of(
        working_.begin(), working_.end(),
        [id](const Light& light) { return light.id == id; });
}

std::optional<Light> SceneManager::findLight(std::uint32_t id) const {
    std::shared_lock lock(mutex_);
    const auto it = std::find_if(
        working_.begin(), working_.end(),
        [id](const Light& light) { return light.id == id; });
    if (it == working_.end()) {
        return std::nullopt;
    }
    return *it;
}

std::size_t SceneManager::lightCount() const {
    std::shared_lock lock(mutex_);
    return working_.size();
}

std::uint64_t SceneManager::version() const {
    std::shared_lock lock(mutex_);
    return version_;
}

SceneSnapshot SceneManager::snapshot() const {
    std::shared_lock lock(mutex_);
    return snapshotBuffers_[activeSnapshot_];
}

SceneSnapshot SceneManager::saveSnapshot() const {
    return snapshot();
}

std::uint64_t SceneManager::restore(const SceneSnapshot& saved) {
    std::unique_lock lock(mutex_);
    if (version_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("scene version exhausted");
    }
    const std::uint64_t nextVersion =
        std::max(version_, saved.version) + static_cast<std::uint64_t>(1);
    working_ = saved.lights;
    version_ = nextVersion;
    for (const Light& light : working_) {
        nextId_ = std::max(nextId_, light.id + 1);
    }
    publishLocked();
    return version_;
}

void SceneManager::publishLocked() {
    const std::size_t back = 1 - activeSnapshot_;
    snapshotBuffers_[back].version = version_;
    snapshotBuffers_[back].lights = working_;
    activeSnapshot_ = back;
}

}  // namespace lightcluster
