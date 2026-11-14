#pragma once

#include "lightcluster/types.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <shared_mutex>
#include <vector>

namespace lightcluster {

class SceneManager {
public:
    explicit SceneManager(std::vector<Light> initialLights);

    SceneManager(const SceneManager&) = delete;
    SceneManager& operator=(const SceneManager&) = delete;

    [[nodiscard]] std::uint32_t addLight(Light light);
    [[nodiscard]] bool removeLight(std::uint32_t id);
    [[nodiscard]] bool updateLight(
        std::uint32_t id,
        const std::function<void(Light&)>& update);

    [[nodiscard]] bool contains(std::uint32_t id) const;
    [[nodiscard]] std::optional<Light> findLight(std::uint32_t id) const;
    [[nodiscard]] std::size_t lightCount() const;
    [[nodiscard]] std::uint64_t version() const;

    [[nodiscard]] SceneSnapshot snapshot() const;
    [[nodiscard]] SceneSnapshot saveSnapshot() const;

    // Restoring is itself a mutation. The published version is always greater
    // than both the live version at restore time and the saved snapshot version.
    [[nodiscard]] std::uint64_t restore(const SceneSnapshot& saved);

private:
    void publishLocked();

    mutable std::shared_mutex mutex_;
    std::vector<Light> working_;
    std::uint64_t version_ = 1;
    std::uint32_t nextId_ = 1;
    std::array<SceneSnapshot, 2> snapshotBuffers_{};
    std::size_t activeSnapshot_ = 0;
};

}  // namespace lightcluster
