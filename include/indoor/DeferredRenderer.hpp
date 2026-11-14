#pragma once

#include "indoor/RenderTypes.hpp"
#include "indoor/Scene.hpp"
#include "indoor/ShadowSystems.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace indoor {

class DeferredRenderer {
public:
    DeferredRenderer();
    ~DeferredRenderer();

    DeferredRenderer(const DeferredRenderer&) = delete;
    DeferredRenderer& operator=(const DeferredRenderer&) = delete;
    DeferredRenderer(DeferredRenderer&&) noexcept;
    DeferredRenderer& operator=(DeferredRenderer&&) noexcept;

    void initialize(const Scene& scene, const RenderSettings& settings,
                    const std::string& shaderDirectory);
    void shutdown();

    // Call after the old GL context has become unrecoverable and before the new
    // context is made current. These handles are abandoned, never deleted under
    // an unrelated context.
    void invalidateContextWithoutDelete();

    // Recreates every GPU object from CPU-persistent state and renders exactly
    // the shadow maps described by the snapshot.
    void recoverFromSnapshot(const Scene& scene, const FrameSnapshot& snapshot,
                             const RenderSettings& settings,
                             const std::string& shaderDirectory);

    void resize(const Scene& scene, int width, int height);
    void switchScene(const Scene& scene);

    FrameSnapshot prepareFrame(const Scene& scene, const Camera& camera,
                               MaterialPresetId preset, const std::vector<std::uint32_t>& dirtyLights,
                               std::uint64_t frameId);

    void render(const Scene& scene, const FrameSnapshot& snapshot, DebugView debugView);
    void blitFinalToDefaultFramebuffer() const;

    [[nodiscard]] std::vector<std::uint8_t> readFinalPixels() const;
    [[nodiscard]] const ShadowFrameStats& stats() const;
    [[nodiscard]] const FrameSnapshot& lastSnapshot() const;
    [[nodiscard]] const RenderSettings& settings() const;
    [[nodiscard]] const ShadowVersionLedger& versionLedger() const;
    [[nodiscard]] bool hasGpuResources() const;

    // Must be called on the render thread at a frame boundary. It physically
    // deletes the GL texture and drops the cache/version entry.
    bool releaseLightShadow(std::uint32_t lightId);
    void releaseAllShadowMaps();

    // Test/diagnostic API; returns zero when the light is not cache-resident.
    [[nodiscard]] unsigned int shadowTextureForLight(std::uint32_t lightId) const;
    [[nodiscard]] int shadowCacheResidentCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace indoor
