#pragma once

#include "core/animation.hpp"
#include "core/scene.hpp"

#include <cstdint>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#else
struct HWND__;
using HWND = HWND__*;
#endif

namespace uid166 {

struct LiveFrameStats {
    uint64_t frame = 0;
    uint32_t frustumVisible = 0;
    uint32_t occluded = 0;
    uint32_t visible = 0;
    uint32_t csmUpdated = 0;
    uint32_t localShadowUpdated = 0;
    uint32_t hardwareQueries = 0;
    uint32_t hardwareResults = 0;
    uint32_t hardwareOccluded = 0;
    uint32_t recordedCommands = 0;
    uint32_t staticBatches = 0;
    uint32_t occlusionQueryDraws = 0;
    uint32_t totalDrawCalls = 0;
    uint64_t recordingMicros = 0;
    bool recordingBudgetExceeded = false;
    bool cameraFast = false;
    float gpuFrameMs = 0.0f;
    uint32_t qualityLevel = 0;
    float renderScale = 1.0f;
    float shadowScale = 1.0f;
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t shadowResolution = 0;
};

class D3D11Renderer {
public:
    static std::unique_ptr<D3D11Renderer> create(HWND window, uint32_t width, uint32_t height,
                                                 SceneDesc scene, CharacterAsset character);
    ~D3D11Renderer();

    D3D11Renderer(const D3D11Renderer&) = delete;
    D3D11Renderer& operator=(const D3D11Renderer&) = delete;

    void resize(uint32_t width, uint32_t height);
    [[nodiscard]] LiveFrameStats renderFrame(uint64_t frame, float deltaSeconds,
                                             const Camera& camera);
    [[nodiscard]] bool deviceRemoved() const;

private:
    struct Impl;
    explicit D3D11Renderer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace uid166

