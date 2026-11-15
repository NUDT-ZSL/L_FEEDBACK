#pragma once

#include "pipeline.hpp"
#include "scene.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scene {

struct TimingEntry {
    std::string name;
    bool enabled = true;
    double milliseconds = 0.0;
};

struct RenderResult {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
    std::vector<TimingEntry> stageTimings;
    std::vector<TimingEntry> effectTimings;
    double totalMilliseconds = 0.0;
    double prepareMilliseconds = 0.0;
    double presentMilliseconds = 0.0;
    double readbackMilliseconds = 0.0;
};

class Renderer {
public:
    Renderer(const Scene& scene, int width, int height,
             const std::string& shaderDirectory);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void applyPipeline(const PipelineConfig& config);
    RenderResult renderFrame(bool capturePixels);

    const PipelineConfig& pipeline() const;
    int width() const noexcept;
    int height() const noexcept;
    const std::string& shaderDirectory() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace scene

