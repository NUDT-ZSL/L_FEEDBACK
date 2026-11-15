#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace vlr {

enum class Injection {
    None,
    Shader,
    Allocation,
    Descriptor,
    DeviceLost
};

struct RendererOptions {
    std::filesystem::path scenePath;
    std::optional<std::filesystem::path> outputPng;
    std::optional<std::filesystem::path> statsJson;
    std::optional<std::filesystem::path> referencePng;
    std::optional<std::filesystem::path> shaderDirectory;
    std::optional<std::filesystem::path> recoveryDirectory;
    std::optional<float> volumeStepOverride;
    std::optional<int> volumeMaxStepsOverride;
    Injection injection = Injection::None;
    bool noReference = false;
    bool clearRecovery = false;
    bool enableValidation = true;
    int frameCount = 2;
};

class Renderer {
public:
    explicit Renderer(RendererOptions options);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void run();
    void cleanup();

    [[nodiscard]] int liveResourceCount() const noexcept;
    [[nodiscard]] long long liveAllocatorCount() const noexcept;
    [[nodiscard]] bool deviceWasLost() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vlr
