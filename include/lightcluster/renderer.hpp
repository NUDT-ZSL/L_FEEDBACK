#pragma once

#include "lightcluster/buffer_sync.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lightcluster {

class Renderer {
public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void initialize(bool headless);
    void upload(const GpuBufferImage& image, bool grew);
    void render(const GpuBufferImage& image);
    [[nodiscard]] std::vector<std::uint8_t> readback(
        std::uint32_t usedBytes);
    void corruptCountForDiagnostic();
    void pollEvents();
    void swapBuffers(bool headless);
    [[nodiscard]] bool shouldClose() const;
    [[nodiscard]] int width() const;
    [[nodiscard]] int height() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lightcluster
