#pragma once

#include "gl.hpp"

#include <string>

namespace scene {

class OffscreenContext {
public:
    OffscreenContext();
    ~OffscreenContext();
    OffscreenContext(const OffscreenContext&) = delete;
    OffscreenContext& operator=(const OffscreenContext&) = delete;

    GLFWwindow* window() const noexcept { return window_; }
    const std::string& rendererName() const noexcept { return rendererName_; }
    const std::string& versionString() const noexcept { return versionString_; }

private:
    GLFWwindow* window_ = nullptr;
    bool glfwInitialized_ = false;
    std::string rendererName_;
    std::string versionString_;
};

} // namespace scene

