#include "indoor/GLUtil.hpp"

#include <glad/gl.h>

#include <stdexcept>

namespace indoor {

std::thread::id RenderThreadGuard::renderThreadId_ = std::thread::id{};

void RenderThreadGuard::bindCurrentThread() {
    renderThreadId_ = std::this_thread::get_id();
}

bool RenderThreadGuard::isRenderThread() {
    return std::this_thread::get_id() == renderThreadId_;
}

void RenderThreadGuard::requireRenderThread(const char* operation) {
    if (!isRenderThread()) {
        throw std::runtime_error(std::string("GPU operation outside render thread: ") +
                                 operation);
    }
}

void checkGlError(const char* operation) {
    const GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        throw std::runtime_error(std::string("OpenGL error 0x") +
                                 std::to_string(error) + " at " + operation);
    }
}

} // namespace indoor
