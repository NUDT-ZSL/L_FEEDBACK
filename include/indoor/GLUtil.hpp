#pragma once

#include <string>
#include <thread>

namespace indoor {

class RenderThreadGuard {
public:
    static void bindCurrentThread();
    static bool isRenderThread();
    static void requireRenderThread(const char* operation);

private:
    static std::thread::id renderThreadId_;
};

void checkGlError(const char* operation);

} // namespace indoor
