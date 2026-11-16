#pragma once

#include <memory>
#include <string>

namespace skin {

using OpenGlProc = void (*)();
using OpenGlProcLoader = OpenGlProc (*)(const char*);

struct GlContextInfo {
    std::string backend;
    std::string vendor;
    std::string renderer;
    std::string version;
    std::string glslVersion;
    int major = 0;
    int minor = 0;
};

class GlContext {
public:
    virtual ~GlContext() = default;

    GlContext(const GlContext&) = delete;
    GlContext& operator=(const GlContext&) = delete;

    static std::unique_ptr<GlContext> create(
        const std::string& requestedBackend,
        std::string& error
    );

    virtual void makeCurrent() = 0;
    virtual void shutdown() = 0;

    [[nodiscard]] const GlContextInfo& info() const { return info_; }

protected:
    GlContext() = default;

    bool loadOpenGl(OpenGlProcLoader loader, std::string& error);

    GlContextInfo info_;
};

}  // namespace skin

