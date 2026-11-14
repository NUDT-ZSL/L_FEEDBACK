#pragma once

#include "common.hpp"
#include "material.hpp"

#include <string>
#include <vector>

namespace cr {

enum class RenderBackend {
    Auto,
    Cpu,
    OpenGL,
};

RenderBackend parseRenderBackend(const std::string& value);
const char* renderBackendName(RenderBackend backend);

Image renderCpuDeferred(const MaterialLibrary& materials, const Camera& camera,
                        const std::vector<Light>& lights);

#ifdef CHARACTER_RENDER_HAVE_OPENGL
Image renderOpenGLDeferred(const MaterialLibrary& materials, const Camera& camera,
                           const std::vector<Light>& lights);
#endif

}  // namespace cr

