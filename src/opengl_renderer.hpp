#pragma once

#include "common.hpp"
#include "gl_loader.hpp"
#include "material.hpp"

#include <vector>

namespace cr {

Image renderOpenGLDeferredWithCurrentContext(const MaterialLibrary& materials,
                                             const Camera& camera,
                                             const std::vector<Light>& lights,
                                             const gl::GlProcLoader& loader);

}  // namespace cr

