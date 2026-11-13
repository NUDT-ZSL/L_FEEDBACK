#pragma once

#include "image.hpp"
#include "scene.hpp"

namespace lfb {

Image renderFrame(const Scene& scene, const FrameDesc& frame);

}  // namespace lfb
