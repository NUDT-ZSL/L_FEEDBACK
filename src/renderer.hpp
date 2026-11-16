#pragma once

#include "bmp.hpp"
#include "config.hpp"
#include "model.hpp"

namespace preview {

Image renderScene(const RenderConfig& config, const Model& model);

} // namespace preview
