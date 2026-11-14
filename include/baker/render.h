#pragma once

#include <functional>
#include <cstdint>

#include "baker/model.h"

namespace baker {

struct RenderOptions {
    float time = 0.0f;
};

using LayerComplete = std::function<void(std::uint32_t completed_layers,
                                         const PixelBuffer&,
                                         const Stats&)>;

PixelBuffer make_empty_buffer(std::uint32_t width, std::uint32_t height);
void render_scene(const Scene& scene,
                  const RenderOptions& options,
                  std::uint32_t first_layer,
                  PixelBuffer& state,
                  Stats& stats,
                  const LayerComplete& on_layer_complete);

std::vector<std::uint8_t> encode_output(const PixelBuffer& pixels);

}  // namespace baker
