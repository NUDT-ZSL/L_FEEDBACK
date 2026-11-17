#pragma once

#include "config.h"
#include "heightfield.h"
#include "image.h"
#include "terrain.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ct {

struct RenderStats {
    int width = 0;
    int height = 0;
    int internal_width = 0;
    int internal_height = 0;
    int supersampling = 1;

    std::uint64_t triangles_total = 0;
    std::uint64_t triangles_frustum_culled = 0;
    std::uint64_t triangles_partially_clipped = 0;
    std::uint64_t triangles_after_frustum_clip = 0;
    std::uint64_t triangles_backface_culled = 0;
    std::uint64_t triangles_after_clipping = 0;
    std::uint64_t triangles_rasterized = 0;

    std::uint64_t covered_samples = 0;
    std::uint64_t anisotropic_samples = 0;
    int max_anisotropic_samples = 0;
    double average_anisotropic_samples = 0.0;

    std::size_t gbuffer_bytes_per_pixel = 0;
    std::size_t gbuffer_pixel_count = 0;
    std::size_t gbuffer_bytes = 0;
};

bool renderTerrain(const HeightField& heightfield, const TerrainMesh& mesh,
                   const AppConfig& config, ColorBuffer& output, RenderStats& stats,
                   std::string& error);

}  // namespace ct
