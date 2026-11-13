#pragma once

#include "terrain/cli.hpp"
#include "terrain/ppm.hpp"
#include "terrain/render_data.hpp"
#include "terrain/terrain.hpp"

namespace terrain {

class SoftwareRenderer {
public:
    SoftwareRenderer(const Terrain& terrain, int width, int height);

    ImageRgb render(const SceneFrame& frame, DebugView view) const;

private:
    const Terrain& terrain_;
    int width_;
    int height_;
};

}  // namespace terrain

