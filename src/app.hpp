#pragma once

#include "terrain/cli.hpp"
#include "terrain/render_data.hpp"
#include "terrain/terrain.hpp"

#include <iosfwd>

namespace terrain {

int runSoftwareApp(const Options& options);

#if TERRAIN_HAS_GL
int runGlApp(const Options& options);
#endif

void printStatsJson(
    std::ostream& output,
    const SceneFrame& frame,
    const Terrain& terrain,
    DebugView view);

}  // namespace terrain

