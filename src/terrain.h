#pragma once

#include "config.h"
#include "heightfield.h"
#include "math.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ct {

struct TerrainVertex {
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
    std::uint32_t material = 1;

    bool operator==(const TerrainVertex& rhs) const;
};

struct TerrainTriangle {
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    std::uint32_t c = 0;
    int lod = 0;
    bool transition = false;
};

struct BlockInfo {
    int block_x = 0;
    int block_z = 0;
    int x0 = 0;
    int z0 = 0;
    int x1 = 0;
    int z1 = 0;
    int lod = 0;
    double distance = 0.0;
    double projected_area_fraction = 0.0;
    std::uint64_t triangles = 0;
};

struct TerrainMesh {
    std::vector<TerrainVertex> vertices;
    std::vector<TerrainTriangle> triangles;
    std::vector<BlockInfo> blocks;
    std::uint64_t terrain_triangles = 0;
    std::uint64_t transition_triangles = 0;
};

struct TerrainBuildStats {
    int block_columns = 0;
    int block_rows = 0;
    std::uint64_t blocks_lod0 = 0;
    std::uint64_t blocks_lod1 = 0;
    std::uint64_t blocks_lod2 = 0;
    std::uint64_t triangles_lod0 = 0;
    std::uint64_t triangles_lod1 = 0;
    std::uint64_t triangles_lod2 = 0;
};

struct TerrainTopologyStats {
    std::uint64_t edges = 0;
    std::uint64_t boundary_edges = 0;
    std::uint64_t manifold_edges = 0;
    std::uint64_t non_manifold_edges = 0;
    std::uint64_t hanging_edges = 0;
};

TerrainMesh buildTerrain(const HeightField& heightfield, const AppConfig& config);
TerrainBuildStats summarizeTerrain(const TerrainMesh& mesh);
TerrainTopologyStats analyzeTopology(const TerrainMesh& mesh);
bool validateTerrainMesh(const TerrainMesh& mesh, std::string& error);

}  // namespace ct
