#include "terrain.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>

namespace ct {
namespace {

struct VertexKey {
    double px;
    double py;
    double pz;
    double nx;
    double ny;
    double nz;
    double u;
    double v;
    std::uint32_t material;

    bool operator==(const VertexKey& rhs) const {
        return px == rhs.px && py == rhs.py && pz == rhs.pz && nx == rhs.nx &&
               ny == rhs.ny && nz == rhs.nz && u == rhs.u && v == rhs.v &&
               material == rhs.material;
    }
};

struct VertexKeyHash {
    std::size_t operator()(const VertexKey& key) const {
        std::size_t seed = 0;
        const auto combine = [&seed](double value) {
            const std::size_t h = std::hash<double>{}(value);
            seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };
        combine(key.px);
        combine(key.py);
        combine(key.pz);
        combine(key.nx);
        combine(key.ny);
        combine(key.nz);
        combine(key.u);
        combine(key.v);
        seed ^= static_cast<std::size_t>(key.material) + 0x9e3779b97f4a7c15ULL +
                (seed << 6) + (seed >> 2);
        return seed;
    }
};

TerrainVertex makeVertex(const HeightField& heightfield, const AppConfig& config, double x,
                         double z) {
    TerrainVertex vertex;
    vertex.position = {x, heightfield.sampleBilinear(x, z), z};
    vertex.normal = heightfield.normalAt(x, z);
    vertex.uv = {x * config.texture.scale, z * config.texture.scale};
    vertex.material = heightfield.materialAt(x, z);
    return vertex;
}

VertexKey makeKey(const TerrainVertex& vertex) {
    return {
        vertex.position.x,
        vertex.position.y,
        vertex.position.z,
        vertex.normal.x,
        vertex.normal.y,
        vertex.normal.z,
        vertex.uv.x,
        vertex.uv.y,
        vertex.material,
    };
}

class MeshBuilder {
public:
    explicit MeshBuilder(TerrainMesh& mesh) : mesh_(mesh) {}

    std::uint32_t addVertex(const TerrainVertex& vertex) {
        const VertexKey key = makeKey(vertex);
        const auto it = lookup_.find(key);
        if (it != lookup_.end()) {
            return it->second;
        }
        if (mesh_.vertices.size() >= std::numeric_limits<std::uint32_t>::max()) {
            return 0;
        }
        const std::uint32_t index = static_cast<std::uint32_t>(mesh_.vertices.size());
        mesh_.vertices.push_back(vertex);
        lookup_.emplace(key, index);
        return index;
    }

    bool addTriangle(std::uint32_t a, std::uint32_t b, std::uint32_t c, int lod,
                     bool transition) {
        if (a == b || b == c || c == a) {
            return false;
        }
        const Vec3& pa = mesh_.vertices[a].position;
        const Vec3& pb = mesh_.vertices[b].position;
        const Vec3& pc = mesh_.vertices[c].position;
        const Vec3 ab = pb - pa;
        const Vec3 ac = pc - pa;
        if (length(cross(ab, ac)) < 1e-11) {
            return false;
        }

        std::array<std::uint32_t, 3> sorted{a, b, c};
        std::sort(sorted.begin(), sorted.end());
        if (!triangle_keys_.insert(sorted).second) {
            return false;
        }
        mesh_.triangles.push_back({a, b, c, lod, transition});
        if (transition) {
            ++mesh_.transition_triangles;
        } else {
            ++mesh_.terrain_triangles;
        }
        return true;
    }

private:
    TerrainMesh& mesh_;
    std::unordered_map<VertexKey, std::uint32_t, VertexKeyHash> lookup_;
    std::set<std::array<std::uint32_t, 3>> triangle_keys_;
};

std::vector<double> makeAxisSamples(int start, int end, int step) {
    std::vector<double> values;
    for (int value = start; value < end; value += step) {
        values.push_back(static_cast<double>(value));
    }
    if (values.empty() || values.back() != static_cast<double>(end)) {
        values.push_back(static_cast<double>(end));
    }
    return values;
}

struct ChainPoint {
    double x;
    double z;
    double parameter;
};

void addTransitionStrip(MeshBuilder& builder, const HeightField& heightfield,
                        const AppConfig& config, const std::vector<double>& coarse,
                        const std::vector<double>& fine, bool horizontal, double fixed) {
    const std::size_t m = coarse.size();
    const std::size_t n = fine.size();
    std::vector<std::uint32_t> coarse_indices(m);
    std::vector<std::uint32_t> fine_indices(n);
    for (std::size_t i = 0; i < m; ++i) {
        const double x = horizontal ? coarse[i] : fixed;
        const double z = horizontal ? fixed : coarse[i];
        coarse_indices[i] = builder.addVertex(makeVertex(heightfield, config, x, z));
    }
    for (std::size_t j = 0; j < n; ++j) {
        const double x = horizontal ? fine[j] : fixed;
        const double z = horizontal ? fixed : fine[j];
        fine_indices[j] = builder.addVertex(makeVertex(heightfield, config, x, z));
    }

    std::size_t i = 0;
    std::size_t j = 0;
    const double inf = std::numeric_limits<double>::infinity();
    while (i + 1 < m || j + 1 < n) {
        const double next_coarse =
            i + 1 < m ? (coarse[i + 1] - coarse[0]) / (coarse.back() - coarse[0]) : inf;
        const double next_fine =
            j + 1 < n ? (fine[j + 1] - fine[0]) / (fine.back() - fine[0]) : inf;
        if (next_coarse <= next_fine) {
            builder.addTriangle(coarse_indices[i], coarse_indices[i + 1], fine_indices[j], 0,
                                true);
            ++i;
        } else {
            builder.addTriangle(coarse_indices[i], fine_indices[j + 1], fine_indices[j], 0,
                                true);
            ++j;
        }
    }
}

struct BlockCandidate {
    int bx;
    int bz;
    int x0;
    int z0;
    int x1;
    int z1;
    int lod;
    double distance;
    double area_fraction;
};

BlockCandidate selectBlockLod(int bx, int bz, int x0, int z0, int x1, int z1,
                              const HeightField& heightfield, const AppConfig& config) {
    const double center_x = (static_cast<double>(x0) + static_cast<double>(x1)) * 0.5;
    const double center_z = (static_cast<double>(z0) + static_cast<double>(z1)) * 0.5;
    double min_height = std::numeric_limits<double>::infinity();
    double max_height = -std::numeric_limits<double>::infinity();
    for (int z = z0; z <= z1; z += std::max(1, (z1 - z0) / 4)) {
        for (int x = x0; x <= x1; x += std::max(1, (x1 - x0) / 4)) {
            const double h = heightfield.sampleBilinear(x, z);
            min_height = std::min(min_height, h);
            max_height = std::max(max_height, h);
        }
    }
    const Vec3 center{center_x, (min_height + max_height) * 0.5, center_z};
    const Vec3 to_center = center - config.camera.position;
    const double distance = std::max(length(to_center), config.camera.near_plane);
    const double half_x = (x1 - x0) * 0.5;
    const double half_z = (z1 - z0) * 0.5;
    const double half_y = (max_height - min_height) * 0.5;
    const double radius = std::sqrt(half_x * half_x + half_y * half_y + half_z * half_z);
    const double tan_half_fov =
        std::tan(config.camera.fov_degrees * kPi / 180.0 * 0.5);
    const double pixels_per_unit =
        (static_cast<double>(config.render.height) * 0.5) / (tan_half_fov * distance);
    const double radius_pixels = radius * pixels_per_unit;
    const double screen_area =
        static_cast<double>(config.render.width) * static_cast<double>(config.render.height);
    double area_fraction = (kPi * radius_pixels * radius_pixels) / screen_area;
    area_fraction = std::clamp(area_fraction, 0.0, 4.0);

    int lod = 2;
    if (distance <= config.lod.lod0_distance ||
        area_fraction >= config.lod.lod0_screen_area) {
        lod = 0;
    } else if (distance <= config.lod.lod1_distance ||
               area_fraction >= config.lod.lod1_screen_area) {
        lod = 1;
    }
    return {bx, bz, x0, z0, x1, z1, lod, distance, area_fraction};
}

void addBlockSurface(MeshBuilder& builder, const HeightField& heightfield,
                     const AppConfig& config, const BlockCandidate& block) {
    const int step = 1 << block.lod;
    const std::vector<double> xs = makeAxisSamples(block.x0, block.x1, step);
    const std::vector<double> zs = makeAxisSamples(block.z0, block.z1, step);
    std::vector<std::vector<std::uint32_t>> indices(
        zs.size(), std::vector<std::uint32_t>(xs.size()));
    for (std::size_t iz = 0; iz < zs.size(); ++iz) {
        for (std::size_t ix = 0; ix < xs.size(); ++ix) {
            indices[iz][ix] = builder.addVertex(
                makeVertex(heightfield, config, xs[ix], zs[iz]));
        }
    }
    for (std::size_t iz = 0; iz + 1 < zs.size(); ++iz) {
        for (std::size_t ix = 0; ix + 1 < xs.size(); ++ix) {
            const std::uint32_t v00 = indices[iz][ix];
            const std::uint32_t v10 = indices[iz][ix + 1];
            const std::uint32_t v01 = indices[iz + 1][ix];
            const std::uint32_t v11 = indices[iz + 1][ix + 1];
            builder.addTriangle(v00, v01, v10, block.lod, false);
            builder.addTriangle(v10, v01, v11, block.lod, false);
        }
    }
}

}  // namespace

bool TerrainVertex::operator==(const TerrainVertex& rhs) const {
    return position.x == rhs.position.x && position.y == rhs.position.y &&
           position.z == rhs.position.z && normal.x == rhs.normal.x &&
           normal.y == rhs.normal.y && normal.z == rhs.normal.z && uv.x == rhs.uv.x &&
           uv.y == rhs.uv.y && material == rhs.material;
}

TerrainMesh buildTerrain(const HeightField& heightfield, const AppConfig& config) {
    TerrainMesh mesh;
    MeshBuilder builder(mesh);

    const int cells = heightfield.size() - 1;
    const int block_size = config.lod.block_size;
    const int block_stride = block_size - 1;
    const int block_columns = (cells + block_stride - 1) / block_stride;
    const int block_rows = block_columns;
    std::vector<BlockCandidate> blocks;
    blocks.reserve(static_cast<std::size_t>(block_columns) * block_rows);

    for (int bz = 0; bz < block_rows; ++bz) {
        const int z0 = bz * block_stride;
        const int z1 = std::min(z0 + block_stride, cells);
        for (int bx = 0; bx < block_columns; ++bx) {
            const int x0 = bx * block_stride;
            const int x1 = std::min(x0 + block_stride, cells);
            BlockCandidate block =
                selectBlockLod(bx, bz, x0, z0, x1, z1, heightfield, config);
            const int step = 1 << block.lod;
            const std::uint64_t nx =
                static_cast<std::uint64_t>((x1 - x0 + step - 1) / step + 1);
            const std::uint64_t nz =
                static_cast<std::uint64_t>((z1 - z0 + step - 1) / step + 1);
            const std::uint64_t cell_count = (nx > 0 ? nx - 1 : 0) * (nz > 0 ? nz - 1 : 0);
            BlockInfo info;
            info.block_x = bx;
            info.block_z = bz;
            info.x0 = x0;
            info.z0 = z0;
            info.x1 = x1;
            info.z1 = z1;
            info.lod = block.lod;
            info.distance = block.distance;
            info.projected_area_fraction = block.area_fraction;
            info.triangles = cell_count * 2;
            mesh.blocks.push_back(info);
            blocks.push_back(block);
        }
    }

    for (const BlockCandidate& block : blocks) {
        addBlockSurface(builder, heightfield, config, block);
    }

    const auto block_at = [&](int bx, int bz) -> const BlockCandidate& {
        return blocks[static_cast<std::size_t>(bz) * block_columns + bx];
    };

    for (int bz = 0; bz < block_rows; ++bz) {
        for (int bx = 0; bx < block_columns; ++bx) {
            const BlockCandidate& block = block_at(bx, bz);
            if (bx + 1 < block_columns) {
                const BlockCandidate& right = block_at(bx + 1, bz);
                if (block.lod != right.lod) {
                    const int coarse_lod = std::min(block.lod, right.lod);
                    const int fine_lod = std::max(block.lod, right.lod);
                    const double boundary_x = static_cast<double>(block.x1);
                    const int start = block.z0;
                    const int end = block.z1;
                    const std::vector<double> coarse = makeAxisSamples(
                        start, end, 1 << coarse_lod);
                    const std::vector<double> fine = makeAxisSamples(start, end, 1 << fine_lod);
                    addTransitionStrip(builder, heightfield, config, coarse, fine, false,
                                       boundary_x);
                }
            }
            if (bz + 1 < block_rows) {
                const BlockCandidate& front = block_at(bx, bz + 1);
                if (block.lod != front.lod) {
                    const int coarse_lod = std::min(block.lod, front.lod);
                    const int fine_lod = std::max(block.lod, front.lod);
                    const double boundary_z = static_cast<double>(block.z1);
                    const int start = block.x0;
                    const int end = block.x1;
                    const std::vector<double> coarse = makeAxisSamples(
                        start, end, 1 << coarse_lod);
                    const std::vector<double> fine = makeAxisSamples(start, end, 1 << fine_lod);
                    addTransitionStrip(builder, heightfield, config, coarse, fine, true,
                                       boundary_z);
                }
            }
        }
    }
    return mesh;
}

TerrainBuildStats summarizeTerrain(const TerrainMesh& mesh) {
    TerrainBuildStats stats;
    int max_x = -1;
    int max_z = -1;
    for (const BlockInfo& block : mesh.blocks) {
        max_x = std::max(max_x, block.block_x);
        max_z = std::max(max_z, block.block_z);
        if (block.lod == 0) {
            ++stats.blocks_lod0;
            stats.triangles_lod0 += block.triangles;
        } else if (block.lod == 1) {
            ++stats.blocks_lod1;
            stats.triangles_lod1 += block.triangles;
        } else {
            ++stats.blocks_lod2;
            stats.triangles_lod2 += block.triangles;
        }
    }
    stats.block_columns = max_x + 1;
    stats.block_rows = max_z + 1;
    return stats;
}

TerrainTopologyStats analyzeTopology(const TerrainMesh& mesh) {
    TerrainTopologyStats stats;
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> edge_counts;
    for (const TerrainTriangle& triangle : mesh.triangles) {
        const std::array<std::pair<std::uint32_t, std::uint32_t>, 3> edges{{
            {std::min(triangle.a, triangle.b), std::max(triangle.a, triangle.b)},
            {std::min(triangle.b, triangle.c), std::max(triangle.b, triangle.c)},
            {std::min(triangle.c, triangle.a), std::max(triangle.c, triangle.a)},
        }};
        for (const auto& edge : edges) {
            ++edge_counts[edge];
        }
    }
    stats.edges = edge_counts.size();

    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    for (const TerrainVertex& vertex : mesh.vertices) {
        min_x = std::min(min_x, vertex.position.x);
        max_x = std::max(max_x, vertex.position.x);
        min_z = std::min(min_z, vertex.position.z);
        max_z = std::max(max_z, vertex.position.z);
    }
    const auto on_outer_boundary = [&](const TerrainVertex& a, const TerrainVertex& b) {
        const bool same_left = a.position.x == min_x && b.position.x == min_x;
        const bool same_right = a.position.x == max_x && b.position.x == max_x;
        const bool same_back = a.position.z == min_z && b.position.z == min_z;
        const bool same_front = a.position.z == max_z && b.position.z == max_z;
        return same_left || same_right || same_back || same_front;
    };

    for (const auto& item : edge_counts) {
        if (item.second == 1) {
            ++stats.boundary_edges;
            const TerrainVertex& a = mesh.vertices[item.first.first];
            const TerrainVertex& b = mesh.vertices[item.first.second];
            if (!on_outer_boundary(a, b)) {
                ++stats.hanging_edges;
            }
        } else if (item.second == 2) {
            ++stats.manifold_edges;
        } else {
            ++stats.non_manifold_edges;
        }
    }
    return stats;
}

bool validateTerrainMesh(const TerrainMesh& mesh, std::string& error) {
    if (mesh.vertices.empty() || mesh.triangles.empty()) {
        error = "terrain mesh is empty";
        return false;
    }
    std::set<std::array<std::uint32_t, 3>> triangle_set;
    for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
        const TerrainTriangle& triangle = mesh.triangles[i];
        if (triangle.a >= mesh.vertices.size() || triangle.b >= mesh.vertices.size() ||
            triangle.c >= mesh.vertices.size()) {
            error = "terrain triangle index is out of range";
            return false;
        }
        if (triangle.a == triangle.b || triangle.b == triangle.c ||
            triangle.c == triangle.a) {
            error = "terrain mesh contains a triangle with a repeated vertex";
            return false;
        }
        std::array<std::uint32_t, 3> sorted{
            triangle.a, triangle.b, triangle.c};
        std::sort(sorted.begin(), sorted.end());
        if (!triangle_set.insert(sorted).second) {
            error = "terrain mesh contains duplicate triangles";
            return false;
        }
    }
    const TerrainTopologyStats topology = analyzeTopology(mesh);
    if (topology.non_manifold_edges > 0) {
        error = "terrain mesh contains a non-manifold edge used by more than two "
                "triangles";
        return false;
    }
    if (topology.hanging_edges > 0) {
        error = "terrain mesh contains an interior hanging edge";
        return false;
    }
    return true;
}

}  // namespace ct
