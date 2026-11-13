#include "terrain/cache.hpp"
#include "terrain/camera_path.hpp"
#include "terrain/lod.hpp"
#include "terrain/ppm.hpp"
#include "terrain/render_data.hpp"
#include "terrain/seams.hpp"
#include "terrain/software_renderer.hpp"
#include "terrain/terrain.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ \
                      << "  " #condition "\n"; \
            ++failures; \
        } \
    } while (false)

terrain::Camera testCamera() {
    terrain::Camera camera;
    camera.position = {256.0f, 220.0f, -80.0f};
    camera.yawDegrees = -12.0f;
    camera.pitchDegrees = -20.0f;
    return camera;
}

void testTerrainDeterminism() {
    const terrain::TerrainParameters parameters;
    terrain::Terrain first(parameters);
    terrain::Terrain second(parameters);
    CHECK(first.heights().size() == 512u * 512u);
    CHECK(first.heights() == second.heights());
    CHECK(first.heightChecksum() == second.heightChecksum());
    CHECK(first.version() == second.version());
    CHECK(first.maxDepth() == 5);
    CHECK((terrain::kPatchQuads << first.maxDepth()) >=
          terrain::kTerrainGridSize - 1);

    for (int z = 0; z < 8; ++z) {
        for (int x = 0; x < 8; ++x) {
            const std::size_t index =
                static_cast<std::size_t>(z * 512 + x);
            const float sampled = first.sampleHeight(
                static_cast<float>(x),
                static_cast<float>(z));
            CHECK(std::abs(sampled - first.heights()[index]) < 1.0e-5f);
        }
    }

    terrain::TerrainParameters changed = parameters;
    changed.seed += 1u;
    terrain::Terrain third(changed);
    CHECK(third.heightChecksum() != first.heightChecksum());
    CHECK(third.version() != first.version());
}

void testCameraFrustum() {
    const terrain::Camera camera = testCamera();
    const terrain::Frustum frustum = camera.frustum(16.0f / 9.0f);
    const terrain::Aabb visible{
        {195.0f, 120.0f, 50.0f},
        {245.0f, 175.0f, 125.0f},
    };
    const terrain::Aabb behind{
        {240.0f, 200.0f, -150.0f},
        {280.0f, 250.0f, -100.0f},
    };
    CHECK(frustum.intersects(visible));
    CHECK(!frustum.intersects(behind));
}

void testSelectionAndStitching() {
    terrain::Terrain terrain{terrain::TerrainParameters{}};
    terrain::LodSelector selector(terrain, 16.0f);
    terrain::SeamSolver seams(terrain);
    terrain::NodeStateCache cache(2048);

    const terrain::Camera camera = testCamera();
    terrain::SceneFrame frame = terrain::buildSceneFrame(
        terrain,
        selector,
        seams,
        cache,
        camera,
        960,
        540,
        terrain::DebugView::Seam,
        7,
        0.7f);

    CHECK(!frame.selection.leaves.empty());
    CHECK(frame.seams.size() == frame.selection.leaves.size());
    CHECK(frame.stats.triangles ==
          static_cast<uint64_t>(frame.selection.leaves.size()) * 512ull);
    CHECK(frame.stats.drawCalls >= 1);
    CHECK(frame.stats.drawCalls <=
          static_cast<uint32_t>(terrain.maxDepth() + 1));
    CHECK(frame.stats.seamSegments > 0);
    CHECK(frame.stats.maxStitchResidual < 1.0e-3f);
    CHECK(frame.stats.stitchVertices > 0);

    for (std::size_t i = 0; i < frame.selection.leaves.size(); ++i) {
        const terrain::LeafNode& leaf = frame.selection.leaves[i];
        CHECK(leaf.key.depth <= terrain.maxDepth());
        CHECK(leaf.x1 > leaf.x0);
        CHECK(leaf.z1 > leaf.z0);
        for (const auto& edgeSegments : frame.seams[i].edgeSegments) {
            CHECK(edgeSegments.size() <= 8);
            for (const terrain::StitchSegment& segment : edgeSegments) {
                const terrain::LeafNode& neighbor =
                    frame.selection.leaves[segment.neighborLeafIndex];
                CHECK(neighbor.key.depth < leaf.key.depth);
                CHECK(segment.t0 >= -1.0e-5f);
                CHECK(segment.t1 <= 1.0f + 1.0e-5f);
                CHECK(segment.t1 >= segment.t0);
            }
        }

        const terrain::NodeCacheEntry* entry =
            cache.find(leaf.key.encode(), terrain.version());
        CHECK(entry != nullptr);
        if (entry != nullptr) {
            CHECK(entry->stitchOffsets.size() ==
                  frame.seams[i].stats.stitchVertices);
        }
    }

    for (std::size_t i = 0; i < frame.selection.leaves.size(); ++i) {
        for (std::size_t j = i + 1; j < frame.selection.leaves.size(); ++j) {
            const auto& a = frame.selection.leaves[i];
            const auto& b = frame.selection.leaves[j];
            const bool separated =
                a.x1 <= b.x0 + 1.0e-4f || b.x1 <= a.x0 + 1.0e-4f ||
                a.z1 <= b.z0 + 1.0e-4f || b.z1 <= a.z0 + 1.0e-4f;
            CHECK(separated);
        }
    }

    const std::vector<terrain::SeamResult> again =
        seams.solve(frame.selection, 8, cache);
    CHECK(again.size() == frame.seams.size());
    for (std::size_t i = 0; i < again.size(); ++i) {
        CHECK(again[i].stats.stitchVertices ==
              frame.seams[i].stats.stitchVertices);
        CHECK(std::abs(again[i].stats.maxResidual -
                       frame.seams[i].stats.maxResidual) < 1.0e-7f);
    }
}

void testSelectionDeterminism() {
    terrain::Terrain terrain{terrain::TerrainParameters{}};
    terrain::NodeStateCache cacheA(2048);
    terrain::NodeStateCache cacheB(2048);
    terrain::LodSelector selector(terrain, 16.0f);
    const terrain::Camera camera = testCamera();

    const terrain::SelectionResult a =
        selector.select(camera, 960, 540, 0, cacheA);
    const terrain::SelectionResult b =
        selector.select(camera, 960, 540, 0, cacheB);
    CHECK(a.leaves.size() == b.leaves.size());
    CHECK(a.triangles == b.triangles);
    CHECK(a.drawCalls == b.drawCalls);
    CHECK(a.leavesByLod == b.leavesByLod);
    for (std::size_t i = 0; i < a.leaves.size(); ++i) {
        CHECK(a.leaves[i].key.encode() == b.leaves[i].key.encode());
        CHECK(a.leaves[i].vertexY == b.leaves[i].vertexY);
    }
}

void testDistanceDependentLod() {
    terrain::Terrain terrain{terrain::TerrainParameters{}};
    terrain::LodSelector selector(terrain, 16.0f);
    terrain::NodeStateCache nearCache(2048);
    terrain::NodeStateCache farCache(2048);

    terrain::Camera nearCamera = testCamera();
    terrain::Camera farCamera;
    farCamera.position = {256.0f, 900.0f, -1300.0f};
    farCamera.yawDegrees = -5.0f;
    farCamera.pitchDegrees = -32.0f;

    const terrain::SelectionResult nearResult =
        selector.select(nearCamera, 960, 540, 0, nearCache);
    const terrain::SelectionResult farResult =
        selector.select(farCamera, 960, 540, 0, farCache);
    CHECK(!nearResult.leaves.empty());
    CHECK(!farResult.leaves.empty());
    CHECK(nearResult.triangles > farResult.triangles);
}

void testCacheReuseAndInvalidation() {
    terrain::NodeStateCache cache(4);
    terrain::CacheStats stats;
    cache.beginFrame();
    terrain::NodeCacheEntry& first = cache.acquire(100, 1, 0, stats);
    first.selectedAsLeaf = true;
    first.selectedDepth = 2;
    cache.storeStitchOffsets(100, {{3, 1.25f}});
    cache.endFrame(0, stats);
    CHECK(stats.misses == 1);

    cache.beginFrame();
    terrain::NodeCacheEntry& reentered = cache.acquire(100, 1, 1, stats);
    CHECK(stats.hits == 1);
    CHECK(reentered.stitchOffsets.size() == 1);
    cache.endFrame(1, stats);

    cache.beginFrame();
    terrain::NodeCacheEntry& invalidated = cache.acquire(100, 2, 2, stats);
    CHECK(stats.versionInvalidations == 1);
    CHECK(invalidated.stitchOffsets.empty());
    cache.endFrame(2, stats);

    cache.beginFrame();
    cache.acquire(101, 2, 3, stats);
    cache.acquire(102, 2, 3, stats);
    cache.acquire(103, 2, 3, stats);
    cache.acquire(104, 2, 3, stats);
    cache.acquire(105, 2, 3, stats);
    cache.endFrame(3, stats);
    cache.beginFrame();
    cache.acquire(105, 2, 4, stats);
    cache.endFrame(4, stats);
    CHECK(stats.evictions > 0);
    CHECK(cache.size() <= cache.capacity());
}

void testCameraPathParser() {
    const std::filesystem::path base =
        std::filesystem::current_path() / "terrain_lod_path_test.txt";
    {
        std::ofstream output(base);
        output << "# time px py pz yaw pitch\n"
               << "0 256 200 500 -10 -20\n"
               << "1 200 120 400 30 -15\n";
    }
    terrain::PathLoadResult result = terrain::loadCameraPath(base.string());
    CHECK(result.ok);
    CHECK(result.path.keyframes.size() == 2);
    const terrain::Camera camera = result.path.sample(0.5f, 1.5f);
    CHECK(std::abs(camera.position.x - 228.0f) < 1.0e-4f);

    {
        std::ofstream output(base);
        output << "0 256 200\n";
    }
    result = terrain::loadCameraPath(base.string());
    CHECK(!result.ok);

    {
        std::ofstream output(base);
        output << "0 256 200 500 -10x -20\n";
    }
    result = terrain::loadCameraPath(base.string());
    CHECK(!result.ok);

    result = terrain::loadCameraPath((base.string() + ".missing"));
    CHECK(!result.ok);
    std::filesystem::remove(base);
}

void testPpmAndSoftwareRender() {
    terrain::Terrain terrain{terrain::TerrainParameters{}};
    terrain::LodSelector selector(terrain, 16.0f);
    terrain::SeamSolver seams(terrain);
    terrain::NodeStateCache cache(2048);
    const terrain::SceneFrame frame = terrain::buildSceneFrame(
        terrain,
        selector,
        seams,
        cache,
        testCamera(),
        320,
        180,
        terrain::DebugView::Seam,
        0,
        0.0f);
    terrain::SoftwareRenderer renderer(terrain, 320, 180);
    const terrain::ImageRgb image =
        renderer.render(frame, terrain::DebugView::Seam);
    CHECK(image.pixels.size() == 320u * 180u * 3u);
    bool hasNonSky = false;
    for (std::size_t i = 0; i < image.pixels.size(); i += 3) {
        if (image.pixels[i + 1] > image.pixels[i + 2] + 12) {
            hasNonSky = true;
            break;
        }
    }
    CHECK(hasNonSky);

    const std::filesystem::path filename =
        std::filesystem::current_path() / "terrain_lod_test.ppm";
    std::string error;
    CHECK(terrain::writePpm(filename.string(), image, error));
    std::ifstream input(filename, std::ios::binary);
    std::string magic;
    input >> magic;
    CHECK(magic == "P6");
    std::filesystem::remove(filename);
}

}  // namespace

int main() {
    testTerrainDeterminism();
    testCameraFrustum();
    testSelectionAndStitching();
    testSelectionDeterminism();
    testDistanceDependentLod();
    testCacheReuseAndInvalidation();
    testCameraPathParser();
    testPpmAndSoftwareRender();

    if (failures == 0) {
        std::cout << "all terrain tests passed\n";
        return 0;
    }
    std::cerr << failures << " test failure(s)\n";
    return 1;
}

