#include "config.h"
#include "error.h"
#include "image_io.h"
#include "lights.h"
#include "noise.h"
#include "offline.h"
#include "renderer_cpu.h"
#include "scene.h"
#include "sorting.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#ifndef DF_TEST_CONFIG_DIR
#define DF_TEST_CONFIG_DIR "config"
#endif

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Fn>
void expectError(df::ErrorCode expected, Fn&& fn, const std::string& message) {
    try {
        fn();
        check(false, message + " (no exception)");
    } catch (const df::AppError& error) {
        check(error.code() == expected,
              message + " (wrong error code " + df::errorCodeName(error.code()) + ")");
    } catch (...) {
        check(false, message + " (wrong exception type)");
    }
}

void testConfig() {
    const std::string dir = std::string(DF_TEST_CONFIG_DIR);
    const df::RenderConfig config = df::loadConfig(dir + "/render.json");
    check(config.width == 640 && config.height == 360, "config dimensions");
    check(config.volumetric.steps == 28, "config fog steps");
    check(df::effectiveFogSteps(config, true) == 32, "offline forces 32 fog steps");
    check(df::effectiveFogSteps(config, false) == 28, "interactive uses configured steps");

    expectError(df::ErrorCode::E_CONFIG_PARSE,
                [&] { (void)df::loadConfig(dir + "/invalid_steps.json"); },
                "out-of-range fog steps rejected");
    expectError(df::ErrorCode::E_CONFIG_PARSE,
                [&] { (void)df::loadConfig(dir + "/invalid_type.json"); },
                "wrong JSON type rejected");
    expectError(df::ErrorCode::E_CONFIG_PARSE,
                [&] { (void)df::loadConfig(dir + "/invalid_syntax.json"); },
                "malformed JSON rejected");
    expectError(df::ErrorCode::E_CONFIG_PARSE,
                [&] { (void)df::loadConfig(dir + "/does_not_exist.json"); },
                "missing config rejected");
}

void testNoise() {
    const df::Vec3 p{1.25f, 2.5f, -3.75f};
    const float a = df::fbm3D(p, 4, 1234u);
    const float b = df::fbm3D(p, 4, 1234u);
    const float c = df::fbm3D(p + df::Vec3{0.1f, 0.0f, 0.0f}, 4, 1234u);
    check(a == b, "3D noise is deterministic");
    check(a >= 0.0f && a <= 1.0f, "3D noise is normalized");
    check(std::fabs(a - c) > 1.0e-7f, "3D noise varies spatially");
    const float d0 = df::densityAt(p, 0.11f, 0.19f);
    const float d1 = df::densityAt(p, 0.22f, 0.19f);
    check(d1 >= d0, "density scales monotonically");
}

void testSorting() {
    df::Scene scene = df::makeDeterministicScene(128, 72);
    scene.transparent.clear();
    df::Primitive a;
    a.id = 3;
    a.center = {0.0f, 0.0f, -2.0f};
    a.opacity = 0.2f;
    df::Primitive b = a;
    b.id = 4;
    b.center = {0.0f, 0.0f, -5.0f};
    df::Primitive c = a;
    c.id = 2;
    c.center = {0.0f, 0.0f, -3.5f};
    scene.transparent = {a, b, c};

    df::TransparentConfig config;
    df::SortResult sorted = df::sortTransparent(scene, config, scene.camera);
    check(sorted.mode == df::SortMode::Exact, "small list uses exact sorting");
    check(sorted.objects.size() == 3, "all visible transparent objects retained");
    check(sorted.objects[0].id == 4 && sorted.objects[1].id == 2 &&
              sorted.objects[2].id == 3,
          "transparent objects sorted far-to-near");

    config.sortingEnabled = false;
    sorted = df::sortTransparent(scene, config, scene.camera);
    check(sorted.mode == df::SortMode::Disabled, "sorting switch disables sorting");
    check(sorted.objects[0].id == 3 && sorted.objects[1].id == 4 &&
              sorted.objects[2].id == 2,
          "disabled sorting preserves scene order");

    df::Scene stress = df::makeSortingStressScene(1025);
    df::TransparentConfig stressConfig;
    sorted = df::sortTransparent(stress, stressConfig, stress.camera);
    check(sorted.mode == df::SortMode::BucketApprox,
          "more than 1024 transparent objects switches to bucket sorting");
    check(sorted.objects.size() == 1025, "stress transparent list complete");
    for (std::size_t i = 1; i < sorted.objects.size(); ++i) {
        check(sorted.objects[i - 1].depth + 0.01f >= sorted.objects[i].depth,
              "bucket sorting remains far-to-near within bucket tolerance: " +
                  std::to_string(sorted.objects[i - 1].depth) + " < " +
                  std::to_string(sorted.objects[i].depth));
        if (failures > 10) break;
    }
}

void testLightSnapshots() {
    df::Scene scene = df::makeDeterministicScene(64, 64);
    df::RenderConfig config;
    config.width = 64;
    config.height = 64;
    config.volumetric.enabled = false;
    df::CpuRenderer renderer;
    df::FrameOutput first = renderer.render(scene, config, false, false);
    check(first.stats.lightVersion == 1, "initial light snapshot has version 1");
    check(first.stats.lightingUploads == 1 && first.stats.fogUploads == 1,
          "both stages upload initial snapshot");
    df::FrameOutput second = renderer.render(scene, config, false, false);
    check(second.stats.lightVersion == 1, "unchanged lights keep snapshot version");
    check(second.stats.lightingUploads == 1 && second.stats.fogUploads == 1,
          "unchanged lights do not re-upload");

    scene.lights[0].position.x += 0.25f;
    df::FrameOutput moved = renderer.render(scene, config, false, false);
    check(moved.stats.lightVersion == 2, "position change increments snapshot version");
    check(moved.stats.lightingUploads == 2 && moved.stats.fogUploads == 2,
          "position change re-uploads to both stages");

    scene.lights[0].color.y += 0.1f;
    df::FrameOutput recolored = renderer.render(scene, config, false, false);
    check(recolored.stats.lightVersion == 3, "color change increments snapshot version");

    scene.lights[0].intensity += 0.5f;
    df::FrameOutput brighter = renderer.render(scene, config, false, false);
    check(brighter.stats.lightVersion == 4, "intensity change increments snapshot version");
    check(brighter.stats.lightingUploads == 4 && brighter.stats.fogUploads == 4,
          "each changed snapshot uploads once per stage");
}

void testGBufferAndDepth() {
    df::Scene scene = df::makeDeterministicScene(192, 108);
    df::RenderConfig config;
    config.width = 192;
    config.height = 108;
    config.volumetric.enabled = false;
    df::CpuRenderer renderer;
    df::FrameOutput frame = renderer.render(scene, config, false, true);
    const std::size_t pixels = static_cast<std::size_t>(frame.width) * frame.height;
    check(frame.gbuffer.gPosition.size() == pixels, "G-Buffer position size");
    check(frame.gbuffer.gNormal.size() == pixels, "G-Buffer normal size");
    check(frame.gbuffer.gAlbedo.size() == pixels, "G-Buffer albedo size");
    check(frame.gbuffer.gMaterial.size() == pixels, "G-Buffer material size");
    check(frame.linearDepth.size() == pixels, "depth buffer size");
    check(frame.stats.transparentDepthWrites == 0,
          "transparent objects do not write fog depth");

    bool found = false;
    for (std::size_t i = 0; i < pixels; ++i) {
        const df::Vec4& position = frame.gbuffer.gPosition[i];
        if (position.w > 0.0f) {
            found = true;
            check(frame.gbuffer.gAlbedo[i].w >= 0.0f && frame.gbuffer.gAlbedo[i].w <= 1.0f,
                  "roughness stored in G-Buffer");
            check(frame.gbuffer.gMaterial[i].x >= 0.0f &&
                      frame.gbuffer.gMaterial[i].x <= 1.0f,
                  "metallic stored in G-Buffer");
            check(std::fabs(frame.linearDepth[i] - position.w) < 1.0e-4f,
                  "depth texture agrees with G-Buffer view depth");
            break;
        }
    }
    check(found, "deterministic scene produces opaque G-Buffer hits");
}

double meanPixelDifference(const std::vector<std::uint8_t>& a,
                           const std::vector<std::uint8_t>& b, float& p99) {
    if (a.size() != b.size() || a.empty()) {
        p99 = 255.0f;
        return 255.0;
    }
    std::vector<int> differences;
    differences.reserve(a.size());
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const int diff = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        differences.push_back(diff);
        sum += static_cast<std::uint64_t>(diff);
    }
    std::sort(differences.begin(), differences.end());
    p99 = static_cast<float>(differences[std::min(differences.size() - 1,
                                                   static_cast<std::size_t>(differences.size() * 0.99))]);
    return static_cast<double>(sum) / static_cast<double>(a.size());
}

void testApproximateSortDifference() {
    df::Scene scene = df::makeSortingStressScene(1025);
    df::RenderConfig exactConfig;
    exactConfig.width = 192;
    exactConfig.height = 108;
    exactConfig.volumetric.enabled = false;
    exactConfig.transparent.exactSortLimit = 100000;
    df::RenderConfig bucketConfig = exactConfig;
    bucketConfig.transparent.exactSortLimit = 1024;
    bucketConfig.transparent.bucketCount = 64;

    df::CpuRenderer exactRenderer;
    df::CpuRenderer bucketRenderer;
    df::FrameOutput exact = exactRenderer.render(scene, exactConfig, false, false);
    df::FrameOutput bucket = bucketRenderer.render(scene, bucketConfig, false, false);
    check(exact.transparent.mode == df::SortMode::Exact, "comparison baseline is exact");
    check(bucket.transparent.mode == df::SortMode::BucketApprox,
          "comparison target uses bucket approximation");

    float p99 = 0.0f;
    const double mean = meanPixelDifference(exact.colorRgb, bucket.colorRgb, p99);
    std::cout << "sort-diff mean=" << mean << " p99=" << p99 << '\n';
    check(mean <= 1.0, "bucket sorting mean pixel difference <= 1/255");
    check(p99 <= 8.0f, "bucket sorting p99 pixel difference <= 8/255");
}

void testOfflineArtifacts() {
    const std::string output = "artifacts/test_offline";
    std::filesystem::remove_all(output);
    df::RenderConfig config;
    config.width = 160;
    config.height = 90;
    config.volumetric.steps = 20;
    const df::OfflineArtifacts artifacts = df::renderOffline(config, output);
    check(artifacts.frame.stats.fogSteps == 32, "offline artifact uses 32 fog steps");
    check(artifacts.frame.stats.transparentDepthWrites == 0,
          "offline transparent depth writes remain zero");
    check(std::filesystem::file_size(artifacts.colorPng) > 100, "color PNG written");
    check(std::filesystem::file_size(artifacts.depthPfm) > 24, "depth PFM written");
    check(std::filesystem::file_size(artifacts.transparentJson) > 100,
          "transparent JSON written");

    const df::RgbImage png = df::readPngRgb(artifacts.colorPng);
    check(png.width == 160 && png.height == 90, "PNG dimensions");
    check(png.pixels.size() == static_cast<std::size_t>(160 * 90 * 3), "PNG RGB payload");
    int depthWidth = 0;
    int depthHeight = 0;
    const std::vector<float> depth = df::readPfm(artifacts.depthPfm, depthWidth, depthHeight);
    check(depthWidth == 160 && depthHeight == 90, "PFM dimensions");
    check(depth.size() == static_cast<std::size_t>(160 * 90), "PFM payload");
    check(*std::max_element(depth.begin(), depth.end()) <= 80.0f,
          "PFM depths bounded");

    std::ifstream jsonInput(artifacts.transparentJson);
    nlohmann::json json;
    jsonInput >> json;
    check(json.at("schema") == "deferred-fog-transparent-list/v1", "transparent JSON schema");
    check(json.at("doesWriteFogDepth") == false, "transparent JSON depth contract");
    check(json.at("objects").size() == artifacts.frame.transparent.objects.size(),
          "transparent JSON list size");
    check(json.at("objects").at(0).at("rank") == 0, "transparent JSON rank starts at zero");
}

} // namespace

int main() {
    testConfig();
    testNoise();
    testSorting();
    testLightSnapshots();
    testGBufferAndDepth();
    testApproximateSortDifference();
    testOfflineArtifacts();
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "all core tests passed\n";
    return 0;
}
