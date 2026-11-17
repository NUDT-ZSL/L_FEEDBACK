#include "json.h"
#include "renderer.h"
#include "sample_scene.h"
#include "scene.h"
#include "sha256.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void testSha256() {
    check(softraster::sha256Hex("") ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "SHA-256 empty vector");
    check(softraster::sha256Hex("abc") ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256 abc vector");
}

void testJsonParser() {
    const auto parsed = softraster::parseJson(R"({"a":[1,2,{"b":"x"}],"n":1e999})");
    check(parsed.value.has_value(), "JSON parser accepts valid structure");
    if (!parsed.value.has_value()) {
        return;
    }
    check(parsed.value->isObject(), "JSON root is an object");
    const softraster::JsonValue* number = parsed.value->find("n");
    check(number != nullptr && number->isNumber(), "JSON large number parsed as number");
    if (number != nullptr && number->isNumber()) {
        check(!std::isfinite(number->asNumber()), "JSON overflow number remains non-finite");
    }
    const auto duplicate = softraster::parseJson(R"({"a":1,"a":2})");
    check(!duplicate.value.has_value(), "duplicate JSON keys are rejected");
    const auto malformed = softraster::parseJson(R"({"a": [1,]})");
    check(!malformed.value.has_value(), "malformed JSON is rejected");
}

std::filesystem::path writeTemporaryScene() {
    const auto path =
        std::filesystem::temp_directory_path() / "softraster_unit_sample_scene.json";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    const std::string json = softraster::buildSampleSceneJson();
    output.write(json.data(), static_cast<std::streamsize>(json.size()));
    output.close();
    return path;
}

void testSampleRender(const std::filesystem::path& path) {
    const auto loaded = softraster::loadScene(path.string());
    check(loaded.scene.has_value(), "sample scene loads");
    if (!loaded.scene.has_value()) {
        if (loaded.error.has_value()) {
            std::cerr << softraster::formatErrorJson(*loaded.error);
        }
        return;
    }
    check(loaded.scene->meshes.size() == 4, "sample scene has four meshes");
    check(loaded.scene->pointLights.size() == 3, "sample scene has three point lights");
    check(loaded.scene->directionalLight.shadowMapSize == 512,
          "sample shadow map resolution is 512");

    const auto first = softraster::renderScene(*loaded.scene);
    check(first.frame.has_value(), "sample scene renders");
    if (!first.frame.has_value()) {
        if (first.error.has_value()) {
            std::cerr << softraster::formatErrorJson(*first.error);
        }
        return;
    }
    const auto second = softraster::renderScene(*loaded.scene);
    check(second.frame.has_value(), "sample scene renders a second time");
    if (!second.frame.has_value()) {
        return;
    }

    check(first.frame->ppm == second.frame->ppm, "repeated rendering is byte-identical");
    check(softraster::sha256Hex(first.frame->ppm) ==
              softraster::sha256Hex(second.frame->ppm),
          "repeated rendering has the same SHA-256");
    check(first.frame->stats.rasterizedTriangles > 0, "rasterized triangle count is positive");
    check(first.frame->stats.shadowSampleCount > 0, "shadow sample count is positive");
    check(first.frame->stats.transparentTriangles > 0,
          "transparent triangle count is positive");
    check(first.frame->stats.transparentSortSwaps > 0,
          "transparent sorting records swaps");
    check(first.frame->stats.transparentFragments > 0,
          "transparent fragments were blended");
    check(first.frame->stats.shadowMapSize == 512,
          "statistics shadow map size matches the scene");
    check(first.frame->stats.ppmBytes == first.frame->ppm.size(),
          "statistics PPM byte count matches the output");
    check(first.frame->stats.peakMemoryEstimatedBytes > 0,
          "peak memory estimate is positive");
    check(first.frame->ppm.size() > 3ULL * 320ULL * 320ULL, "PPM has a complete payload");
    check(first.frame->ppm[0] == 'P' && first.frame->ppm[1] == '6',
          "output starts with the P6 PPM signature");
    std::cout << "sample sha256=" << softraster::sha256Hex(first.frame->ppm)
              << " rasterized=" << first.frame->stats.rasterizedTriangles
              << " clipped=" << first.frame->stats.clippedTriangles
              << " shadowSamples=" << first.frame->stats.shadowSampleCount
              << " sortSwaps=" << first.frame->stats.transparentSortSwaps
              << " peakMemory=" << first.frame->stats.peakMemoryEstimatedBytes << '\n';
}

}  // namespace

int main() {
    testSha256();
    testJsonParser();
    const auto path = writeTemporaryScene();
    testSampleRender(path);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "all unit tests passed\n";
    return 0;
}
