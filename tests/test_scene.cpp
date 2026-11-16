#include "renderer/error.hpp"
#include "renderer/scene.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string read_binary(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::in | std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot read " + path.string());
    }
    std::ostringstream content;
    content << stream.rdbuf();
    return content.str();
}

void expect_invalid_scene(const std::string& path) {
    try {
        (void)renderer::load_scene(path);
    } catch (const renderer::AppError& error) {
        check(error.code() == "E_INVALID_SCENE",
              "expected E_INVALID_SCENE, got " + error.code());
        return;
    }
    throw std::runtime_error("invalid scene was accepted: " + path);
}

}  // namespace

int main() {
    using namespace renderer;
    try {
        const std::filesystem::path base =
            std::filesystem::temp_directory_path() / "offline_hdr_renderer_scene_tests";
        std::filesystem::remove_all(base);
        std::filesystem::create_directories(base);

        generate_samples((base / "first").string());
        generate_samples((base / "second").string());

        for (const char* sampleName :
             {"scene.json", "invalid_missing_camera.json",
              "invalid_atlas_capacity.json"}) {
            const std::string first =
                read_binary(base / "first" / sampleName);
            const std::string second =
                read_binary(base / "second" / sampleName);
            check(first == second,
                  std::string("sample generation is not byte-deterministic: ")
                      + sampleName);
        }

        const std::string firstSample =
            read_binary(base / "first" / "scene.json");
        Scene scene = load_scene((base / "first" / "scene.json").string());
        check(scene.materials.size() == 4, "unexpected material count");
        check(scene.meshes.size() == 4, "unexpected mesh count");
        check(scene.lights.size() == 3, "unexpected light count");
        check(scene.lights[0].type == LightType::Directional,
              "first light should be directional");
        check(scene.lights[1].type == LightType::Point,
              "second light should be point");
        check(scene.lights[2].type == LightType::Spot,
              "third light should be spot");
        check(shadow_tile_demand(scene) == 8, "point light must reserve six cube faces");
        check(shadow_tile_capacity(scene) == 16, "atlas capacity is incorrect");
        check(effective_tile_resolution(scene, "fast") == 128,
              "fast mode atlas scale was not applied");
        check(effective_tile_resolution(scene, "high") == 256,
              "high mode atlas scale was not applied");
        check(scene.settings.fast.shadowSamples == 1,
              "fast shadow sample count is incorrect");
        check(scene.settings.high.shadowSamples == 25,
              "high shadow sample count is incorrect");
        check(scene.settings.fast.shadowSamples
                  < scene.settings.high.shadowSamples,
              "high mode must increase shadow samples");
        check(scene.settings.fast.atlasScale < scene.settings.high.atlasScale,
              "high mode must increase atlas resolution");
        check(scene.bounds.valid(), "scene bounds were not computed");

        expect_invalid_scene((base / "first" / "invalid_missing_camera.json").string());
        expect_invalid_scene((base / "first" / "invalid_atlas_capacity.json").string());

        nlohmann::json oversized = nlohmann::json::parse(firstSample);
        while (oversized["lights"].size() < 33) {
            oversized["lights"].push_back(oversized["lights"][0]);
        }
        {
            std::ofstream stream(base / "invalid_light_count.json",
                                 std::ios::out | std::ios::binary);
            stream << oversized.dump(2) << '\n';
        }
        expect_invalid_scene((base / "invalid_light_count.json").string());

        {
            std::ofstream stream(base / "invalid_json.json",
                                 std::ios::out | std::ios::binary);
            stream << "{\"version\": 1, \"camera\":";
        }
        expect_invalid_scene((base / "invalid_json.json").string());

        nlohmann::json maximum = nlohmann::json::parse(firstSample);
        nlohmann::json maximumLights = nlohmann::json::array();
        nlohmann::json pointLight = maximum["lights"][1];
        pointLight["castShadow"] = false;
        for (int index = 0; index < 32; ++index) {
            maximumLights.push_back(pointLight);
        }
        maximum["lights"] = maximumLights;
        {
            std::ofstream stream(base / "maximum_32_lights.json",
                                 std::ios::out | std::ios::binary);
            stream << maximum.dump(2) << '\n';
        }
        const Scene maximumScene =
            load_scene((base / "maximum_32_lights.json").string());
        check(maximumScene.lights.size() == 32,
              "32 lights should be accepted");
        check(shadow_tile_demand(maximumScene) == 0,
              "non-shadow lights must not reserve atlas tiles");

        std::cout << "scene_tests=passed samples=deterministic"
                  << " lights=3 shadow_tiles=8/16"
                  << " fast_tile=128 high_tile=256"
                  << " invalid_cases=4 max_lights=32\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "scene_tests=failed error=" << error.what() << '\n';
        return 1;
    }
}
