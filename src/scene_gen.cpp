#include "errors.hpp"
#include "png.hpp"
#include "scene.hpp"

#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

std::string makeSceneJson(std::uint32_t seed) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schemaVersion\": 1,\n"
        << "  \"name\": \"deterministic-volume-shadow-sample\",\n"
        << "  \"camera\": {\n"
        << "    \"position\": [4.2, 2.8, 6.5],\n"
        << "    \"target\": [0.0, 1.2, 0.0],\n"
        << "    \"up\": [0.0, 1.0, 0.0],\n"
        << "    \"fovYDegrees\": 46.0,\n"
        << "    \"nearMeters\": 0.1,\n"
        << "    \"farMeters\": 40.0\n"
        << "  },\n"
        << "  \"output\": {\n"
        << "    \"width\": 384,\n"
        << "    \"height\": 216,\n"
        << "    \"exposure\": 1.0,\n"
        << "    \"gain\": 1.0,\n"
        << "    \"pngPath\": \"render.png\",\n"
        << "    \"statsPath\": \"stats.json\",\n"
        << "    \"referencePngPath\": \"render.reference.png\",\n"
        << "    \"background\": [0.006, 0.012, 0.022]\n"
        << "  },\n"
        << "  \"noise\": {\n"
        << "    \"path\": \"noise.raw\",\n"
        << "    \"dimension\": 32,\n"
        << "    \"seed\": " << seed << "\n"
        << "  },\n"
        << "  \"volume\": {\n"
        << "    \"boundsMin\": [-6.0, 0.0, -6.0],\n"
        << "    \"boundsMax\": [6.0, 5.0, 6.0],\n"
        << "    \"density\": 0.105,\n"
        << "    \"absorption\": 0.025,\n"
        << "    \"scattering\": 0.58,\n"
        << "    \"anisotropy\": 0.28,\n"
        << "    \"stepMeters\": 0.12,\n"
        << "    \"maxSteps\": 192,\n"
        << "    \"noiseFrequencyPerMeter\": 0.16\n"
        << "  },\n"
        << "  \"directionalLight\": {\n"
        << "    \"direction\": [0.82, -0.28, -0.35],\n"
        << "    \"color\": [1.0, 0.92, 0.78],\n"
        << "    \"intensity\": 4.5\n"
        << "  },\n"
        << "  \"pointLight\": {\n"
        << "    \"position\": [2.4, 2.35, 1.7],\n"
        << "    \"color\": [1.0, 0.52, 0.22],\n"
        << "    \"intensity\": 10.0,\n"
        << "    \"rangeMeters\": 12.0\n"
        << "  },\n"
        << "  \"shadows\": {\n"
        << "    \"resolution\": 1024,\n"
        << "    \"pcfKernel\": 5,\n"
        << "    \"directionalBias\": 0.008,\n"
        << "    \"pointBias\": 0.003\n"
        << "  },\n"
        << "  \"geometry\": {\n"
        << "    \"ground\": {\n"
        << "      \"height\": 0.0,\n"
        << "      \"sizeMeters\": 10.0,\n"
        << "      \"albedo\": [0.24, 0.22, 0.19]\n"
        << "    },\n"
        << "    \"spheres\": [\n"
        << "      {\"center\": [-1.35, 0.85, -0.55], \"radius\": 0.85, "
           "\"albedo\": [0.78, 0.28, 0.16]},\n"
        << "      {\"center\": [1.05, 0.62, 0.15], \"radius\": 0.62, "
           "\"albedo\": [0.18, 0.62, 0.42]},\n"
        << "      {\"center\": [0.20, 1.75, -1.55], \"radius\": 0.55, "
           "\"albedo\": [0.24, 0.42, 0.82]},\n"
        << "      {\"center\": [-2.15, 2.15, -1.25], \"radius\": 0.90, "
           "\"albedo\": [0.52, 0.32, 0.62]}\n"
        << "    ]\n"
        << "  }\n"
        << "}\n";
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            std::cerr << "usage: scene_gen <output-directory> [seed]\n";
            return 2;
        }
        const std::filesystem::path directory = std::filesystem::absolute(argv[1]);
        std::uint32_t seed = 1337;
        if (argc == 3) {
            try {
                const unsigned long parsed = std::stoul(argv[2]);
                if (parsed > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::out_of_range("seed");
                }
                seed = static_cast<std::uint32_t>(parsed);
            } catch (const std::exception&) {
                std::cerr << "E_USAGE: seed must be an unsigned integer\n";
                return 13;
            }
        }
        std::error_code ec;
        std::filesystem::create_directories(directory, ec);
        if (ec) {
            vlr::fail(vlr::ErrorCode::Io,
                      "cannot create sample directory: " + directory.string());
        }
        const std::string sceneText = makeSceneJson(seed);
        vlr::writeTextAtomic(directory / "scene.json", sceneText);
        vlr::writeBytesAtomic(directory / "noise.raw",
                              vlr::generateNoiseRaw(32, seed));
        std::cout << "generated " << (directory / "scene.json").string() << "\n";
        std::cout << "generated " << (directory / "noise.raw").string() << "\n";
        std::cout << "noise_seed=" << seed << " dimension=32 format=float32_le\n";
        return 0;
    } catch (const vlr::AppError& error) {
        std::cerr << vlr::errorCodeName(error.code()) << ": " << error.what() << "\n";
        return error.exitCode();
    } catch (const std::exception& error) {
        std::cerr << "E_INTERNAL: " << error.what() << "\n";
        return 1;
    }
}
