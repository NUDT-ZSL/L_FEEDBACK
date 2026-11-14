#include "error.h"
#include "image_io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

std::uint64_t fnv1aBytes(const std::vector<std::uint8_t>& data) {
    std::uint64_t hash = 1469598103934665603ull;
    for (std::uint8_t value : data) {
        hash ^= value;
        hash *= 1099511628211ull;
    }
    return hash;
}

std::uint64_t fnv1aFloats(const std::vector<float>& data) {
    std::uint64_t hash = 1469598103934665603ull;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(data.data());
    for (std::size_t i = 0; i < data.size() * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

float percentile(std::vector<float> values, float fraction) {
    if (values.empty()) return 0.0f;
    std::sort(values.begin(), values.end());
    const std::size_t index = std::min(
        values.size() - 1,
        static_cast<std::size_t>(static_cast<double>(values.size()) * fraction));
    return values[index];
}

struct Images {
    df::RgbImage color;
    int depthWidth = 0;
    int depthHeight = 0;
    std::vector<float> depth;
    Json metadata;
    Json transparent;
};

Images loadArtifacts(const std::string& directory, bool& ok) {
    Images images;
    images.color = df::readPngRgb(directory + "/color.png");
    images.depth = df::readPfm(directory + "/depth.pfm", images.depthWidth, images.depthHeight);
    std::ifstream metadataInput(directory + "/frame_metadata.json");
    std::ifstream transparentInput(directory + "/transparent_objects.json");
    if (!metadataInput || !transparentInput) {
        std::cerr << "missing JSON artifacts in " << directory << '\n';
        ok = false;
        return images;
    }
    metadataInput >> images.metadata;
    transparentInput >> images.transparent;
    return images;
}

bool validateSingle(const std::string& directory, Json& result) {
    bool ok = true;
    Images images;
    try {
        images = loadArtifacts(directory, ok);
    } catch (const std::exception& error) {
        std::cerr << "artifact load failed: " << error.what() << '\n';
        return false;
    }
    if (!ok) return false;

    const int width = images.metadata.at("resolution").at("width").get<int>();
    const int height = images.metadata.at("resolution").at("height").get<int>();
    const float farPlane = images.metadata.at("camera").at("farPlane").get<float>();
    const int fogSteps = images.metadata.at("volumetric").at("steps").get<int>();
    const int transparentDepthWrites =
        images.metadata.at("transparent").at("transparentDepthWrites").get<int>();
    const std::uint64_t colorHash =
        images.metadata.at("hashes").at("colorFnv1a64").get<std::uint64_t>();
    const std::uint64_t depthHash =
        images.metadata.at("hashes").at("depthFnv1a64").get<std::uint64_t>();

    auto condition = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAIL: " << message << '\n';
            ok = false;
        }
    };

    condition(images.color.width == width && images.color.height == height,
              "PNG resolution matches metadata");
    condition(images.depthWidth == width && images.depthHeight == height,
              "PFM resolution matches metadata");
    condition(images.color.pixels.size() ==
                  static_cast<std::size_t>(width) * height * 3u,
              "PNG RGB payload size");
    condition(images.depth.size() == static_cast<std::size_t>(width) * height,
              "PFM payload size");
    condition(fogSteps == 32, "offline volumetric integration uses exactly 32 steps");
    condition(transparentDepthWrites == 0,
              "transparent geometry does not write fog depth");
    condition(images.metadata.at("gbuffer").at("minComponentBits").get<int>() >= 16,
              "G-Buffer component precision >= 16 bits");
    condition(fnv1aBytes(images.color.pixels) == colorHash,
              "color PNG FNV-1a hash matches metadata");
    condition(fnv1aFloats(images.depth) == depthHash,
              "depth PFM FNV-1a hash matches metadata");

    float minimumDepth = std::numeric_limits<float>::max();
    float maximumDepth = -std::numeric_limits<float>::max();
    bool finite = true;
    for (float depth : images.depth) {
        finite = finite && std::isfinite(depth);
        minimumDepth = std::min(minimumDepth, depth);
        maximumDepth = std::max(maximumDepth, depth);
    }
    condition(finite, "all depth values are finite");
    condition(minimumDepth >= 0.0f && maximumDepth <= farPlane + 1.0e-3f,
              "depth values stay within camera range");

    const Json& transparent = images.transparent;
    condition(transparent.at("schema") == "deferred-fog-transparent-list/v1",
              "transparent list schema");
    condition(transparent.at("premultipliedAlpha").get<bool>(),
              "transparent list declares premultiplied alpha");
    condition(!transparent.at("doesWriteFogDepth").get<bool>(),
              "transparent list declares no fog-depth writes");
    const auto& objects = transparent.at("objects");
    bool ordered = true;
    for (std::size_t i = 0; i < objects.size(); ++i) {
        if (objects[i].at("rank").get<int>() != static_cast<int>(i)) ordered = false;
        if (i > 0) {
            const float previous = objects[i - 1].at("viewDepth").get<float>();
            const float current = objects[i].at("viewDepth").get<float>();
            if (current > previous + 1.0e-4f) ordered = false;
        }
    }
    condition(ordered, "transparent objects are ranked far-to-near");
    condition(objects.size() == images.metadata.at("transparent").at("count").get<std::size_t>(),
              "transparent list count matches metadata");

    result = {
        {"directory", directory},
        {"backend", images.metadata.at("backend")},
        {"resolution", {{"width", width}, {"height", height}}},
        {"colorHash", colorHash},
        {"depthHash", depthHash},
        {"depthRange", {{"min", minimumDepth}, {"max", maximumDepth}}},
        {"transparentCount", objects.size()},
        {"sortMode", images.metadata.at("transparent").at("sortMode")},
        {"lightSnapshotVersion", images.metadata.at("lightSnapshotVersion")},
        {"lightUploads", images.metadata.at("lightUploads")},
        {"fogSteps", fogSteps},
        {"checksPassed", ok},
    };
    return ok;
}

bool compare(const Images& a, const Images& b, Json& report, bool requireExact) {
    std::vector<float> colorDifferences;
    for (std::size_t i = 0; i < a.color.pixels.size(); ++i) {
        colorDifferences.push_back(
            static_cast<float>(std::abs(static_cast<int>(a.color.pixels[i]) -
                                        static_cast<int>(b.color.pixels[i]))));
    }
    std::vector<float> depthDifferences;
    for (std::size_t i = 0; i < a.depth.size(); ++i) {
        depthDifferences.push_back(std::abs(a.depth[i] - b.depth[i]));
    }
    double colorSum = 0.0;
    double depthSum = 0.0;
    for (float value : colorDifferences) colorSum += value;
    for (float value : depthDifferences) depthSum += value;
    report = {
        {"colorMeanAbsDiff",
         colorDifferences.empty() ? 0.0 : colorSum / colorDifferences.size()},
        {"colorP99AbsDiff", percentile(colorDifferences, 0.99f)},
        {"depthMeanAbsDiff",
         depthDifferences.empty() ? 0.0 : depthSum / depthDifferences.size()},
        {"depthP99AbsDiff", percentile(depthDifferences, 0.99f)},
        {"exactMatch", colorSum == 0.0 && depthSum == 0.0},
    };
    const bool exact = colorSum == 0.0 && depthSum == 0.0;
    if (requireExact) {
        std::cerr << (exact ? "PASS: " : "FAIL: ") << "artifacts require exact match\n";
    }
    return !requireExact || exact;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: deferred_verify <artifact-dir> [compare-dir] [--exact]\n";
        return 2;
    }
    try {
        const std::string directory = argv[1];
        bool ok = true;
        Json primary;
        ok = validateSingle(directory, primary) && ok;

        Json comparison = nullptr;
        if (argc >= 3 && std::string(argv[2]) != "--exact") {
            const std::string compareDirectory = argv[2];
            bool compareOk = true;
            const Images a = loadArtifacts(directory, compareOk);
            const Images b = loadArtifacts(compareDirectory, compareOk);
            if (!compareOk) return 1;
            if (a.color.width != b.color.width || a.color.height != b.color.height) {
                std::cerr << "FAIL: compared images have different dimensions\n";
                ok = false;
            } else {
                ok = compare(a, b, comparison,
                             argc >= 4 && std::string(argv[3]) == "--exact") && ok;
            }
        }
        Json result = {{"schema", "deferred-fog-verification/v1"},
                       {"primary", primary},
                       {"comparison", comparison},
                       {"passed", ok}};
        std::cout << result.dump(2) << '\n';
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "E_RENDER_FAIL: verification failed: " << error.what() << '\n';
        return 8;
    }
}
