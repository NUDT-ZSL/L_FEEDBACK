#include "config.h"
#include "crc32.h"
#include "density.h"
#include "errors.h"
#include "json.h"
#include "renderer.h"
#include "snapshot.h"
#include "thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

int gFailures = 0;

void require(bool condition, const std::string& message) {
    if (!condition) {
        ++gFailures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void runTest(const std::string& name, const std::function<void()>& test) {
    const int before = gFailures;
    try {
        test();
    } catch (const std::exception& error) {
        ++gFailures;
        std::cerr << "FAIL: " << name << " threw: " << error.what() << '\n';
    }
    std::cout << (gFailures == before ? "PASS: " : "FAIL: ") << name << '\n';
}

std::uint64_t fnv1a(const std::vector<std::uint8_t>& bytes) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const std::uint8_t byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

bool relativeEqual(const std::vector<cloud::Vec3>& a, const std::vector<cloud::Vec3>& b,
                   double tolerance) {
    if (a.size() != b.size()) {
        return false;
    }
    double maximumMagnitude = 1.0e-6;
    double maximumDifference = 0.0;
    for (std::size_t index = 0; index < a.size(); ++index) {
        maximumMagnitude = std::max({maximumMagnitude, static_cast<double>(std::abs(a[index].x)),
                                     static_cast<double>(std::abs(a[index].y)),
                                     static_cast<double>(std::abs(a[index].z))});
        maximumDifference =
            std::max({maximumDifference, static_cast<double>(std::abs(a[index].x - b[index].x)),
                      static_cast<double>(std::abs(a[index].y - b[index].y)),
                      static_cast<double>(std::abs(a[index].z - b[index].z))});
    }
    return maximumDifference / maximumMagnitude < tolerance;
}

cloud::SnapshotData snapshotFromState(const cloud::RenderState& state, const cloud::AppConfig& config,
                                      const cloud::DensityField& density) {
    cloud::SnapshotData data;
    data.seed = density.seed();
    data.cloudType = static_cast<std::uint32_t>(density.type());
    data.densityResolution = static_cast<std::uint32_t>(density.resolution());
    data.frameIndex = state.frameIndex;
    data.width = static_cast<std::uint32_t>(config.width);
    data.height = static_cast<std::uint32_t>(config.height);
    data.historyLength = static_cast<std::uint32_t>(config.maxHistoryLength);
    data.nextLight = cloud::defaultLight();
    data.view = state.previousView;
    data.projection = state.previousProjection;
    data.accumulatedColor = state.accumulatedColor;
    data.depth = state.depth;
    data.counts = state.counts;
    return data;
}

void restoreState(cloud::RenderState& state, const cloud::SnapshotData& snapshot) {
    state.configure(static_cast<int>(snapshot.width), static_cast<int>(snapshot.height));
    state.hasHistory = true;
    state.frameIndex = snapshot.frameIndex;
    state.previousView = snapshot.view;
    state.previousProjection = snapshot.projection;
    state.previousCameraPosition =
        snapshot.view.inverse().transformPoint(cloud::Vec3{0.0f, 0.0f, 0.0f});
    state.accumulatedColor = snapshot.accumulatedColor;
    state.depth = snapshot.depth;
    state.counts = snapshot.counts;
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path outputDirectory =
        argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path("test-output");
    std::error_code error;
    std::filesystem::remove_all(outputDirectory, error);
    std::filesystem::create_directories(outputDirectory, error);
    if (error) {
        std::cerr << "Cannot create test output directory\n";
        return 2;
    }

    runTest("json_parser", [] {
        const cloud::json::Value value =
            cloud::json::parse(R"({"a":[1,2,3],"b":true,"c":"x\ny"})");
        require(value.find("a")->asArray().size() == 3, "array length");
        require(value.find("b")->asBool(), "boolean value");
        const std::string dumped = cloud::json::stringify(value);
        const cloud::json::Value roundTrip = cloud::json::parse(dumped);
        require(roundTrip.find("c")->asString() == "x\ny", "string round trip");
    });

    runTest("crc32_reference", [] {
        const std::string text = "123456789";
        const std::uint32_t checksum =
            cloud::crc32(text.data(), static_cast<std::size_t>(text.size()));
        require(checksum == 0xcbf43926u, "CRC32 IEEE reference vector");
    });

    runTest("density_level_selection", [] {
        require(cloud::chooseDensityResolution(256.0) == 128, "256 MiB selects 128^3");
        require(cloud::chooseDensityResolution(80.0) == 64, "80 MiB selects 64^3");
        require(cloud::chooseDensityResolution(40.0) == 32, "40 MiB selects 32^3");
        bool threw = false;
        try {
            static_cast<void>(cloud::chooseDensityResolution(16.0));
        } catch (const std::exception&) {
            threw = true;
        }
        require(threw, "16 MiB must be rejected");
    });

    cloud::WorkerPool pool(4);
    std::optional<cloud::DensityField> asyncDensity;
    runTest("density_generation_async", [&] {
        asyncDensity.emplace(cloud::DensityField::build(1337, cloud::CloudType::Cumulus, 32, &pool,
                                                       true));
        require(asyncDensity->resolution() == 32, "async density resolution");
        require(asyncDensity->maximumDensity() > asyncDensity->minimumDensity(),
                "density field has variance");
        require(asyncDensity->meanDensity() > 1.0, "density field contains cloud mass");
    });

    runTest("density_generation_deterministic_sync_vs_async", [&] {
        if (!asyncDensity.has_value()) {
            throw std::runtime_error("async density prerequisite failed");
        }
        const cloud::DensityField sync =
            cloud::DensityField::build(1337, cloud::CloudType::Cumulus, 32, nullptr, false);
        require(asyncDensity->data().size() == sync.data().size(), "density byte counts");
        require(asyncDensity->data() == sync.data(), "sync and async density bytes differ");
        require(fnv1a(asyncDensity->data()) == fnv1a(sync.data()), "density hashes differ");
    });

    runTest("density_128_byte_identical", [&] {
        const cloud::DensityField async = cloud::DensityField::build(
            20261114, cloud::CloudType::Cumulus, 128, &pool, true);
        const cloud::DensityField sync = cloud::DensityField::build(
            20261114, cloud::CloudType::Cumulus, 128, nullptr, false);
        require(async.resolution() == 128, "128^3 async resolution");
        require(async.data().size() == 128u * 128u * 128u, "128^3 byte count");
        require(async.data() == sync.data(), "128^3 repeated generation differs");
        std::cout << "  128^3 FNV-1a: 0x" << std::hex << fnv1a(async.data())
                  << std::dec << '\n';
    });

    runTest("real_cumulus_and_stratus_differ", [&] {
        cloud::DensityField cumulus =
            cloud::DensityField::build(481516, cloud::CloudType::Cumulus, 32, &pool, true);
        cloud::DensityField stratus =
            cloud::DensityField::build(481516, cloud::CloudType::Stratus, 32, &pool, true);
        require(cumulus.data() != stratus.data(), "real cumulus/stratus textures differ");
        require(cumulus.maximumDensity() > cumulus.minimumDensity() &&
                    stratus.maximumDensity() > stratus.minimumDensity(),
                "both parameterized shapes contain density variance");
    });

    runTest("density_seed_and_shape_differ", [] {
        const cloud::DensityField otherSeed =
            cloud::DensityField::makeTestPattern(24, 991, cloud::CloudType::Cumulus);
        const cloud::DensityField stratus =
            cloud::DensityField::makeTestPattern(24, 991, cloud::CloudType::Stratus);
        require(otherSeed.data() != stratus.data(), "cloud shape parameter changes texture");
    });

    runTest("snapshot_v2_round_trip", [&] {
        cloud::SnapshotData data;
        data.seed = 1337;
        data.cloudType = 1;
        data.densityResolution = 64;
        data.frameIndex = 17;
        data.width = 16;
        data.height = 9;
        data.historyLength = 12;
        data.nextLight = cloud::defaultLight();
        data.nextLight.ambient = 0.27f;
        data.view = cloud::Mat4::lookAt({2.0f, 1.0f, 2.0f}, {0.5f, 0.5f, 0.5f}, {0, 1, 0});
        data.projection =
            cloud::Mat4::perspective(0.8f, 16.0f / 9.0f, 0.05f, 100.0f);
        data.accumulatedColor.resize(16 * 9);
        data.depth.resize(16 * 9);
        data.counts.resize(16 * 9);
        for (std::size_t index = 0; index < data.accumulatedColor.size(); ++index) {
            data.accumulatedColor[index] = {static_cast<float>(index) * 0.01f, 0.2f, 0.3f};
            data.depth[index] = static_cast<float>(index) * 0.001f;
            data.counts[index] = static_cast<std::uint32_t>(index + 2);
        }
        const std::string path = (outputDirectory / "roundtrip_v2.cld").string();
        cloud::saveSnapshot(path, data, 2);
        const cloud::SnapshotData loaded = cloud::loadSnapshot(path);
        require(loaded.version == 2, "v2 version");
        require(loaded.seed == data.seed, "seed");
        require(loaded.frameIndex == data.frameIndex, "frame index");
        require(loaded.historyLength == data.historyLength, "history length");
        require(std::abs(loaded.nextLight.ambient - data.nextLight.ambient) < 1.0e-7f,
                "lighting ambient");
        require(loaded.accumulatedColor == data.accumulatedColor, "accumulation buffer");
        require(loaded.depth == data.depth, "depth buffer");
        require(loaded.counts == data.counts, "count buffer");
    });

    runTest("snapshot_v1_upgrade_defaults", [&] {
        cloud::SnapshotData data;
        data.seed = 7;
        data.width = 8;
        data.height = 6;
        data.frameIndex = 3;
        data.densityResolution = 128;
        data.view = cloud::Mat4::identity();
        data.projection = cloud::Mat4::identity();
        data.accumulatedColor.assign(48, cloud::Vec3{0.1f, 0.2f, 0.3f});
        data.depth.assign(48, cloud::kSkyDepth);
        data.counts.assign(48, 4);
        const std::string path = (outputDirectory / "upgrade_v1.cld").string();
        cloud::saveSnapshot(path, data, 1);
        const cloud::SnapshotData loaded = cloud::loadSnapshot(path);
        require(loaded.upgradedFromV1, "v1 upgrade marker");
        require(loaded.hasLight, "v1 lighting defaults materialized");
        require(loaded.historyLength == 8, "v1 history length default");
        require(loaded.nextLight.ambient == cloud::defaultLight().ambient,
                "v1 ambient default");
        require(loaded.accumulatedColor == data.accumulatedColor,
                "v1 accumulation buffer offset table");
        require(loaded.depth == data.depth, "v1 depth buffer offset table");
        require(loaded.counts == data.counts, "v1 count buffer offset table");
    });

    runTest("snapshot_crc_and_version_errors", [&] {
        const std::string goodPath = (outputDirectory / "roundtrip_v2.cld").string();
        std::ifstream input(goodPath, std::ios::binary);
        std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                        std::istreambuf_iterator<char>());
        require(bytes.size() > 32, "snapshot file exists");

        std::vector<std::uint8_t> corrupt = bytes;
        corrupt.back() ^= 0x5au;
        const std::string corruptPath = (outputDirectory / "corrupt.cld").string();
        std::ofstream corruptOutput(corruptPath, std::ios::binary | std::ios::trunc);
        corruptOutput.write(reinterpret_cast<const char*>(corrupt.data()),
                            static_cast<std::streamsize>(corrupt.size()));
        corruptOutput.close();
        bool crcCaught = false;
        try {
            static_cast<void>(cloud::loadSnapshot(corruptPath));
        } catch (const cloud::BusinessError& business) {
            crcCaught = business.code() == "E_SNAPSHOT_CRC" &&
                        business.exitCode() == cloud::kExitSnapshotCrc;
        }
        require(crcCaught, "CRC error code and exit code");

        std::vector<std::uint8_t> unknown = bytes;
        unknown[8] = 99;
        unknown[9] = 0;
        unknown[10] = 0;
        unknown[11] = 0;
        const std::string unknownPath = (outputDirectory / "unknown.cld").string();
        std::ofstream unknownOutput(unknownPath, std::ios::binary | std::ios::trunc);
        unknownOutput.write(reinterpret_cast<const char*>(unknown.data()),
                            static_cast<std::streamsize>(unknown.size()));
        unknownOutput.close();
        bool versionCaught = false;
        try {
            static_cast<void>(cloud::loadSnapshot(unknownPath));
        } catch (const cloud::BusinessError& business) {
            versionCaught = business.code() == "E_VERSION_UNSUPPORTED" &&
                            business.exitCode() == cloud::kExitVersionUnsupported;
        }
        require(versionCaught, "unsupported version error code");

        bool missingCaught = false;
        try {
            static_cast<void>(cloud::loadSnapshot((outputDirectory / "missing.cld").string()));
        } catch (const cloud::BusinessError& business) {
            missingCaught = business.code() == "E_SNAPSHOT_MISSING" &&
                            business.exitCode() == cloud::kExitSnapshotMissing;
        }
        require(missingCaught, "missing snapshot error code");
    });

    std::optional<cloud::DensityField> renderDensity;
    runTest("prepare_render_density", [&] {
        if (asyncDensity.has_value()) {
            renderDensity.emplace(std::move(*asyncDensity));
            asyncDensity.reset();
        } else {
            throw std::runtime_error("density prerequisite failed");
        }
    });

    runTest("half_resolution_psnr", [&] {
        if (!renderDensity.has_value()) {
            throw std::runtime_error("render density prerequisite failed");
        }
        cloud::RenderSettings settings;
        settings.width = 320;
        settings.height = 180;
        settings.renderScale = 2;
        settings.maxSteps = 64;
        settings.lightSteps = 5;
        settings.maxHistoryLength = 8;
        cloud::Renderer renderer(settings, *renderDensity);
        const cloud::Camera camera =
            cloud::Camera::fromLookAt({1.85f, 1.2f, 1.85f}, {0.5f, 0.47f, 0.5f}, {0, 1, 0}, 50.0f,
                                      320.0f / 180.0f, 0.05f, 100.0f);
        const cloud::Light light = cloud::defaultLight();
        const cloud::SampleFrame half = renderer.renderSample(camera, light, 2);
        const cloud::SampleFrame full = renderer.renderSample(camera, light, 1);
        const double psnr =
            cloud::Renderer::psnrRGB8(renderer.encodeRGBA(half.color), renderer.encodeRGBA(full.color));
        std::cout << "  measured half/full PSNR: " << psnr << " dB\n";
        require(psnr >= 30.0, "half-resolution upsample PSNR must be at least 30 dB");
    });

    runTest("temporal_snapshot_continuation", [&] {
        if (!renderDensity.has_value()) {
            throw std::runtime_error("render density prerequisite failed");
        }
        cloud::RenderSettings settings;
        settings.width = 200;
        settings.height = 112;
        settings.renderScale = 2;
        settings.maxSteps = 48;
        settings.lightSteps = 4;
        settings.maxHistoryLength = 8;
        cloud::Renderer renderer(settings, *renderDensity);
        const cloud::Light light = cloud::defaultLight();
        const auto cameraAt = [](std::uint32_t frameIndex) {
            const float angle = (35.0f + static_cast<float>(frameIndex) * 2.0f) * cloud::kPi / 180.0f;
            return cloud::Camera::fromLookAt(
                {0.5f + 2.35f * std::cos(angle), 1.42f, 0.5f + 2.35f * std::sin(angle)},
                {0.5f, 0.47f, 0.5f}, {0, 1, 0}, 50.0f, 200.0f / 112.0f, 0.05f, 100.0f);
        };

        cloud::RenderState continuous;
        continuous.configure(settings.width, settings.height);
        for (std::uint32_t frame = 0; frame <= 2; ++frame) {
            const cloud::Camera camera = cameraAt(frame);
            const cloud::SampleFrame sample = renderer.renderSample(camera, light, settings.renderScale);
            const cloud::FrameStats stats =
                renderer.accumulate(continuous, sample, camera, frame);
            require(stats.countsStrictlyMonotonic, "serial accumulation count monotonic");
            require(stats.reprojectionHits + stats.reprojectionMisses ==
                        stats.reprojectionAttempts,
                    "reprojection hit/miss totals");
            if (frame == 1) {
                cloud::AppConfig config;
                config.width = settings.width;
                config.height = settings.height;
                config.maxHistoryLength = settings.maxHistoryLength;
                cloud::saveSnapshot((outputDirectory / "render_interrupt.cld").string(),
                                    snapshotFromState(continuous, config, *renderDensity), 2);
            }
        }

        const cloud::SnapshotData snapshot =
            cloud::loadSnapshot((outputDirectory / "render_interrupt.cld").string());
        cloud::RenderState restored;
        restoreState(restored, snapshot);
        const cloud::Camera camera = cameraAt(2);
        const cloud::SampleFrame sample = renderer.renderSample(camera, light, settings.renderScale);
        const cloud::FrameStats stats = renderer.accumulate(restored, sample, camera, 2);
        std::cout << "  snapshot frame=" << snapshot.frameIndex
                  << " restored state frame=" << restored.frameIndex << '\n';
        require(stats.countsStrictlyMonotonic, "restored accumulation count monotonic");
        require(restored.frameIndex == 2, "restored frame index");
        require(restored.accumulatedColor.size() == continuous.accumulatedColor.size(),
                "restored buffer size");
        require(relativeEqual(restored.accumulatedColor, continuous.accumulatedColor, 1.0e-5),
                "restored final HDR image differs from uninterrupted run");
        require(renderer.encodeRGBA(restored.accumulatedColor) ==
                    renderer.encodeRGBA(continuous.accumulatedColor),
                "restored final RGBA image is not byte-identical");
        require(restored.counts == continuous.counts, "restored counts differ");
    });

    pool.waitIdle();

    if (gFailures != 0) {
        std::cerr << gFailures << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "All tests passed\n";
    return 0;
}

