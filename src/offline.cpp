#include "offline.h"

#include "error.h"
#include "image_io.h"
#include "scene.h"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace df {
namespace {

using Json = nlohmann::json;

Json vec3Json(const Vec3& value) {
    return Json::array({value.x, value.y, value.z});
}

std::uint64_t fnv1a(const std::uint8_t* data, std::size_t size) {
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

std::uint64_t fnv1a(const float* data, std::size_t size) {
    return fnv1a(reinterpret_cast<const std::uint8_t*>(data), size * sizeof(float));
}

void writeJson(const std::string& path, const Json& value) {
    std::ofstream out(path);
    if (!out) {
        throw AppError(ErrorCode::E_IO, "cannot open JSON for writing: " + path);
    }
    out << value.dump(2) << '\n';
    if (!out) {
        throw AppError(ErrorCode::E_IO, "failed while writing JSON: " + path);
    }
}

void writeCString(const std::string& path, const char* data) {
    std::ofstream out(path);
    if (!out) {
        throw AppError(ErrorCode::E_IO, "cannot open file for writing: " + path);
    }
    out << data;
    if (!out) {
        throw AppError(ErrorCode::E_IO, "failed while writing file: " + path);
    }
}

} // namespace

OfflineArtifacts writeFrameArtifacts(FrameOutput frame, const Scene& scene,
                                     const RenderConfig& config,
                                     const std::string& outputDirectory) {
    OfflineArtifacts artifacts;
    artifacts.outputDirectory = outputDirectory;
    std::error_code ec;
    std::filesystem::create_directories(outputDirectory, ec);
    if (ec) {
        throw AppError(ErrorCode::E_IO,
                       "cannot create output directory '" + outputDirectory + "': " + ec.message());
    }

    artifacts.frame = std::move(frame);

    artifacts.colorPng = outputDirectory + "/color.png";
    artifacts.depthPfm = outputDirectory + "/depth.pfm";
    artifacts.transparentJson = outputDirectory + "/transparent_objects.json";
    artifacts.metadataJson = outputDirectory + "/frame_metadata.json";

    writePngRgb(artifacts.colorPng, artifacts.frame.width, artifacts.frame.height,
                artifacts.frame.colorRgb);
    writePfm(artifacts.depthPfm, artifacts.frame.width, artifacts.frame.height,
             artifacts.frame.linearDepth);

    artifacts.colorHash =
        fnv1a(artifacts.frame.colorRgb.data(), artifacts.frame.colorRgb.size());
    artifacts.depthHash =
        fnv1a(artifacts.frame.linearDepth.data(), artifacts.frame.linearDepth.size());

    Json transparent = {
        {"schema", "deferred-fog-transparent-list/v1"},
        {"sortMode", sortModeName(artifacts.frame.transparent.mode)},
        {"sortingEnabled", artifacts.frame.transparent.mode != SortMode::Disabled},
        {"approximate", artifacts.frame.transparent.approximate},
        {"premultipliedAlpha", true},
        {"doesWriteFogDepth", false},
        {"objects", Json::array()},
    };
    int rank = 0;
    for (const SortedTransparent& object : artifacts.frame.transparent.objects) {
        transparent["objects"].push_back({
            {"rank", rank++},
            {"id", object.id},
            {"viewDepth", object.depth},
            {"opacity", object.opacity},
            {"center", vec3Json(object.center)},
        });
    }
    writeJson(artifacts.transparentJson, transparent);

    Json lights = Json::array();
    for (const Light& light : scene.lights) {
        lights.push_back({
            {"id", light.id},
            {"type", light.type == LightType::Point ? "point" : "directional"},
            {"position", vec3Json(light.position)},
            {"direction", vec3Json(light.direction)},
            {"color", vec3Json(light.color)},
            {"intensity", light.intensity},
        });
    }

    Json metadata = {
        {"schema", "deferred-fog-frame/v1"},
        {"scene", scene.name},
        {"backend", artifacts.frame.stats.backend},
        {"resolution", {{"width", artifacts.frame.width}, {"height", artifacts.frame.height}}},
        {"camera",
         {{"position", vec3Json(scene.camera.position)},
          {"target", vec3Json(scene.camera.target)},
          {"up", vec3Json(scene.camera.up)},
          {"fovYDegrees", scene.camera.fovYDegrees},
          {"nearPlane", scene.camera.nearPlane},
          {"farPlane", scene.camera.farPlane}}},
        {"lights", lights},
        {"lightSnapshotVersion", artifacts.frame.stats.lightVersion},
        {"lightUploads",
         {{"lighting", artifacts.frame.stats.lightingUploads},
          {"volumetric", artifacts.frame.stats.fogUploads}}},
        {"volumetric",
         {{"steps", artifacts.frame.stats.fogSteps},
          {"offlineFixedSteps", true},
          {"densityScale", config.volumetric.densityScale},
          {"maxDistance", config.volumetric.maxDistance}}},
        {"transparent",
         {{"sortMode", sortModeName(artifacts.frame.transparent.mode)},
          {"count", artifacts.frame.transparent.objects.size()},
          {"drawn", artifacts.frame.stats.transparentDrawn},
          {"transparentDepthWrites", artifacts.frame.stats.transparentDepthWrites}}},
        {"gbuffer",
         {{"targets",
           Json::array({
               {{"name", "gPosition"}, {"format", "RGBA32F"}, {"fields", {"position.xyz", "viewDepth"}}},
               {{"name", "gNormal"}, {"format", "RGBA32F"}, {"fields", {"normal.xyz", "objectId"}}},
               {{"name", "gAlbedo"}, {"format", "RGBA32F"}, {"fields", {"albedo.rgb", "roughness"}}},
               {{"name", "gMaterial"}, {"format", "RGBA32F"}, {"fields", {"metallic", "reserved", "reserved", "valid"}}},
           })},
          {"minComponentBits", 16},
          {"activeFormat", "RGBA32F"}}},
        {"hashes",
         {{"colorFnv1a64", artifacts.colorHash},
          {"depthFnv1a64", artifacts.depthHash}}},
        {"timing", {{"cpuMilliseconds", artifacts.frame.stats.cpuMilliseconds}}},
    };
    if (!artifacts.frame.stats.gpuRenderer.empty()) {
        metadata["gpu"] = {
            {"renderer", artifacts.frame.stats.gpuRenderer},
            {"version", artifacts.frame.stats.gpuVersion},
        };
    }
    writeJson(artifacts.metadataJson, metadata);

    writeCString(outputDirectory + "/gbuffer_layout.txt",
                 "G-Buffer layout (MRT)\n"
                 "0: RGBA32F gPosition = position.xyz, viewDepth\n"
                 "1: RGBA32F gNormal   = normal.xyz, objectId\n"
                 "2: RGBA32F gAlbedo   = albedo.rgb, roughness\n"
                 "3: RGBA32F gMaterial = metallic, reserved, reserved, valid\n"
                 "All color attachments use 32-bit float components (>= 16-bit requirement).\n");

    return artifacts;
}

OfflineArtifacts renderOffline(const RenderConfig& config,
                               const std::string& outputDirectory) {
    const Scene scene = makeDeterministicScene(config.width, config.height);
    CpuRenderer renderer;
    FrameOutput frame = renderer.render(scene, config, true, true);
    return writeFrameArtifacts(std::move(frame), scene, config, outputDirectory);
}

} // namespace df

