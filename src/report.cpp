#include "report.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

namespace skel {
namespace {

using Json = nlohmann::ordered_json;

Json cameraJson(const Camera& camera) {
    Json result;
    result["eye"] = {camera.eye.x, camera.eye.y, camera.eye.z};
    result["target"] = {camera.target.x, camera.target.y, camera.target.z};
    result["up"] = {camera.up.x, camera.up.y, camera.up.z};
    result["fov_y_degrees"] = camera.fieldOfViewYDegrees;
    result["near"] = camera.nearPlane;
    result["far"] = camera.farPlane;
    return result;
}

Json buildJson(const RenderReport& report) {
    Json root;
    root["schema"] = "skelpreview.statistics.v1";
    root["asset"] = {
        {"file", report.assetFile},
        {"frame_count", report.assetFrameCount},
        {"fps", report.framesPerSecond},
        {"composition_order",
         compositionOrderName(report.compositionOrder)},
    };
    root["render"] = {
        {"mode", report.mode},
        {"width", report.width},
        {"height", report.height},
        {"camera", cameraJson(report.camera)},
        {"total_frames", report.frames.size()},
        {"vertex_count", report.vertexCount},
        {"triangle_count", report.triangleCount},
        {"normalized_vertex_count", report.normalizedVertexCount},
        {"normalization_policy",
         "divide all contributing weights by their sum when "
         "abs(sum-1) > 1e-6"},
        {"nan_positions", report.totalNanPositions()},
        {"nan_normals", report.totalNanNormals()},
    };

    Json frames = Json::array();
    for (const FrameRecord& record : report.frames) {
        frames.push_back({
            {"frame", record.frame},
            {"nan_positions", record.stats.nanPositions},
            {"nan_normals", record.stats.nanNormals},
            {"normalized_vertex_count",
             record.stats.normalizedVertices},
            {"depth_file", record.depthFile},
            {"color_file", record.colorFile},
            {"elapsed_ms", record.elapsedMs},
        });
    }
    root["frames"] = std::move(frames);
    root["timing"] = {
        {"clock", "steady_clock"},
        {"total_elapsed_ms", report.elapsedMs},
    };
    return root;
}

} // namespace

std::uint64_t RenderReport::totalNanPositions() const {
    std::uint64_t total = 0;
    for (const FrameRecord& frame : frames) {
        total += frame.stats.nanPositions;
    }
    return total;
}

std::uint64_t RenderReport::totalNanNormals() const {
    std::uint64_t total = 0;
    for (const FrameRecord& frame : frames) {
        total += frame.stats.nanNormals;
    }
    return total;
}

std::string statisticsJson(const RenderReport& report) {
    return buildJson(report).dump(2) + "\n";
}

bool writeStatisticsJson(const std::filesystem::path& path,
                         const RenderReport& report, Error& error) {
    const std::string text = statisticsJson(report);
    try {
        const Json verification = Json::parse(text);
        if (!verification.contains("frames") ||
            !verification["frames"].is_array()) {
            error = makeError(ErrorCode::InvalidJson,
                              "generated statistics JSON lacks a frames array");
            return false;
        }
    } catch (const nlohmann::json::exception& exception) {
        error = makeError(ErrorCode::InvalidJson,
                          "generated statistics JSON failed to parse: " +
                              std::string(exception.what()));
        return false;
    }

    if (!path.parent_path().empty()) {
        std::error_code filesystemError;
        std::filesystem::create_directories(path.parent_path(),
                                            filesystemError);
        if (filesystemError) {
            error = makeError(ErrorCode::WriteFailed,
                              "cannot create report directory: " +
                                  filesystemError.message());
            return false;
        }
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        error = makeError(ErrorCode::WriteFailed,
                          "cannot open report output: " + path.string());
        return false;
    }
    stream << text;
    stream.flush();
    if (!stream) {
        error = makeError(ErrorCode::WriteFailed,
                          "failed to write report output: " + path.string());
        return false;
    }

    // The report is an interface consumed by pipeline tools. Validate what was
    // actually persisted and classify malformed JSON as an asset error.
    std::ifstream verificationStream(path, std::ios::binary);
    std::ostringstream readBack;
    readBack << verificationStream.rdbuf();
    RenderReport ignored;
    if (!verificationStream || !parseStatisticsJson(readBack.str(), ignored,
                                                   error)) {
        if (!error) {
            error = makeError(ErrorCode::ReadFailed,
                              "cannot read back statistics report");
        }
        return false;
    }
    error = {};
    return true;
}

bool parseStatisticsJson(const std::string& text, RenderReport& report,
                         Error& error) {
    try {
        const Json root = Json::parse(text);
        if (!root.contains("schema") || !root["schema"].is_string() ||
            root["schema"].get<std::string>() !=
                "skelpreview.statistics.v1") {
            error = makeError(ErrorCode::InvalidJson,
                              "statistics schema identifier is missing");
            return false;
        }
        if (!root.contains("render") || !root["render"].is_object() ||
            !root.contains("frames") || !root["frames"].is_array()) {
            error = makeError(ErrorCode::InvalidJson,
                              "statistics document is incomplete");
            return false;
        }
        RenderReport parsed;
        const Json& render = root["render"];
        parsed.mode = render.value("mode", std::string{});
        parsed.width = render.value("width", 0U);
        parsed.height = render.value("height", 0U);
        parsed.vertexCount = render.value("vertex_count", std::size_t{0});
        parsed.triangleCount =
            render.value("triangle_count", std::size_t{0});
        parsed.normalizedVertexCount =
            render.value("normalized_vertex_count", 0U);
        for (const Json& frame : root["frames"]) {
            FrameRecord record;
            record.frame = frame.value("frame", 0U);
            record.stats.frame = record.frame;
            record.stats.nanPositions =
                frame.value("nan_positions", std::uint64_t{0});
            record.stats.nanNormals =
                frame.value("nan_normals", std::uint64_t{0});
            record.stats.normalizedVertices =
                frame.value("normalized_vertex_count", 0U);
            record.depthFile = frame.value("depth_file", std::string{});
            record.colorFile = frame.value("color_file", std::string{});
            record.elapsedMs = frame.value("elapsed_ms", 0.0);
            parsed.frames.push_back(std::move(record));
        }
        if (root.contains("asset") && root["asset"].is_object()) {
            const Json& asset = root["asset"];
            parsed.assetFile = asset.value("file", std::string{});
            parsed.assetFrameCount = asset.value("frame_count", 0U);
            parsed.framesPerSecond = asset.value("fps", 0.0F);
            const std::string order =
                asset.value("composition_order", std::string{});
            for (std::uint32_t value = 0;
                 value <= static_cast<std::uint32_t>(CompositionOrder::SRT);
                 ++value) {
                const auto candidate = static_cast<CompositionOrder>(value);
                if (order == compositionOrderName(candidate)) {
                    parsed.compositionOrder = candidate;
                    break;
                }
            }
        }
        report = std::move(parsed);
        error = {};
        return true;
    } catch (const nlohmann::json::exception& exception) {
        error = makeError(ErrorCode::InvalidJson,
                          "statistics JSON parse failed: " +
                              std::string(exception.what()));
        return false;
    }
}

} // namespace skel
