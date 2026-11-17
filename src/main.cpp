#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "animation.h"
#include "cascade.h"
#include "cli.h"
#include "error.h"
#include "gl_context.h"
#include "image_io.h"
#include "renderer.h"
#include "scene.h"

namespace sr {
namespace {

std::string jsonString(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (char ch : value) {
        switch (ch) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                output << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<int>(
                              static_cast<unsigned char>(ch))
                       << std::dec << std::setfill(' ');
            } else {
                output << ch;
            }
            break;
        }
    }
    output << '"';
    return output.str();
}

std::string jsonNumber(double value) {
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}

std::string jsonVec3(const glm::dvec3& value) {
    std::ostringstream output;
    output << '[' << jsonNumber(value.x) << ',' << jsonNumber(value.y) << ','
           << jsonNumber(value.z) << ']';
    return output.str();
}

std::string jsonMatrix(const glm::dmat4& matrix) {
    std::ostringstream output;
    output << '[';
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            if (column != 0 || row != 0) {
                output << ',';
            }
            output << jsonNumber(matrix[column][row]);
        }
    }
    output << ']';
    return output.str();
}

std::string hex64(std::uint64_t value) {
    std::ostringstream output;
    output << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return output.str();
}

void writeTextFile(const std::filesystem::path& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        fail(ExitCode::OutputIo, "E_OUTPUT_IO",
             "cannot open statistics file: " + path.string());
    }
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    stream.flush();
    if (!stream) {
        fail(ExitCode::OutputIo, "E_OUTPUT_IO",
             "failed to write statistics file: " + path.string());
    }
}

void prepareOutputDirectory(const std::filesystem::path& directory) {
    std::error_code error;
    if (std::filesystem::exists(directory, error)) {
        if (!std::filesystem::is_directory(directory, error)) {
            fail(ExitCode::OutputIo, "E_OUTPUT_IO",
                 "output path exists but is not a directory: " + directory.string());
        }
    } else {
        std::filesystem::create_directories(directory, error);
        if (error) {
            fail(ExitCode::OutputIo, "E_OUTPUT_IO",
                 "cannot create output directory: " + directory.string() +
                     " (" + error.message() + ")");
        }
    }

    const std::filesystem::path probe = directory / ".write_probe";
    {
        std::ofstream stream(probe, std::ios::binary | std::ios::trunc);
        if (!stream) {
            fail(ExitCode::OutputIo, "E_OUTPUT_IO",
                 "output directory is not writable: " + directory.string());
        }
        stream << "write-test\n";
        stream.flush();
        if (!stream) {
            fail(ExitCode::OutputIo, "E_OUTPUT_IO",
                 "output directory write probe failed: " + directory.string());
        }
    }
    std::filesystem::remove(probe, error);
    if (error) {
        fail(ExitCode::OutputIo, "E_OUTPUT_IO",
             "cannot remove output write probe: " + error.message());
    }

    const std::filesystem::path statsDirectory = directory / "stats";
    std::filesystem::create_directories(statsDirectory, error);
    if (error) {
        fail(ExitCode::OutputIo, "E_OUTPUT_IO",
             "cannot create statistics directory: " + statsDirectory.string() +
                 " (" + error.message() + ")");
    }
}

std::string frameNumber(int frame) {
    std::ostringstream output;
    output << std::setw(6) << std::setfill('0') << frame;
    return output.str();
}

std::string materialJson(const Material& material) {
    std::ostringstream output;
    output << "{\"baseColor\":" << jsonVec3(material.baseColor)
           << ",\"roughness\":" << jsonNumber(material.roughness)
           << ",\"specular\":" << jsonNumber(material.specular)
           << ",\"checkerScale\":" << jsonNumber(material.checkerScale)
           << '}';
    return output.str();
}

std::string lightsJson(const Lights& lights) {
    std::ostringstream output;
    output << "{"
           << "\"directional\":{\"directionTowardScene\":"
           << jsonVec3(lights.directional.directionTowardScene)
           << ",\"color\":" << jsonVec3(lights.directional.color)
           << ",\"intensity\":" << jsonNumber(lights.directional.intensity)
           << "},"
           << "\"point0\":{\"position\":" << jsonVec3(lights.point0.position)
           << ",\"range\":" << jsonNumber(lights.point0.range)
           << ",\"color\":" << jsonVec3(lights.point0.color)
           << ",\"intensity\":" << jsonNumber(lights.point0.intensity)
           << "},"
           << "\"point1\":{\"position\":" << jsonVec3(lights.point1.position)
           << ",\"range\":" << jsonNumber(lights.point1.range)
           << ",\"color\":" << jsonVec3(lights.point1.color)
           << ",\"intensity\":" << jsonNumber(lights.point1.intensity)
           << "},"
           << "\"ambientColor\":" << jsonVec3(lights.ambientColor)
           << ",\"ambientIntensity\":" << jsonNumber(lights.ambientIntensity)
           << '}';
    return output.str();
}

std::string buildFrameStats(const Options& options,
                            int frame,
                            double timeSeconds,
                            const std::string& contextDescription,
                            const std::string& glVersion,
                            const std::string& glRenderer,
                            const Camera& camera,
                            const Bounds& visibleBounds,
                            const PoseResult& pose,
                            const Skeleton& skeleton,
                            const MeshRevisionInfo& meshInfo,
                            const CascadeSet& cascades,
                            const Lights& lights,
                            const Material& characterMaterial,
                            const Material& occluderMaterial,
                            const Material& groundMaterial,
                            const FrameRenderResult& renderResult,
                            const std::string& colorFile,
                            const std::string& depthFile,
                            std::uint64_t colorHash,
                            std::uint64_t depthHash,
                            std::uint64_t deformedPositionHash) {
    std::ostringstream output;
    output << std::setprecision(17);
    output << "{\n";
    output << "  \"frame\": " << frame << ",\n";
    output << "  \"timeSeconds\": " << jsonNumber(timeSeconds) << ",\n";
    output << "  \"resolution\": {\"width\": " << options.width
           << ", \"height\": " << options.height << "},\n";
    output << "  \"context\": " << jsonString(contextDescription) << ",\n";
    output << "  \"glVersion\": " << jsonString(glVersion) << ",\n";
    output << "  \"glRenderer\": " << jsonString(glRenderer) << ",\n";
    output << "  \"mesh\": {\"revision\": " << meshInfo.revision
           << ", \"variant\": " << meshInfo.variant
           << ", \"vertices\": " << meshInfo.vertexCount
           << ", \"triangles\": " << meshInfo.triangleCount
           << ", \"maxInfluences\": " << meshInfo.maxInfluences
           << ", \"minWeightSum\": " << jsonNumber(meshInfo.minWeightSum)
           << ", \"maxWeightSum\": " << jsonNumber(meshInfo.maxWeightSum)
           << "},\n";
    output << "  \"camera\": {\"eye\": " << jsonVec3(camera.eye)
           << ", \"target\": " << jsonVec3(camera.target)
           << ", \"near\": " << jsonNumber(camera.nearPlane)
           << ", \"far\": " << jsonNumber(camera.farPlane)
           << ", \"aspect\": " << jsonNumber(camera.aspect) << "},\n";
    output << "  \"visibleCasterBounds\": {\"min\": "
           << jsonVec3(visibleBounds.min) << ", \"max\": "
           << jsonVec3(visibleBounds.max) << "},\n";

    output << "  \"pose\": {\"bones\": [";
    for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
        if (i != 0) {
            output << ", ";
        }
        output << "{\"name\": " << jsonString(skeleton.bones[i].name)
               << ", \"parent\": " << skeleton.bones[i].parent
               << ", \"segmentIndex\": " << pose.samples[i].segmentIndex
               << ", \"segmentTimes\": [" << jsonNumber(pose.samples[i].segmentTime0)
               << ", " << jsonNumber(pose.samples[i].segmentTime1) << "]"
               << ", \"fraction\": " << jsonNumber(pose.samples[i].fraction)
               << ", \"clampedBefore\": "
               << (pose.samples[i].clampedBefore ? "true" : "false")
               << ", \"clampedAfter\": "
               << (pose.samples[i].clampedAfter ? "true" : "false")
               << '}';
    }
    output << "], \"worldMatrices\": [";
    for (std::size_t i = 0; i < pose.worldMatrices.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << jsonMatrix(pose.worldMatrices[i]);
    }
    output << "], \"skinMatrices\": [";
    for (std::size_t i = 0; i < pose.skinMatrices.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << jsonMatrix(pose.skinMatrices[i]);
    }
    output << "], \"cpuDeformedPositionFNV64\": "
           << jsonString(hex64(deformedPositionHash)) << "},\n";

    output << "  \"cascadeConfig\": {\"count\": " << options.cascadeCount
           << ", \"shadowMapResolution\": " << options.shadowSize
           << ", \"splitLambda\": " << jsonNumber(options.splitLambda)
           << ", \"sampling\": \"3x3 PCF, CLAMP_TO_BORDER\""
           << ", \"depthBias\": 0.00045"
           << ", \"slopeBias\": 0.0028"
           << ", \"shadowPolygonOffset\": [2.0, 4.0]},\n";
    output << "  \"cascades\": [";
    for (std::size_t i = 0; i < cascades.cascades.size(); ++i) {
        const Cascade& cascade = cascades.cascades[i];
        if (i != 0) {
            output << ", ";
        }
        output << "{\"index\": " << cascade.index
               << ", \"viewDepthRange\": [" << jsonNumber(cascade.splitNear)
               << ", " << jsonNumber(cascade.splitFar) << "]"
               << ", \"occupiedPixelFraction\": "
               << jsonNumber(i < renderResult.shadowCoverage.size()
                                 ? renderResult.shadowCoverage[i]
                                 : 0.0)
               << ", \"lightCenter\": " << jsonVec3(cascade.lightCenter)
               << ", \"radius\": " << jsonNumber(cascade.radius)
               << ", \"texelWorldSize\": " << jsonNumber(cascade.texelWorldSize)
               << ", \"lightViewProjection\": "
               << jsonMatrix(cascade.lightViewProjection) << '}';
    }
    output << "],\n";
    output << "  \"lights\": " << lightsJson(lights) << ",\n";
    output << "  \"materials\": {\"character\": "
           << materialJson(characterMaterial) << ", \"occluders\": "
           << materialJson(occluderMaterial) << ", \"ground\": "
           << materialJson(groundMaterial) << "},\n";
    output << "  \"timingMs\": {\"shadow\": "
           << jsonNumber(renderResult.shadowPassMs)
           << ", \"depthPrepass\": " << jsonNumber(renderResult.depthPrepassMs)
           << ", \"color\": " << jsonNumber(renderResult.colorPassMs)
           << ", \"readback\": " << jsonNumber(renderResult.readbackMs)
           << ", \"total\": " << jsonNumber(renderResult.totalMs) << "},\n";
    output << "  \"images\": {\"color\": " << jsonString(colorFile)
           << ", \"depth\": " << jsonString(depthFile)
           << ", \"colorPixelFNV64\": " << jsonString(hex64(colorHash))
           << ", \"depthPixelFNV64\": " << jsonString(hex64(depthHash))
           << "}\n";
    output << "}\n";
    return output.str();
}

std::uint64_t hashDeformedPositions(const MeshData& mesh) {
    std::vector<float> positions;
    positions.reserve(mesh.vertices.size() * 3U);
    for (const SkinnedVertex& vertex : mesh.vertices) {
        positions.push_back(vertex.px);
        positions.push_back(vertex.py);
        positions.push_back(vertex.pz);
    }
    return fnv1a64(positions.data(), positions.size() * sizeof(float));
}

int run(const Options& options) {
    validateCascadeConfig(CascadeConfig{
        options.cascadeCount, options.shadowSize, options.splitLambda, 1.25});
    prepareOutputDirectory(options.outputDirectory);

    auto context = createGLContext(options.context);
    context->makeCurrent();

    RendererSettings rendererSettings;
    rendererSettings.width = options.width;
    rendererSettings.height = options.height;
    rendererSettings.shadowSize = options.shadowSize;
    rendererSettings.cascadeCount = options.cascadeCount;
    rendererSettings.forceShaderError = options.testError == "shader";
    rendererSettings.forceFramebufferError = options.testError == "framebuffer";

    Renderer renderer(*context, rendererSettings);
    DeterministicScene scene(options.meshVariant);
    renderer.uploadCharacter(scene.characterBindMesh());
    renderer.uploadStaticMeshes(scene.occluderMesh(), scene.groundMesh());

    const Lights lights = scene.lights();
    const Material characterMaterial = scene.characterMaterial();
    const Material occluderMaterial = scene.occluderMaterial();
    const Material groundMaterial = scene.groundMaterial();
    const CascadeConfig cascadeConfig{
        options.cascadeCount, options.shadowSize, options.splitLambda, 1.25};

    double totalFrameMs = 0.0;
    std::vector<std::string> imageRecords;
    std::cout << "Context: " << context->description() << '\n';
    std::cout << "Output: " << std::filesystem::absolute(options.outputDirectory)
              << '\n';

    for (int offset = 0; offset < options.frameCount; ++offset) {
        const int frame = options.startFrame + offset;
        const double timeSeconds = static_cast<double>(frame) / options.fps;

        if (frame == options.regenerateFrame) {
            scene.rebuildCharacter(options.regenerateVariant);
            renderer.uploadCharacter(scene.characterBindMesh());
            std::cout << "Mesh rebuilt in-process before frame " << frame
                      << " (revision " << scene.revisionInfo().revision
                      << ", variant " << scene.revisionInfo().variant << ")\n";
        }

        const PoseResult pose = evaluatePose(scene.skeleton(), timeSeconds);
        Camera camera = evaluateCamera(timeSeconds);
        camera.aspect = static_cast<double>(options.width) /
                        static_cast<double>(options.height);
        updateCameraMatrices(camera);
        const Bounds visibleBounds = scene.visibleCasterBounds(pose);
        const CascadeSet cascades =
            buildCascades(camera, visibleBounds, lights.directional, cascadeConfig);
        const MeshData deformed = scene.deformedCharacter(pose);

        const FrameRenderResult result = renderer.renderFrame(
            pose, camera, cascades, lights,
            characterMaterial, occluderMaterial, groundMaterial);

        const std::string number = frameNumber(frame);
        const std::string colorName = "frame_" + number + "_color.png";
        const std::string depthName = "frame_" + number + "_depth.png";
        const std::filesystem::path colorPath = options.outputDirectory / colorName;
        const std::filesystem::path depthPath = options.outputDirectory / depthName;
        writeColorPngRgba8(colorPath, options.width, options.height,
                           result.colorRgbaTopDown);
        writeDepthPng16(depthPath, options.width, options.height,
                        result.depthTopDown);

        const std::uint64_t colorHash = fnv1a64(
            result.colorRgbaTopDown.data(), result.colorRgbaTopDown.size());
        const std::uint64_t depthHash = fnv1a64(
            result.depthTopDown.data(),
            result.depthTopDown.size() * sizeof(float));
        const std::uint64_t deformedHash = hashDeformedPositions(deformed);

        const std::string stats = buildFrameStats(
            options, frame, timeSeconds, context->description(),
            renderer.glVersion(), renderer.glRenderer(), camera,
            visibleBounds, pose, scene.skeleton(), scene.revisionInfo(),
            cascades, lights, characterMaterial, occluderMaterial,
            groundMaterial, result, colorName, depthName,
            colorHash, depthHash, deformedHash);
        writeTextFile(options.outputDirectory / "stats" /
                          ("frame_" + number + ".json"), stats);

        totalFrameMs += result.totalMs;
        std::ostringstream imageRecord;
        imageRecord << "{\"frame\":" << frame << ",\"color\":"
                    << jsonString(colorName) << ",\"depth\":"
                    << jsonString(depthName) << ",\"colorPixelFNV64\":"
                    << jsonString(hex64(colorHash)) << ",\"depthPixelFNV64\":"
                    << jsonString(hex64(depthHash)) << '}';
        imageRecords.push_back(imageRecord.str());

        std::cout << "Frame " << frame << ": " << options.width << 'x'
                  << options.height << ", cascades="
                  << cascades.cascades.size() << "x"
                  << options.shadowSize << ", shadow="
                  << std::fixed << std::setprecision(2)
                  << result.shadowPassMs << " ms, depth="
                  << result.depthPrepassMs << " ms, color="
                  << result.colorPassMs << " ms, total="
                  << result.totalMs << " ms\n" << std::defaultfloat;
    }

    std::ostringstream summary;
    summary << std::setprecision(17);
    summary << "{\n";
    summary << "  \"program\": \"skinned_csm_renderer\",\n";
    summary << "  \"context\": " << jsonString(context->description()) << ",\n";
    summary << "  \"glVersion\": " << jsonString(renderer.glVersion()) << ",\n";
    summary << "  \"glRenderer\": " << jsonString(renderer.glRenderer()) << ",\n";
    summary << "  \"startFrame\": " << options.startFrame << ",\n";
    summary << "  \"frameCount\": " << options.frameCount << ",\n";
    summary << "  \"fps\": " << jsonNumber(options.fps) << ",\n";
    summary << "  \"resolution\": {\"width\": " << options.width
            << ", \"height\": " << options.height << "},\n";
    summary << "  \"shadowMapResolution\": " << options.shadowSize << ",\n";
    summary << "  \"cascadeCount\": " << options.cascadeCount << ",\n";
    summary << "  \"splitLambda\": " << jsonNumber(options.splitLambda) << ",\n";
    summary << "  \"initialMeshVariant\": " << options.meshVariant << ",\n";
    summary << "  \"regenerateFrame\": " << options.regenerateFrame << ",\n";
    summary << "  \"regenerateVariant\": " << options.regenerateVariant << ",\n";
    summary << "  \"totalFrameCpuMs\": " << jsonNumber(totalFrameMs) << ",\n";
    summary << "  \"images\": [";
    for (std::size_t i = 0; i < imageRecords.size(); ++i) {
        if (i != 0) {
            summary << ',';
        }
        summary << imageRecords[i];
    }
    summary << "],\n";
    summary << "  \"determinism\": \"No randomness, wall-clock time, or external "
               "assets are used for pixels. Repeated runs with identical parameters "
               "and the same binary/GPU/driver produce byte-identical PNG files; "
               "timings are intentionally not deterministic.\"\n";
    summary << "}\n";
    writeTextFile(options.outputDirectory / "stats" / "summary.json",
                  summary.str());

    std::cout << "Rendered " << options.frameCount << " frame(s) in "
              << std::fixed << std::setprecision(2) << totalFrameMs
              << " ms (CPU wall time). Statistics: "
              << (options.outputDirectory / "stats" / "summary.json")
              << std::defaultfloat << '\n';
    return static_cast<int>(ExitCode::Ok);
}

} // namespace
} // namespace sr

int main(int argc, char** argv) {
    try {
        const sr::Options options = sr::parseCommandLine(argc, argv);
        if (options.helpRequested) {
            std::cout << sr::usageText();
            return static_cast<int>(sr::ExitCode::Ok);
        }
        return sr::run(options);
    } catch (const sr::AppError& error) {
        std::cerr << "error[" << error.symbol() << "]: " << error.what() << '\n';
        return static_cast<int>(error.code());
    } catch (const std::exception& error) {
        std::cerr << "error[E_INTERNAL]: " << error.what() << '\n';
        return static_cast<int>(sr::ExitCode::Internal);
    } catch (...) {
        std::cerr << "error[E_INTERNAL]: unknown fatal error\n";
        return static_cast<int>(sr::ExitCode::Internal);
    }
}
