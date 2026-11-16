#include "test_runner.hpp"

#include "asset_reader.hpp"
#include "offline_renderer.hpp"
#include "report.hpp"
#include "sample_generator.hpp"
#include "scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <nlohmann/json.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace skel {
namespace {

class TestHarness {
public:
    explicit TestHarness(std::filesystem::path directory)
        : directory_(std::move(directory)) {
        std::error_code error;
        std::filesystem::create_directories(directory_, error);
        if (error) {
            std::cerr << "cannot create test directory: " << error.message()
                      << "\n";
        }
    }

    void check(bool condition, const std::string& name) {
        ++checks_;
        if (condition) {
            ++passed_;
            std::cout << "[PASS] " << name << "\n";
        } else {
            ++failed_;
            std::cout << "[FAIL] " << name << "\n";
        }
    }

    void expectError(ErrorCode actual, ErrorCode expected,
                     const std::string& name) {
        check(actual == expected,
              name + " (expected " + std::to_string(
                  static_cast<int>(expected)) + " / " +
                  errorCodeName(expected) + ", got " +
                  std::to_string(static_cast<int>(actual)) + " / " +
                  errorCodeName(actual) + ")");
    }

    const std::filesystem::path& directory() const { return directory_; }

    int summary() const {
        std::cout << "\n" << passed_ << "/" << checks_
                  << " checks passed, " << failed_ << " failed\n";
        return failed_ == 0 ? 0 : 1;
    }

private:
    std::filesystem::path directory_;
    int checks_ = 0;
    int passed_ = 0;
    int failed_ = 0;
};

bool near(float actual, float expected, float tolerance = 1.0e-5F) {
    return std::abs(actual - expected) <= tolerance;
}

bool matrixNear(const glm::mat4& actual, const glm::mat4& expected,
                float tolerance = 1.0e-5F) {
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            if (!near(actual[column][row], expected[column][row],
                      tolerance)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<std::uint8_t> readBytes(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return {};
    }
    const std::streamoff size = stream.tellg();
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    stream.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()), size);
    }
    return bytes;
}

std::string readText(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream),
                       std::istreambuf_iterator<char>());
}

Asset makeSingleBoneAsset(const glm::vec3& translation,
                          const glm::quat& rotation,
                          const glm::vec3& scale,
                          const glm::mat4& inverseBind,
                          const glm::vec3& position,
                          const glm::vec3& normal) {
    Asset asset;
    asset.compositionOrder = CompositionOrder::TRS;
    asset.frameCount = 1;
    asset.framesPerSecond = 24.0F;
    Node node;
    node.name = "bone";
    node.firstSample = 0;
    node.sampleCount = 1;
    node.inverseBind = inverseBind;
    asset.nodes.push_back(node);
    asset.samples.push_back(
        AnimationSample{0.0F, translation, rotation, scale});

    Mesh mesh;
    mesh.name = "mesh";
    Vertex vertex;
    vertex.position = position;
    vertex.normal = normal;
    vertex.color = glm::vec3(1.0F);
    vertex.weightCount = 1;
    vertex.boneIndices = {0, 0, 0, 0};
    vertex.weights = {1.0F, 0.0F, 0.0F, 0.0F};
    mesh.vertices.push_back(vertex);
    asset.meshes.push_back(std::move(mesh));
    return asset;
}

glm::mat4 referenceComposition(CompositionOrder order,
                               const glm::vec3& translation,
                               const glm::quat& rotation,
                               const glm::vec3& scale) {
    const glm::mat4 t = glm::translate(glm::mat4(1.0F), translation);
    const glm::mat4 r = glm::mat4_cast(glm::normalize(rotation));
    const glm::mat4 s = glm::scale(glm::mat4(1.0F), scale);
    switch (order) {
    case CompositionOrder::TRS: return t * r * s;
    case CompositionOrder::TSR: return t * s * r;
    case CompositionOrder::RTS: return r * t * s;
    case CompositionOrder::RST: return r * s * t;
    case CompositionOrder::STR: return s * t * r;
    case CompositionOrder::SRT: return s * r * t;
    }
    return glm::mat4(0.0F);
}

void testReader(TestHarness& harness) {
    const std::filesystem::path normalPath =
        harness.directory() / "normal.ska";
    Error error;
    harness.check(
        generateAssetFile("normal", CompositionOrder::TRS, normalPath,
                          error),
        "generator writes normal asset");

    Asset asset;
    harness.check(AssetReader::load(normalPath, asset, error),
                  "reader loads normal asset");
    harness.check(asset.nodes.size() == 5U, "reader node count");
    harness.check(asset.meshes.size() == 2U, "reader mesh count");
    harness.check(asset.vertexCount() == 16U, "reader vertex count");
    harness.check(asset.triangleCount() == 24U,
                  "reader triangle count");
    harness.check(asset.samples.size() == 120U, "reader sample count");

    struct ExpectedError {
        const char* preset;
        ErrorCode code;
    };
    const std::array<ExpectedError, 9> expected{{
        {"cycle", ErrorCode::ParentCycle},
        {"oob-index", ErrorCode::InvalidTriangleIndex},
        {"oob-bone", ErrorCode::InvalidBoneIndex},
        {"nonmonotonic", ErrorCode::NonMonotonicTime},
        {"zero-quat", ErrorCode::ZeroQuaternion},
        {"bad-weight-count", ErrorCode::InvalidWeightCount},
        {"bad-weights", ErrorCode::WeightSumZero},
        {"bad-order", ErrorCode::InvalidCompositionOrder},
        {"truncated", ErrorCode::Truncated},
    }};
    for (const ExpectedError& item : expected) {
        const std::filesystem::path path =
            harness.directory() / (std::string(item.preset) + ".ska");
        Error generationError;
        const bool generated = generateAssetFile(
            item.preset, CompositionOrder::TRS, path, generationError);
        Asset poisoned = asset;
        Error loadError;
        const bool loaded = AssetReader::load(path, poisoned, loadError);
        harness.check(generated && !loaded,
                      std::string(item.preset) + " asset is rejected");
        harness.expectError(loadError.code, item.code,
                            std::string(item.preset) + " error code");
        harness.check(poisoned.nodes.empty() && poisoned.meshes.empty() &&
                          poisoned.samples.empty(),
                      std::string(item.preset) +
                          " leaves no partially loaded asset");
    }

    std::vector<std::uint8_t> bytes = readBytes(normalPath);
    if (bytes.size() >= 4U) {
        bytes[0] = 'X';
    }
    Asset magicAsset;
    harness.check(!AssetReader::parse(bytes, magicAsset, error),
                  "reader rejects bad magic");
    harness.expectError(error.code, ErrorCode::InvalidMagic,
                        "bad magic error code");
}

void testTransformComposition(TestHarness& harness) {
    const glm::vec3 translation(1.0F, 2.0F, 3.0F);
    const glm::quat rotation =
        glm::angleAxis(glm::radians(73.0F), glm::normalize(
            glm::vec3(0.2F, 0.8F, 0.4F)));
    const glm::quat unnormalized = rotation * 5.0F;
    const glm::vec3 scale(2.0F, 0.5F, 1.25F);
    for (std::uint32_t orderValue = 0; orderValue <= 5U; ++orderValue) {
        const auto order = static_cast<CompositionOrder>(orderValue);
        glm::mat4 actual;
        Error error;
        const bool success = composeLocalTransform(
            order, translation, unnormalized, scale, actual, error);
        const glm::mat4 expected = referenceComposition(
            order, translation, rotation, scale);
        harness.check(success && matrixNear(actual, expected),
                      std::string("composition order ") +
                          compositionOrderName(order) +
                          " matches reference within 1e-5");
    }

    glm::mat4 unchanged(7.0F);
    const glm::mat4 previous = unchanged;
    Error zeroError;
    const bool rejected = !composeLocalTransform(
        CompositionOrder::TRS, translation,
        glm::quat(0.0F, 0.0F, 0.0F, 0.0F), scale, unchanged, zeroError);
    harness.check(rejected && unchanged == previous,
                  "zero quaternion is rejected without output mutation");
    harness.expectError(zeroError.code, ErrorCode::ZeroQuaternion,
                        "zero quaternion error code");

    // A two-node hierarchy validates parent * local ordering through a baked
    // vertex position.
    Asset hierarchy;
    hierarchy.frameCount = 1;
    hierarchy.framesPerSecond = 24.0F;
    Node root;
    root.name = "root";
    root.firstSample = 0;
    root.sampleCount = 1;
    Node child;
    child.name = "child";
    child.parent = 0;
    child.firstSample = 1;
    child.sampleCount = 1;
    hierarchy.nodes = {root, child};
    hierarchy.samples = {
        AnimationSample{0.0F, glm::vec3(1.0F, 0.0F, 0.0F),
                        glm::quat(1.0F, 0.0F, 0.0F, 0.0F),
                        glm::vec3(1.0F)},
        AnimationSample{0.0F, glm::vec3(0.0F, 2.0F, 0.0F),
                        glm::quat(1.0F, 0.0F, 0.0F, 0.0F),
                        glm::vec3(1.0F)},
    };
    Mesh mesh;
    Vertex vertex;
    vertex.position = glm::vec3(0.0F);
    vertex.normal = glm::vec3(0.0F, 0.0F, 1.0F);
    vertex.color = glm::vec3(1.0F);
    vertex.weightCount = 1;
    vertex.boneIndices = {1, 0, 0, 0};
    vertex.weights = {1.0F, 0.0F, 0.0F, 0.0F};
    mesh.vertices.push_back(vertex);
    hierarchy.meshes.push_back(mesh);
    BakedFrame baked;
    Error hierarchyError;
    const bool evaluated = evaluateFrame(hierarchy, 0, baked,
                                          hierarchyError);
    harness.check(evaluated && near(baked.positions[0].x, 1.0F) &&
                      near(baked.positions[0].y, 2.0F),
                  "world hierarchy uses parent * local");

    // The asset composition order drives the same point through evaluation.
    for (std::uint32_t orderValue = 0; orderValue <= 5U; ++orderValue) {
        const auto order = static_cast<CompositionOrder>(orderValue);
        Asset single = makeSingleBoneAsset(
            translation, unnormalized, scale, glm::mat4(1.0F),
            glm::vec3(1.0F, -2.0F, 0.5F), glm::vec3(0.0F, 1.0F, 0.0F));
        single.compositionOrder = order;
        BakedFrame frame;
        Error evaluationError;
        const bool ok = evaluateFrame(single, 0, frame, evaluationError);
        const glm::vec4 expected = referenceComposition(
            order, translation, rotation, scale) *
            glm::vec4(1.0F, -2.0F, 0.5F, 1.0F);
        harness.check(ok && near(frame.positions[0].x, expected.x) &&
                          near(frame.positions[0].y, expected.y) &&
                          near(frame.positions[0].z, expected.z),
                      std::string("asset order ") +
                          compositionOrderName(order) +
                          " is honored by evaluation");
    }
}

void testSkinning(TestHarness& harness) {
    const glm::vec3 translation(1.0F, 2.0F, 3.0F);
    const glm::quat identity(1.0F, 0.0F, 0.0F, 0.0F);
    const glm::vec3 position(0.25F, -1.0F, 2.0F);
    const glm::mat4 inverseWorld =
        glm::inverse(glm::translate(glm::mat4(1.0F), translation));
    Asset inverseBindAsset = makeSingleBoneAsset(
        translation, identity, glm::vec3(1.0F), inverseWorld, position,
        glm::vec3(0.0F, 0.0F, 1.0F));
    BakedFrame inverseBindFrame;
    Error error;
    harness.check(evaluateFrame(inverseBindAsset, 0, inverseBindFrame, error),
                  "inverse-bind skinning frame evaluates");
    harness.check(
        near(inverseBindFrame.positions[0].x, position.x) &&
            near(inverseBindFrame.positions[0].y, position.y) &&
            near(inverseBindFrame.positions[0].z, position.z),
        "world * inverse-bind returns bind position");
    harness.check(matrixNear(inverseBindFrame.skinMatrices[0],
                             glm::mat4(1.0F)),
                  "skin matrix is identity at bind pose");

    const glm::vec3 nonUniformScale(2.0F, 1.0F, 0.5F);
    const glm::vec3 sourceNormal =
        glm::normalize(glm::vec3(1.0F, 1.0F, 0.0F));
    Asset normalAsset = makeSingleBoneAsset(
        glm::vec3(0.0F), identity, nonUniformScale, glm::mat4(1.0F),
        glm::vec3(1.0F, 0.0F, 0.0F), sourceNormal);
    BakedFrame normalFrame;
    harness.check(evaluateFrame(normalAsset, 0, normalFrame, error),
                  "non-uniform skinning frame evaluates");
    harness.check(near(normalFrame.positions[0].x, 2.0F),
                  "position uses skin matrix");
    const glm::vec3 expectedNormal =
        glm::normalize(glm::vec3(0.5F, 1.0F, 0.0F));
    harness.check(near(normalFrame.normals[0].x, expectedNormal.x) &&
                      near(normalFrame.normals[0].y, expectedNormal.y) &&
                      near(normalFrame.normals[0].z, expectedNormal.z),
                  "normal uses inverse-transpose under non-uniform scale");

    Asset normalRig;
    harness.check(
        AssetReader::load(harness.directory() / "normal.ska", normalRig,
                          error),
        "normal rig reloads for weight test");
    BakedFrame weightedFrame;
    harness.check(evaluateFrame(normalRig, 0, weightedFrame, error),
                  "normal rig frame with weight normalization evaluates");
    harness.check(weightedFrame.stats.normalizedVertices == 1U,
                  "one non-unit vertex weight sum is counted");
    harness.check(weightedFrame.indices.size() == 72U,
                  "weight normalization preserves binding topology");

    Asset singularRig;
    harness.check(
        generateAssetFile("singular", CompositionOrder::TRS,
                          harness.directory() / "singular.ska", error),
        "singular rig generates for NaN accounting test");
    harness.check(
        AssetReader::load(harness.directory() / "singular.ska", singularRig,
                          error),
        "singular rig loads for NaN accounting test");
    BakedFrame singularFrame;
    harness.check(evaluateFrame(singularRig, 0, singularFrame, error),
                  "singular rig evaluates without throwing");
    harness.check(singularFrame.stats.nanNormals > 0U,
                  "degenerate skin matrix produces counted normal NaNs");
    harness.check(singularFrame.stats.nanPositions == 0U,
                  "position and normal NaN counts are separate");
}

void testOfflineDeterminismAndSharedPath(TestHarness& harness) {
    Asset asset;
    Error error;
    harness.check(
        AssetReader::load(harness.directory() / "normal.ska", asset, error),
        "asset loads for deterministic render test");

    RenderSettings settings;
    settings.width = 96;
    settings.height = 72;

    BakedFrame realtimeBake;
    BakedFrame offlineBake;
    harness.check(evaluateFrame(asset, 5, realtimeBake, error),
                  "realtime/shared bake succeeds");
    harness.check(evaluateFrame(asset, 5, offlineBake, error),
                  "offline bake succeeds");
    harness.check(realtimeBake.positions.size() ==
                      offlineBake.positions.size() &&
                      realtimeBake.indices == offlineBake.indices,
                  "realtime and offline share baked topology");
    const std::vector<InterleavedVertex> realtimeVertices =
        makeInterleavedVertices(realtimeBake);
    bool vertexConsistency = realtimeVertices.size() ==
                             realtimeBake.positions.size();
    for (std::size_t index = 0;
         vertexConsistency && index < realtimeVertices.size(); ++index) {
        vertexConsistency =
            realtimeVertices[index].position[0] ==
                realtimeBake.positions[index].x &&
            realtimeVertices[index].normal[1] ==
                realtimeBake.normals[index].y &&
            realtimeVertices[index].color[2] ==
                realtimeBake.colors[index].z;
    }
    harness.check(vertexConsistency,
                  "OpenGL interleave consumes the offline baked arrays");

    OfflineImage firstImage;
    OfflineImage secondImage;
    harness.check(rasterizeOffline(realtimeBake, settings, firstImage, error),
                  "first deterministic rasterization succeeds");
    harness.check(rasterizeOffline(offlineBake, settings, secondImage, error),
                  "second deterministic rasterization succeeds");
    harness.check(firstImage.depth == secondImage.depth &&
                      firstImage.rgb == secondImage.rgb,
                  "shared bake yields bit-identical image buffers");

    const std::filesystem::path firstDirectory =
        harness.directory() / "deterministic_a";
    const std::filesystem::path secondDirectory =
        harness.directory() / "deterministic_b";
    harness.check(writeOfflineImages(firstImage, settings,
                                     firstDirectory / "depth_0005.pfm",
                                     firstDirectory / "color_0005.ppm", error),
                  "first image files write");
    harness.check(writeOfflineImages(secondImage, settings,
                                     secondDirectory / "depth_0005.pfm",
                                     secondDirectory / "color_0005.ppm", error),
                  "second image files write");
    harness.check(readBytes(firstDirectory / "depth_0005.pfm") ==
                      readBytes(secondDirectory / "depth_0005.pfm"),
                  "PFM output is byte-identical across runs");
    harness.check(readBytes(firstDirectory / "color_0005.ppm") ==
                      readBytes(secondDirectory / "color_0005.ppm"),
                  "PPM output is byte-identical across runs");
    int decodedWidth = 0;
    int decodedHeight = 0;
    int decodedChannels = 0;
    stbi_uc* decoded = stbi_load(
        (firstDirectory / "color_0005.ppm").string().c_str(),
        &decodedWidth, &decodedHeight, &decodedChannels, 3);
    harness.check(decoded != nullptr && decodedWidth == 96 &&
                      decodedHeight == 72 && decodedChannels == 3,
                  "stb_image decodes the emitted PPM");
    if (decoded != nullptr) {
        stbi_image_free(decoded);
    }

    RenderReport reportA;
    reportA.assetFile = "normal.ska";
    reportA.assetFrameCount = asset.frameCount;
    reportA.framesPerSecond = asset.framesPerSecond;
    reportA.compositionOrder = asset.compositionOrder;
    reportA.width = settings.width;
    reportA.height = settings.height;
    reportA.camera = settings.camera;
    reportA.vertexCount = asset.vertexCount();
    reportA.triangleCount = asset.triangleCount();
    reportA.normalizedVertexCount =
        realtimeBake.stats.normalizedVertices;
    FrameRecord record;
    record.frame = 5;
    record.stats = realtimeBake.stats;
    record.depthFile = "depth_0005.pfm";
    record.colorFile = "color_0005.ppm";
    record.elapsedMs = 1.25;
    reportA.frames.push_back(record);
    reportA.elapsedMs = 4.5;
    RenderReport reportB = reportA;
    reportB.frames[0].elapsedMs = 999.0;
    reportB.elapsedMs = 1234.0;
    harness.check(
        writeStatisticsJson(firstDirectory / "report.json", reportA, error),
        "first statistics report writes and parses");
    harness.check(
        writeStatisticsJson(secondDirectory / "report.json", reportB, error),
        "second statistics report writes and parses");

    auto deterministicJson = [](const std::filesystem::path& path) {
        auto document = nlohmann::json::parse(readText(path));
        document.erase("timing");
        for (auto& frame : document["frames"]) {
            frame.erase("elapsed_ms");
        }
        return document.dump();
    };
    harness.check(deterministicJson(firstDirectory / "report.json") ==
                      deterministicJson(secondDirectory / "report.json"),
                  "JSON deterministic fields ignore timing values");

    RenderReport parsedReport;
    harness.check(parseStatisticsJson(readText(firstDirectory / "report.json"),
                                      parsedReport, error),
                  "statistics report parses back");
    harness.check(!parseStatisticsJson("{not-json", parsedReport, error),
                  "malformed statistics JSON is rejected");
    harness.expectError(error.code, ErrorCode::InvalidJson,
                        "malformed JSON maps to asset error code");
}

} // namespace

int runSelfTests(const std::filesystem::path& workDirectory) {
    TestHarness harness(workDirectory);
    std::cout << "SkelPreview self-tests\n"
              << "work directory: " << workDirectory.string() << "\n\n";
    testReader(harness);
    testTransformComposition(harness);
    testSkinning(harness);
    testOfflineDeterminismAndSharedPath(harness);
    return harness.summary();
}

} // namespace skel
