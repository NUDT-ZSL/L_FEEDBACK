#include "assets.h"
#include "image_io.h"
#include "options.h"
#include "renderer.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string& message) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::cerr << "FAIL: " << message << "\n";
    } else {
        std::cout << "ok: " << message << "\n";
    }
}

cv::RendererConfig testConfig(int width = 320, int height = 240, bool shadows = true) {
    cv::RendererConfig config;
    config.width = width;
    config.height = height;
    config.cascadeResolution = 256;
    config.pointShadowResolution = 96;
    config.shadows = shadows;
    return config;
}

bool matricesEqual(const cv::Mat4& a, const cv::Mat4& b) {
    return std::memcmp(a.m.data(), b.m.data(), sizeof(float) * 16) == 0;
}

bool matricesNear(const cv::Mat4& a, const cv::Mat4& b, float epsilon = 1.0e-6f) {
    for (size_t i = 0; i < a.m.size(); ++i) {
        if (std::abs(a.m[i] - b.m[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    std::cout << std::setprecision(8);

    cv::AssetBundle assets = cv::makeProceduralScene();
    const cv::Scene& scene = assets.scene;
    check(scene.materials.size() >= 8, "procedural material palette exists");
    check(scene.meshes.size() >= 3, "character, floor and environment meshes exist");
    check(scene.skeleton.bones.size() >= 15, "skeleton has head/arms/legs hierarchy");
    check(scene.normalMap.width == 256 && scene.normalMap.height == 256,
          "procedural tangent-space normal map is 256x256");
    check(scene.totalTriangles() > 5000, "generated scene contains a substantial mesh");

    bool topologyValid = true;
    for (const cv::Mesh& mesh : scene.meshes) {
        if (mesh.indices.size() % 3 != 0 ||
            mesh.triangleMaterial.size() != mesh.indices.size() / 3) {
            topologyValid = false;
        }
        for (uint32_t index : mesh.indices) {
            if (index >= mesh.vertices.size()) {
                topologyValid = false;
            }
        }
    }
    check(topologyValid, "mesh indices and per-triangle materials are in range");

    bool weightsValid = true;
    bool sawMultipleInfluences = false;
    bool sawMirroredTangent = false;
    for (const cv::Mesh& mesh : scene.meshes) {
        for (const cv::Vertex& vertex : mesh.vertices) {
            float sum = 0.0f;
            int nonzero = 0;
            for (int i = 0; i < cv::kMaxBoneInfluences; ++i) {
                const float weight = vertex.boneWeights[static_cast<size_t>(i)];
                if (weight < 0.0f) {
                    weightsValid = false;
                }
                sum += weight;
                if (weight > 0.0f) {
                    ++nonzero;
                    if (vertex.boneIndices[static_cast<size_t>(i)] >=
                        scene.skeleton.bones.size()) {
                        weightsValid = false;
                    }
                }
            }
            if (!mesh.skinned) {
                continue;
            }
            if (std::abs(sum - 1.0f) > 1.0e-5f) {
                weightsValid = false;
            }
            if (nonzero > 1) {
                sawMultipleInfluences = true;
            }
            if (vertex.tangent.w < 0.0f) {
                sawMirroredTangent = true;
            }
        }
    }
    check(weightsValid, "all skinned weights are normalized and bone indices are valid");
    check(sawMultipleInfluences, "skin vertices exercise linear blending of influences");
    check(sawMirroredTangent, "mirrored UV vertices carry tangent.w=-1");

    bool normalMapVaries = false;
    for (size_t i = 3; i < scene.normalMap.rgb.size(); i += 97) {
        if (scene.normalMap.rgb[i] != scene.normalMap.rgb[0]) {
            normalMapVaries = true;
            break;
        }
    }
    check(normalMapVaries, "normal map is non-constant procedural data");

    const std::vector<cv::Mat4> bones5 =
        cv::buildBoneMatrices(scene.skeleton, 5);
    const std::vector<cv::Mat4> bones5Again =
        cv::buildBoneMatrices(scene.skeleton, 5);
    const std::vector<cv::Mat4> bones6 =
        cv::buildBoneMatrices(scene.skeleton, 6);
    bool boneRepeat = true;
    bool animationMoves = false;
    for (size_t i = 0; i < bones5.size(); ++i) {
        boneRepeat = boneRepeat && matricesEqual(bones5[i], bones5Again[i]);
        animationMoves = animationMoves || !matricesNear(bones5[i], bones6[i]);
    }
    check(boneRepeat, "fixed-tick bone matrices are bitwise repeatable");
    check(animationMoves, "fixed-tick animation changes the skeleton pose");

    const cv::Mesh& character = scene.meshes[0];
    int capeMaterial = -1;
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        if (scene.materials[i].name == "Cape") {
            capeMaterial = static_cast<int>(i);
        }
    }
    check(capeMaterial >= 0 && scene.materials[static_cast<size_t>(capeMaterial)].transparent,
          "cape material exists on the transparent path");
    const std::vector<cv::FrameVertex> skinA =
        cv::skinVertices(character, bones5);
    const std::vector<cv::FrameVertex> skinB =
        cv::skinVertices(character, bones5Again);
    bool skinRepeat = skinA.size() == skinB.size();
    if (skinRepeat) {
        for (size_t i = 0; i < skinA.size(); ++i) {
            skinRepeat = skinRepeat &&
                         std::memcmp(&skinA[i].worldPosition, &skinB[i].worldPosition,
                                     sizeof(cv::Vec3)) == 0;
        }
    }
    check(skinRepeat, "CPU skinning is repeatable for the same tick");

    cv::AABB capeBounds;
    for (size_t triangle = 0; triangle < character.indices.size() / 3; ++triangle) {
        if (character.triangleMaterial[triangle] !=
            static_cast<uint16_t>(capeMaterial)) {
            continue;
        }
        for (int corner = 0; corner < 3; ++corner) {
            const uint32_t index =
                character.indices[triangle * 3 + static_cast<size_t>(corner)];
            capeBounds.expand(skinA[index].worldPosition);
        }
    }
    check(capeBounds.valid() && capeBounds.min.y < 0.8f && capeBounds.min.z < -0.10f,
          "skinned cape retains its hanging back-side geometry");

    cv::Camera camera;
    cv::Renderer rendererA(scene, assets.lighting, testConfig());
    cv::Renderer rendererB(scene, assets.lighting, testConfig());
    rendererA.render(scene, assets.lighting, camera, 0, false);
    rendererB.render(scene, assets.lighting, camera, 0, false);
    const uint64_t hashA = cv::hashImage(rendererA.image());
    const uint64_t hashB = cv::hashImage(rendererB.image());
    check(rendererA.image().valid(), "renderer produces a valid RGB framebuffer");
    check(hashA == hashB, "two independent runs produce byte-identical pixels");
    check(rendererA.stats().gridVisibleTriangles < rendererA.stats().totalTriangles,
          "CPU frustum culling removes triangles");
    check(rendererA.stats().cullRatio > 0.0f && rendererA.stats().cullRatio < 1.0f,
          "culling ratio is reported between zero and one");
    check(rendererA.stats().visibleCells < rendererA.stats().totalCells,
          "visible grid-cell set is a subset of all cells");
    check(rendererA.stats().transparentTriangles > 0,
          "transparent material triangles are submitted");
    check(rendererA.stats().transparentOrderValid,
          "transparent triangles are back-to-front with stable tie breakers");
    check(rendererA.stats().opaqueCommands < rendererA.stats().opaqueTriangles,
          "opaque state grouping batches multiple triangles per command");
    check(rendererA.stats().shadowMaps == 14,
          "14 shadow maps: 2 cascades plus 2 lights x 6 cube faces");
    check(rendererA.stats().shadowTriangles > 0, "shadow pass rasterizes casters");

    rendererA.render(scene, assets.lighting, camera, 1, false);
    const uint64_t hashTick1 = cv::hashImage(rendererA.image());
    check(hashTick1 != hashA, "animation tick changes rendered pixels");

    cv::Renderer noShadows(scene, assets.lighting, testConfig(320, 240, false));
    noShadows.render(scene, assets.lighting, camera, 1, false);
    const uint64_t hashNoShadows = cv::hashImage(noShadows.image());
    check(noShadows.stats().shadowMaps == 0, "shadow disable path creates no shadow maps");
    check(hashNoShadows != hashTick1, "shadow maps visibly affect final shading");

    rendererA.render(scene, assets.lighting, camera, 1, true);
    const uint64_t hashWire = cv::hashImage(rendererA.image());
    check(hashWire != hashTick1, "wireframe mode changes rendered pixels");

    rendererA.render(scene, assets.lighting, camera, 0, false);
    check(cv::hashImage(rendererA.image()) == hashA,
          "returning to the same tick reproduces the original frame");

    cv::Camera culledCamera = camera;
    culledCamera.target = {100.0f, 1.13f, 0.0f};
    rendererA.render(scene, assets.lighting, culledCamera, 0, false);
    check(rendererA.stats().gridVisibleTriangles == 0 && rendererA.stats().cullRatio == 0.0f,
          "camera looking away culls every triangle");

    rendererA.render(scene, assets.lighting, camera, 0, false);
    rendererA.resize(257, 199);
    rendererA.render(scene, assets.lighting, camera, 0, false);
    check(rendererA.image().width == 257 && rendererA.image().height == 199,
          "resize rebuilds the framebuffer at the requested size");
    rendererA.resize(320, 240);
    rendererA.render(scene, assets.lighting, camera, 0, false);
    check(cv::hashImage(rendererA.image()) == hashA,
          "resize back to original size restores byte-identical output");

    const std::string ppmPath = "charview_test_frame.ppm";
    const std::string bmpPath = "charview_test_frame.bmp";
    const cv::ExportReport ppm = cv::exportPpmP6(ppmPath, rendererA.image());
    const cv::ExportReport bmp = cv::exportBmp24(bmpPath, rendererA.image());
    check(ppm.ok, "PPM(P6) export succeeds and validates its header");
    check(bmp.ok, "BMP(24-bit) export succeeds and validates its header");
    check(ppm.pixelHash == hashA && bmp.pixelHash == hashA,
          "export reports the exact framebuffer hash");
    check(cv::validatePpmHeader(ppmPath), "PPM header validates on disk");
    check(cv::validateBmpHeader(bmpPath), "BMP header validates on disk");

    cv::Image ppmDecoded;
    cv::Image bmpDecoded;
    std::string decodeError;
    const bool ppmLoaded = cv::loadPpmP6(ppmPath, ppmDecoded, &decodeError);
    const bool bmpLoaded = cv::loadBmp24(bmpPath, bmpDecoded, &decodeError);
    check(ppmLoaded && bmpLoaded, "both exported files decode successfully");
    check(ppmDecoded.rgb == rendererA.image().rgb,
          "decoded PPM pixels equal the window/framebuffer pixels");
    check(bmpDecoded.rgb == rendererA.image().rgb,
          "decoded BMP pixels equal the window/framebuffer pixels");

    cv::Image whitespaceFirst;
    whitespaceFirst.width = 2;
    whitespaceFirst.height = 1;
    whitespaceFirst.rgb = {32, 10, 10, 40, 50, 60};
    const cv::ExportReport whitespacePpm =
        cv::exportPpmP6("charview_whitespace_first.ppm", whitespaceFirst);
    cv::Image whitespaceDecoded;
    const bool whitespaceLoaded =
        cv::loadPpmP6("charview_whitespace_first.ppm", whitespaceDecoded);
    check(whitespacePpm.ok && whitespaceLoaded &&
              whitespaceDecoded.rgb == whitespaceFirst.rgb,
          "PPM parser does not consume a whitespace-valued first pixel byte");

    {
        std::ofstream badPpm("charview_bad_header.ppm", std::ios::binary);
        badPpm << "P5\n1 1\n255\n";
        const char byte = 0;
        badPpm.write(&byte, 1);
    }
    {
        std::ofstream badBmp("charview_bad_header.bmp", std::ios::binary);
        badBmp << "NOT-A-BMP";
    }
    check(!cv::validatePpmHeader("charview_bad_header.ppm"),
          "invalid PPM magic is rejected");
    check(!cv::validateBmpHeader("charview_bad_header.bmp"),
          "invalid BMP signature is rejected");

    const char* args[] = {"charview", "--width", "640", "--height=360", "--tick", "12",
                          "--ppm-key", "x", "--bmp-key=Y", "--no-shadows",
                          "--yaw", "180", "--pitch=-5", "--distance=7"};
    const cv::AppOptions options =
        cv::parseOptions(static_cast<int>(sizeof(args) / sizeof(args[0])),
                         const_cast<char**>(args));
    check(options.valid, "command-line options parse successfully");
    check(options.width == 640 && options.height == 360 && options.startTick == 12,
          "numeric command-line options are applied");
    check(options.ppmKey == 'X' && options.bmpKey == 'Y' && !options.shadows,
          "export hotkeys and shadow disable are configurable");
    check(options.hasYaw && options.hasPitch && options.hasCameraDistance &&
              std::abs(options.yawDegrees - 180.0f) < 1.0e-5f &&
              std::abs(options.pitchDegrees + 5.0f) < 1.0e-5f &&
              std::abs(options.cameraDistance - 7.0f) < 1.0e-5f,
          "reproducible camera overrides are parsed");

    std::cout << "\nchecks=" << g_checks << " failures=" << g_failures
              << " hash=0x" << std::hex << hashA << std::dec << "\n";
    return g_failures == 0 ? 0 : 1;
}

