#include "assets.h"
#include "image_io.h"
#include "options.h"
#include "renderer.h"

#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    const cv::AppOptions options = cv::parseOptions(argc, argv);
    if (options.helpRequested) {
        cv::printUsage(argv[0]);
        return 0;
    }
    if (!options.valid) {
        std::cerr << "error: " << options.error << "\n";
        cv::printUsage(argv[0]);
        return 2;
    }

    cv::AssetBundle assets = cv::makeProceduralScene();
    cv::RendererConfig config;
    config.width = options.width;
    config.height = options.height;
    config.cascadeResolution = options.cascadeResolution;
    config.pointShadowResolution = options.pointShadowResolution;
    config.shadows = options.shadows;
    cv::Renderer renderer(assets.scene, assets.lighting, config);

    cv::Camera camera;
    if (options.hasYaw) {
        camera.yawRadians = options.yawDegrees * cv::kPi / 180.0f;
    }
    if (options.hasPitch) {
        camera.pitchRadians = options.pitchDegrees * cv::kPi / 180.0f;
    }
    if (options.hasCameraDistance) {
        camera.distance = options.cameraDistance;
    }
    renderer.render(assets.scene, assets.lighting, camera, options.startTick,
                    options.wireframe);
    const cv::FrameStats& stats = renderer.stats();
    std::cout << "[cull] total=" << stats.totalTriangles
              << " grid_candidates=" << stats.gridCandidateTriangles
              << " visible=" << stats.gridVisibleTriangles
              << " ratio=" << std::fixed << std::setprecision(4) << stats.cullRatio
              << " cells=" << stats.visibleCells << "/" << stats.totalCells << "\n";
    std::cout << "[draw] opaque=" << stats.opaqueTriangles
              << " transparent=" << stats.transparentTriangles
              << " commands=" << (stats.opaqueCommands + stats.transparentCommands)
              << "+wire:" << stats.wireCommands
              << " transparent_order=" << (stats.transparentOrderValid ? "ok" : "bad")
              << " shadow_maps=" << stats.shadowMaps
              << " shadow_triangles=" << stats.shadowTriangles << "\n";
    std::cout << "[timing] cull_ms=" << std::setprecision(2) << stats.cullMs
              << " shadow_ms=" << stats.shadowMs
              << " shade_ms=" << stats.colorMs
              << " total_ms=" << stats.totalMs << "\n";

    const cv::ExportReport ppm = cv::exportPpmP6(options.ppmPath, renderer.image());
    if (!ppm.ok) {
        std::cerr << "PPM export failed: " << ppm.error << "\n";
        return 1;
    }
    const cv::ExportReport bmp = cv::exportBmp24(options.bmpPath, renderer.image());
    if (!bmp.ok) {
        std::cerr << "BMP export failed: " << bmp.error << "\n";
        return 1;
    }
    std::cout << "[export] ppm=" << options.ppmPath << " hash=0x" << std::hex
              << ppm.pixelHash << std::dec << "\n";
    std::cout << "[export] bmp=" << options.bmpPath << " hash=0x" << std::hex
              << bmp.pixelHash << std::dec << "\n";
    std::cout << "[image] " << renderer.width() << "x" << renderer.height()
              << " framebuffer_hash=0x" << std::hex << cv::hashImage(renderer.image())
              << std::dec << "\n";
    return 0;
}

