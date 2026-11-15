#include "gl_backend.h"
#include "image.h"
#include "options.h"
#include "renderer.h"
#include "scene.h"
#include "stats.h"
#include "validation.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    using namespace pbr;

    Options options;
    std::string error;
    if (!parseOptions(argc, argv, options, error)) {
        std::cerr << "Argument error: " << error << "\n\n";
        printUsage(argv[0]);
        return static_cast<int>(ExitCode::InvalidArguments);
    }
    if (options.showHelp) {
        printUsage(argv[0]);
        return static_cast<int>(ExitCode::Ok);
    }

    Scene scene;
    if (!loadScene(options.scenePath, scene, error)) {
        std::cerr << error << "\n";
        return static_cast<int>(ExitCode::SceneLoadFailed);
    }

    OutputPaths outputPaths;
    if (!buildOutputPaths(options.outputPrefix, options.exposure, options.tonemap, outputPaths, error)) {
        std::cerr << "Output path error: " << error << "\n";
        return static_cast<int>(ExitCode::RenderOrIoFailed);
    }

    GLContext context;
    if (!context.initialize(kResolution, kResolution, error)) {
        std::cerr << "Render initialization error: " << error << "\n";
        return static_cast<int>(ExitCode::RenderOrIoFailed);
    }
    std::cout << "OpenGL: " << context.rendererSummary() << "\n";

    Renderer renderer;
    if (!renderer.initialize(error)) {
        std::cerr << "Render initialization error: " << error << "\n";
        return static_cast<int>(ExitCode::RenderOrIoFailed);
    }

    RenderSettings settings;
    settings.samples = options.samples;
    settings.subframes = options.subframes;

    Image image;
    std::cout << "Rendering 512x512, samples=" << settings.samples
              << ", subframes=" << settings.subframes << ", scene=" << options.scenePath << "\n";
    if (!renderer.render(scene, settings, image, error)) {
        std::cerr << "Render error: " << error << "\n";
        return static_cast<int>(ExitCode::RenderOrIoFailed);
    }

    ColorPipelineInfo colorInfo;
    colorInfo.exposureEv = options.exposure;
    colorInfo.tonemap = options.tonemap;
    colorInfo.samples = settings.samples;
    colorInfo.subframes = settings.subframes;
    colorInfo.seed = settings.seed;
    colorInfo.scenePath = options.scenePath;

    if (!writeHdr(outputPaths.hdr, image, colorInfo, error)) {
        std::cerr << error << "\n";
        return static_cast<int>(ExitCode::RenderOrIoFailed);
    }
    if (!writePng(outputPaths.png, image, colorInfo, error)) {
        std::cerr << error << "\n";
        return static_cast<int>(ExitCode::RenderOrIoFailed);
    }
    std::cout << "Wrote HDR: " << outputPaths.hdr << "\n";
    std::cout << "Wrote PNG: " << outputPaths.png << "\n";

    const RenderStatistics statistics = computeStatistics(scene, image);
    if (!options.writeReferencePath.empty()) {
        const EnvironmentSH environmentSh = computeEnvironmentSH(scene.environment);
        if (!writeReferenceStats(options.writeReferencePath, options.scenePath, colorInfo, statistics,
                                 environmentSh, error)) {
            std::cerr << error << "\n";
            return static_cast<int>(ExitCode::RenderOrIoFailed);
        }
        std::cout << "Wrote reference statistics: " << options.writeReferencePath << "\n";
    }

    if (options.mode == Mode::Validate) {
        ReferenceStats reference;
        if (!loadReferenceStats(options.referencePath, reference, error)) {
            std::cerr << error << "\n";
            return static_cast<int>(ExitCode::RenderOrIoFailed);
        }
        const ValidationResult validation = validateStatistics(statistics, reference, colorInfo);
        if (!writeValidationReport(options.reportPath, validation, statistics, reference, colorInfo, error)) {
            std::cerr << error << "\n";
            return static_cast<int>(ExitCode::RenderOrIoFailed);
        }
        std::cout << "Validation: " << (validation.passes ? "PASS" : "FAIL")
                  << " (" << options.reportPath << ")\n";
        if (!validation.configurationMatches) {
            std::cout << "Validation note: " << validation.configurationMessage << "\n";
        }
    }

    return static_cast<int>(ExitCode::Ok);
}
