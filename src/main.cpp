#include "Options.h"
#include "Renderer.h"
#include "Scene.h"
#include "Stats.h"
#include "WicPng.h"

#include <iostream>

namespace {

class ComInitializer {
public:
    ComInitializer() {
        const HRESULT result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(result)) {
            throw std::runtime_error("CoInitializeEx failed: " + HResultText(result));
        }
    }

    ~ComInitializer() { CoUninitialize(); }

    ComInitializer(const ComInitializer&) = delete;
    ComInitializer& operator=(const ComInitializer&) = delete;
};

void RemovePreviousFrameOutputs(const std::filesystem::path& directory) {
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::wstring name = entry.path().filename().wstring();
        if (name.rfind(L"frame_", 0) != 0) {
            continue;
        }
        const std::wstring extension = entry.path().extension().wstring();
        if (extension == L".png" || extension == L".json") {
            std::error_code ignored;
            std::filesystem::remove(entry.path(), ignored);
        }
    }
}

FrameStats CaptureFrameStats(const Renderer& renderer, uint32_t frame,
                             double animationTime) {
    FrameStats stats;
    stats.frame = frame;
    stats.animationTimeSeconds = animationTime;
    stats.drawCalls = Renderer::kDrawCallsPerFrame;
    stats.shadowMapSizes.fill(Renderer::kShadowMapSize);
    stats.cascadeSplitRatios = renderer.CascadeSplitRatios();
    stats.cascadeSplitDistances = renderer.SplitDistances();
    stats.cascades = renderer.Cascades();
    stats.slopeOffset = renderer.SlopeOffset();
    stats.constOffset = renderer.ConstOffset();
    stats.deviceLossRecoveries = renderer.DeviceLossRecoveryCount();
    return stats;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    ParseResult parsed;
    try {
        parsed = ParseCommandLine(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        std::cerr << "Run deterministic_csm --help for usage.\n";
        return 2;
    }

    if (parsed.showHelp) {
        PrintUsage(std::wcout);
        return 0;
    }

    const Options& options = parsed.options;

    try {
        ComInitializer com;
        std::filesystem::create_directories(options.outputDirectory);
        RemovePreviousFrameOutputs(options.outputDirectory);

        const SceneData scene = GenerateDeterministicScene();
        Renderer renderer(scene, options);
        renderer.Initialize();
        WicPngWriter pngWriter;

        AggregateStats aggregate;
        aggregate.frameCount = options.frames;
        aggregate.width = Renderer::kWidth;
        aggregate.height = Renderer::kHeight;
        aggregate.vertexCount = static_cast<uint32_t>(scene.vertices.size());
        aggregate.indexCount = static_cast<uint32_t>(scene.indices.size());
        aggregate.cascadeSplitRatios = options.cascadeSplits;
        aggregate.cascadeSplitDistances = renderer.SplitDistances();
        aggregate.hasDeviceLossInjection = options.hasDeviceLossInjection;
        aggregate.deviceLossInjectionFrame = options.deviceLossInjectionFrame;
        aggregate.slopeOffset = options.slopeOffset;
        aggregate.constOffset = options.constOffset;
        aggregate.frames.reserve(options.frames);

        for (uint32_t frame = 0; frame < options.frames; ++frame) {
            const double animationTime =
                static_cast<double>(frame) / 60.0;
            bool completed = false;

            for (uint32_t attempt = 0; attempt < 3; ++attempt) {
                try {
                    renderer.RenderFrame(animationTime);
                    const std::vector<uint8_t> pixels = renderer.ReadColorBuffer();
                    const std::string baseName =
                        FrameBaseName(frame, options.frames);
                    const std::filesystem::path pngPath =
                        options.outputDirectory / (baseName + ".png");
                    pngWriter.Write(pngPath, Renderer::kWidth, Renderer::kHeight,
                                    pixels);

                    const FrameStats frameStats =
                        CaptureFrameStats(renderer, frame, animationTime);
                    const std::filesystem::path jsonPath =
                        options.outputDirectory / (baseName + ".json");
                    WriteTextFile(jsonPath, BuildFrameJson(frameStats));
                    aggregate.frames.push_back(frameStats);
                    completed = true;
                    break;
                } catch (const DeviceLostError&) {
                    if (attempt + 1 == 3) {
                        throw;
                    }
                    renderer.RecoverFromDeviceLoss();
                }
            }

            if (!completed) {
                throw std::runtime_error("frame could not be completed");
            }

            if (options.hasDeviceLossInjection &&
                frame == options.deviceLossInjectionFrame) {
                renderer.SimulateDeviceLossAndRecover();
            }
        }

        aggregate.deviceLossRecoveries = renderer.DeviceLossRecoveryCount();
        aggregate.cascadeSplitDistances = renderer.SplitDistances();
        const std::filesystem::path aggregatePath =
            options.outputDirectory / "stats.json";
        WriteTextFile(aggregatePath, BuildAggregateJson(aggregate));

        std::wcout << L"Rendered " << options.frames << L" frames to "
                   << options.outputDirectory.wstring() << L"\n";
        std::wcout << L"Device loss recoveries: "
                   << renderer.DeviceLossRecoveryCount() << L"\n";
        return 0;
    } catch (const DeviceLostError& error) {
        std::cerr << "error: unrecoverable Direct3D device loss: "
                  << HResultText(error.hr) << "\n";
        return 3;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}

