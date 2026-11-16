#include "core/benchmark.hpp"
#include "core/error.hpp"
#include "win/d3d11_renderer.hpp"

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

namespace {

struct CliOptions {
    bool benchmark = false;
    bool generateOnly = false;
    bool help = false;
    uid166::BenchmarkOptions benchmarkOptions{};
    uint32_t windowWidth = 1600;
    uint32_t windowHeight = 900;
    uint32_t liveFrames = 0;
};

uint64_t parseUnsigned(const char* text, const char* option) {
    try {
        size_t consumed = 0;
        const std::string value(text);
        const unsigned long long parsed = std::stoull(value, &consumed, 10);
        if (consumed != value.size()) throw std::invalid_argument("trailing");
        return static_cast<uint64_t>(parsed);
    } catch (const std::exception&) {
        throw uid166::AppError(uid166::ErrorCode::usage,
                               std::string("invalid value for ") + option + ": " + text);
    }
}

CliOptions parseCommandLine(int argc, char** argv) {
    CliOptions options;
    options.benchmarkOptions.data.dataDir = "artifacts/generated";
    options.benchmarkOptions.outputPath = "artifacts/uid166_benchmark.json";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw uid166::AppError(uid166::ErrorCode::usage,
                                       std::string("missing value after ") + name);
            }
            return argv[++i];
        };
        if (arg == "--benchmark" || arg == "--headless") {
            options.benchmark = true;
        } else if (arg == "--generate-only") {
            options.generateOnly = true;
        } else if (arg == "--reuse-data") {
            options.benchmarkOptions.data.reuseExisting = true;
        } else if (arg == "--no-gpu-load") {
            options.benchmarkOptions.simulateGpuLoad = false;
        } else if (arg == "--frames") {
            options.benchmarkOptions.frames =
                static_cast<uint32_t>(parseUnsigned(next("--frames"), "--frames"));
        } else if (arg == "--objects") {
            options.benchmarkOptions.data.objectCount =
                static_cast<uint32_t>(parseUnsigned(next("--objects"), "--objects"));
        } else if (arg == "--seed") {
            options.benchmarkOptions.data.seed = parseUnsigned(next("--seed"), "--seed");
        } else if (arg == "--data-dir") {
            options.benchmarkOptions.data.dataDir = next("--data-dir");
        } else if (arg == "--output") {
            options.benchmarkOptions.outputPath = next("--output");
        } else if (arg == "--record-budget-us") {
            options.benchmarkOptions.recording.budgetMicros = static_cast<uint32_t>(
                parseUnsigned(next("--record-budget-us"), "--record-budget-us"));
        } else if (arg == "--workers") {
            options.benchmarkOptions.recording.workerCount = static_cast<uint32_t>(
                parseUnsigned(next("--workers"), "--workers"));
        } else if (arg == "--width") {
            options.windowWidth = static_cast<uint32_t>(
                parseUnsigned(next("--width"), "--width"));
        } else if (arg == "--height") {
            options.windowHeight = static_cast<uint32_t>(
                parseUnsigned(next("--height"), "--height"));
        } else if (arg == "--live-frames") {
            options.liveFrames = static_cast<uint32_t>(
                parseUnsigned(next("--live-frames"), "--live-frames"));
        } else if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else {
            throw uid166::AppError(uid166::ErrorCode::usage, "unknown option: " + arg);
        }
    }
    if (options.windowWidth < 320 || options.windowHeight < 240) {
        throw uid166::AppError(uid166::ErrorCode::usage, "window must be at least 320x240");
    }
    return options;
}

void printUsage() {
    std::cout
        << "UID166 native DirectX 11 scene\n"
        << "  uid166 [--benchmark] [--frames N] [--objects N] [--seed N]\n"
        << "         [--data-dir DIR] [--output FILE] [--reuse-data]\n"
        << "         [--record-budget-us N] [--workers N] [--no-gpu-load]\n"
        << "  uid166 --generate-only [--data-dir DIR]\n"
        << "  uid166 --width 1600 --height 900 [--live-frames N]\n";
}

LRESULT CALLBACK windowProcedure(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_CLOSE:
        DestroyWindow(window);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_ERASEBKGND:
        return 1;
    default:
        return DefWindowProcW(window, message, wParam, lParam);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions cli = parseCommandLine(argc, argv);
        if (cli.help) {
            printUsage();
            return 0;
        }
        if (cli.generateOnly) {
            const auto data = uid166::prepareSceneData(cli.benchmarkOptions.data);
            std::cout << "generated scene=" << data.paths.scene.string()
                      << " animation=" << data.paths.animation.string()
                      << " sceneHash=" << data.sceneHash
                      << " animationHash=" << data.animationHash << '\n';
            return 0;
        }
        if (cli.benchmark) {
            auto data = uid166::prepareSceneData(cli.benchmarkOptions.data);
            const auto summary = uid166::runBenchmark(cli.benchmarkOptions, std::move(data));
            std::cout << "benchmark output=" << cli.benchmarkOptions.outputPath.string()
                      << " frameStatsHash=" << summary.frameStatsHash
                      << " validationPassed=" << (summary.validationPassed ? "true" : "false")
                      << " stableRun=" << summary.longestStableMatrixRun
                      << " defaultQualityRestored="
                      << (summary.defaultQualityRestored ? "true" : "false") << '\n';
            return summary.validationPassed ? 0 : 5000;
        }
        // Live mode needs the original scene descriptor for the animated camera; the
        // renderer receives copies, so retain one before moving into it.
        uid166::PreparedData data = uid166::prepareSceneData(cli.benchmarkOptions.data);
        const uid166::CameraConfig cameraConfig = data.scene.camera;
        SetProcessDPIAware();
        HINSTANCE instance = GetModuleHandleW(nullptr);
        const wchar_t* className = L"UID166D3D11SceneWindow";
        WNDCLASSEXW windowClass{};
        windowClass.cbSize = sizeof(windowClass);
        windowClass.style = CS_HREDRAW | CS_VREDRAW;
        windowClass.lpfnWndProc = windowProcedure;
        windowClass.hInstance = instance;
        windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        windowClass.lpszClassName = className;
        if (!RegisterClassExW(&windowClass)) {
            throw uid166::AppError(uid166::ErrorCode::deviceInit, "RegisterClassExW failed");
        }
        RECT rect{0, 0, static_cast<LONG>(cli.windowWidth),
                  static_cast<LONG>(cli.windowHeight)};
        AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
        HWND window = CreateWindowExW(
            0, className, L"UID166 - D3D11 Skinned Character + Culled City",
            WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
            rect.right - rect.left, rect.bottom - rect.top,
            nullptr, nullptr, instance, nullptr);
        if (!window) {
            throw uid166::AppError(uid166::ErrorCode::deviceInit, "CreateWindowExW failed");
        }
        RECT client{};
        GetClientRect(window, &client);
        auto renderer = uid166::D3D11Renderer::create(
            window, static_cast<uint32_t>(client.right - client.left),
            static_cast<uint32_t>(client.bottom - client.top),
            std::move(data.scene), std::move(data.character));
        ShowWindow(window, SW_SHOW);
        UpdateWindow(window);

        const auto start = std::chrono::steady_clock::now();
        auto previous = start;
        uint64_t frame = 0;
        bool running = true;
        while (running) {
            if (cli.liveFrames != 0 && frame >= cli.liveFrames) break;
            MSG message{};
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
                if (message.message == WM_QUIT) {
                    running = false;
                    break;
                }
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
            if (!running) break;
            if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) break;
            RECT clientRect{};
            GetClientRect(window, &clientRect);
            const uint32_t width = static_cast<uint32_t>(std::max<LONG>(0, clientRect.right));
            const uint32_t height = static_cast<uint32_t>(std::max<LONG>(0, clientRect.bottom));
            if (width == 0 || height == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            renderer->resize(width, height);
            const auto now = std::chrono::steady_clock::now();
            float delta = std::chrono::duration<float>(now - previous).count();
            previous = now;
            delta = std::clamp(delta, 0.0f, 0.05f);
            const float elapsed = std::chrono::duration<float>(now - start).count();
            uid166::Camera camera = uid166::cameraAtTime(cameraConfig, elapsed);
            camera.aspect = static_cast<float>(width) / static_cast<float>(height);
            const uid166::LiveFrameStats stats = renderer->renderFrame(frame, delta, camera);
            std::cout << "{\"frame\":" << stats.frame
                      << ",\"frustumVisible\":" << stats.frustumVisible
                      << ",\"occluded\":" << stats.occluded
                      << ",\"visible\":" << stats.visible
                      << ",\"csmUpdated\":" << stats.csmUpdated
                      << ",\"localShadowUpdated\":" << stats.localShadowUpdated
                      << ",\"hwQueries\":" << stats.hardwareQueries
                      << ",\"hwResults\":" << stats.hardwareResults
                      << ",\"hwOccluded\":" << stats.hardwareOccluded
                      << ",\"occlusionQueryDraws\":" << stats.occlusionQueryDraws
                      << ",\"drawCalls\":" << stats.totalDrawCalls
                      << ",\"recordUs\":" << stats.recordingMicros
                      << ",\"recordBudgetExceeded\":"
                      << (stats.recordingBudgetExceeded ? "true" : "false")
                      << ",\"cameraFast\":" << (stats.cameraFast ? "true" : "false")
                      << ",\"gpuMs\":" << stats.gpuFrameMs
                      << ",\"qualityLevel\":" << stats.qualityLevel
                      << ",\"renderScale\":" << stats.renderScale
                      << ",\"shadowScale\":" << stats.shadowScale
                      << "}\n" << std::flush;
            ++frame;
        }
        return 0;
    } catch (const uid166::AppError& error) {
        uid166::reportError(error);
        return static_cast<int>(error.code());
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] code=5000 name=INTERNAL message=" << error.what() << '\n';
        return 5000;
    }
}
