#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <windowsx.h>
#if __has_include(<DirectXMath.h>)
#include <DirectXMath.h>
#elif __has_include(<directxmath.h>)
#include <directxmath.h>
#else
#error "DirectXMath headers are required (install the Windows SDK C++ desktop workload)"
#endif

#include "assets.h"
#include "image_io.h"
#include "options.h"
#include "renderer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

namespace {

struct WindowState {
    cv::Renderer* renderer = nullptr;
    cv::Scene* scene = nullptr;
    cv::Lighting* lighting = nullptr;
    cv::Camera camera;
    cv::AppOptions options;
    uint32_t tick = 0;
    bool paused = false;
    bool wireframe = false;
    bool running = true;
    bool dirty = true;
    bool dragging = false;
    DirectX::XMFLOAT2 lastMouse{0.0f, 0.0f};
    UINT ppmVk = 'P';
    UINT bmpVk = 'B';
    std::vector<uint8_t> dib;
    int dibWidth = 0;
    int dibHeight = 0;
    uint64_t renderedFrames = 0;
    std::ofstream logFile;
    std::string logPath;
};

WindowState* g_state = nullptr;

void logMessage(const std::string& message) {
    std::cout << message << "\n";
    if (g_state && g_state->logFile) {
        g_state->logFile << message << "\n";
        g_state->logFile.flush();
    }
}

UINT asciiToVirtualKey(char key) {
    const SHORT scan = VkKeyScanA(key);
    if (scan == -1) {
        return 0;
    }
    return static_cast<UINT>(scan & 0xff);
}

void rebuildDib(WindowState& state) {
    const int width = state.renderer->width();
    const int height = state.renderer->height();
    if (width == state.dibWidth && height == state.dibHeight && !state.dib.empty()) {
        return;
    }
    state.dibWidth = width;
    state.dibHeight = height;
    state.dib.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * 4, 0);
}

void copyImageToDib(WindowState& state) {
    const cv::Image& image = state.renderer->image();
    rebuildDib(state);
    if (!image.valid() || image.width != state.dibWidth ||
        image.height != state.dibHeight) {
        return;
    }
    for (size_t pixel = 0;
         pixel < static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
         ++pixel) {
        const size_t src = pixel * 3;
        const size_t dst = pixel * 4;
        state.dib[dst + 0] = image.rgb[src + 2];
        state.dib[dst + 1] = image.rgb[src + 1];
        state.dib[dst + 2] = image.rgb[src + 0];
        state.dib[dst + 3] = 0;
    }
}

void present(WindowState& state, HDC dc, int destinationWidth, int destinationHeight) {
    if (state.dib.empty()) {
        return;
    }
    BITMAPINFO bitmapInfo{};
    bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmapInfo.bmiHeader.biWidth = state.dibWidth;
    bitmapInfo.bmiHeader.biHeight = -state.dibHeight;
    bitmapInfo.bmiHeader.biPlanes = 1;
    bitmapInfo.bmiHeader.biBitCount = 32;
    bitmapInfo.bmiHeader.biCompression = BI_RGB;
    StretchDIBits(dc, 0, 0, destinationWidth, destinationHeight, 0, 0, state.dibWidth,
                  state.dibHeight, state.dib.data(), &bitmapInfo, DIB_RGB_COLORS,
                  SRCCOPY);
}

void logFrameStatistics(WindowState& state, bool force) {
    const cv::FrameStats& stats = state.renderer->stats();
    if (!force && state.renderedFrames % 30 != 0) {
        return;
    }
    std::ostringstream out;
    out << "[cull] tick=" << state.tick << " total=" << stats.totalTriangles
        << " grid_candidates=" << stats.gridCandidateTriangles
        << " visible=" << stats.gridVisibleTriangles << " ratio=" << std::fixed
        << std::setprecision(4) << stats.cullRatio << " cells=" << stats.visibleCells
        << "/" << stats.totalCells;
    logMessage(out.str());
    out.str({});
    out.clear();
    out << "[draw] opaque=" << stats.opaqueTriangles
        << " transparent=" << stats.transparentTriangles
        << " opaque_cmds=" << stats.opaqueCommands
        << " transparent_cmds=" << stats.transparentCommands
        << " state_switches=" << stats.stateSwitches
        << " transparent_order=" << (stats.transparentOrderValid ? "ok" : "bad")
        << " shadow_maps=" << stats.shadowMaps << " tick=" << state.tick;
    logMessage(out.str());
}

void renderFrame(WindowState& state) {
    state.renderer->render(*state.scene, *state.lighting, state.camera, state.tick,
                           state.wireframe);
    copyImageToDib(state);
    ++state.renderedFrames;
    state.dirty = false;
    logFrameStatistics(state, false);
}

void exportImage(WindowState& state, bool ppm) {
    if (state.dirty) {
        renderFrame(state);
    }
    const cv::ExportReport report =
        ppm ? cv::exportPpmP6(state.options.ppmPath, state.renderer->image())
            : cv::exportBmp24(state.options.bmpPath, state.renderer->image());
    if (report.ok) {
        std::ostringstream out;
        out << "[export] " << (ppm ? "PPM" : "BMP") << " path=" << report.path
            << " pixels=0x" << std::hex << report.pixelHash << std::dec
            << " header=valid";
        logMessage(out.str());
    } else {
        logMessage(std::string("[export-error] ") + (ppm ? "PPM " : "BMP ") +
                   report.error);
    }
}

void processKeyDown(WindowState& state, HWND window, WPARAM key, LPARAM lParam) {
    const bool repeat = (lParam & (1ll << 30)) != 0;
    if (repeat) {
        return;
    }
    if (key == VK_ESCAPE) {
        PostMessageW(window, WM_CLOSE, 0, 0);
        return;
    }
    if (key == VK_SPACE) {
        state.paused = !state.paused;
        logMessage(std::string("[input] ") + (state.paused ? "paused" : "running"));
        return;
    }
    if (key == 'N') {
        if (state.paused) {
            ++state.tick;
            state.dirty = true;
            logMessage("[input] single-step tick=" + std::to_string(state.tick));
        }
        return;
    }
    if (key == 'F') {
        state.wireframe = !state.wireframe;
        state.dirty = true;
        logMessage(std::string("[input] mode=") +
                   (state.wireframe ? "wireframe" : "solid"));
        return;
    }
    if (state.ppmVk != 0 && key == state.ppmVk) {
        exportImage(state, true);
        return;
    }
    if (state.bmpVk != 0 && key == state.bmpVk) {
        exportImage(state, false);
    }
}

LRESULT CALLBACK windowProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
    WindowState* state = g_state;
    switch (message) {
        case WM_ERASEBKGND:
            return 1;
        case WM_SIZE: {
            if (!state || !state->renderer) {
                return 0;
            }
            const int width = LOWORD(lParam);
            const int height = HIWORD(lParam);
            if (width > 0 && height > 0) {
                state->renderer->resize(width, height);
                state->dirty = true;
            }
            return 0;
        }
        case WM_PAINT: {
            PAINTSTRUCT paint;
            HDC dc = BeginPaint(window, &paint);
            RECT client{};
            GetClientRect(window, &client);
            if (state) {
                present(*state, dc, client.right - client.left, client.bottom - client.top);
            }
            EndPaint(window, &paint);
            return 0;
        }
        case WM_KEYDOWN:
            if (state) {
                processKeyDown(*state, window, wParam, lParam);
            }
            return 0;
        case WM_LBUTTONDOWN:
            if (state) {
                state->dragging = true;
                state->lastMouse = {static_cast<float>(GET_X_LPARAM(lParam)),
                                    static_cast<float>(GET_Y_LPARAM(lParam))};
                SetCapture(window);
            }
            return 0;
        case WM_LBUTTONUP:
            if (state) {
                state->dragging = false;
                ReleaseCapture();
            }
            return 0;
        case WM_MOUSEMOVE:
            if (state && state->dragging) {
                const float x = static_cast<float>(GET_X_LPARAM(lParam));
                const float y = static_cast<float>(GET_Y_LPARAM(lParam));
                const float dx = x - state->lastMouse.x;
                const float dy = y - state->lastMouse.y;
                state->lastMouse = {x, y};
                state->camera.yawRadians += dx * 0.008f;
                state->camera.pitchRadians =
                    std::clamp(state->camera.pitchRadians + dy * 0.006f, -1.15f, 1.15f);
                state->dirty = true;
            }
            return 0;
        case WM_MOUSEWHEEL:
            if (state) {
                const int wheel = GET_WHEEL_DELTA_WPARAM(wParam);
                state->camera.distance =
                    std::clamp(state->camera.distance - static_cast<float>(wheel) / 120.0f *
                                                         0.55f,
                               2.6f, 18.0f);
                state->dirty = true;
            }
            return 0;
        case WM_CLOSE:
            DestroyWindow(window);
            return 0;
        case WM_DESTROY:
            ReleaseCapture();
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(window, message, wParam, lParam);
    }
}

}  // namespace

int main(int argc, char** argv) {
    cv::AppOptions options = cv::parseOptions(argc, argv);
    if (options.helpRequested) {
        cv::printUsage(argv[0]);
        return 0;
    }
    if (!options.valid) {
        std::cerr << "error: " << options.error << "\n";
        cv::printUsage(argv[0]);
        return 2;
    }

    SetProcessDPIAware();
    cv::AssetBundle assets = cv::makeProceduralScene();
    cv::RendererConfig config;
    config.width = options.width;
    config.height = options.height;
    config.cascadeResolution = options.cascadeResolution;
    config.pointShadowResolution = options.pointShadowResolution;
    config.shadows = options.shadows;
    cv::Renderer renderer(assets.scene, assets.lighting, config);

    WindowState state;
    state.renderer = &renderer;
    state.scene = &assets.scene;
    state.lighting = &assets.lighting;
    state.options = options;
    state.tick = options.startTick;
    state.paused = options.startPaused;
    state.wireframe = options.wireframe;
    if (options.hasYaw) {
        state.camera.yawRadians = options.yawDegrees * cv::kPi / 180.0f;
    }
    if (options.hasPitch) {
        state.camera.pitchRadians = options.pitchDegrees * cv::kPi / 180.0f;
    }
    if (options.hasCameraDistance) {
        state.camera.distance = options.cameraDistance;
    }
    state.ppmVk = asciiToVirtualKey(options.ppmKey);
    state.bmpVk = asciiToVirtualKey(options.bmpKey);
    state.logPath = "charview.log";
    state.logFile.open(state.logPath, std::ios::out | std::ios::trunc);
    g_state = &state;

    WNDCLASSEXW windowClass{};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    windowClass.lpfnWndProc = windowProc;
    windowClass.hInstance = GetModuleHandleW(nullptr);
    windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    windowClass.hbrBackground = nullptr;
    windowClass.lpszClassName = L"CharViewSoftRendererWindow";
    if (!RegisterClassExW(&windowClass)) {
        std::cerr << "RegisterClassExW failed: " << GetLastError() << "\n";
        return 1;
    }

    RECT desired{0, 0, options.width, options.height};
    AdjustWindowRect(&desired, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(
        0, windowClass.lpszClassName, L"CharView - deterministic soft rasterizer",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, desired.right - desired.left,
        desired.bottom - desired.top, nullptr, nullptr, windowClass.hInstance, nullptr);
    if (!window) {
        std::cerr << "CreateWindowExW failed: " << GetLastError() << "\n";
        return 1;
    }
    ShowWindow(window, SW_SHOW);
    UpdateWindow(window);

    logMessage("[start] CharView deterministic software renderer");
    logMessage("[config] framebuffer=" + std::to_string(options.width) + "x" +
               std::to_string(options.height) + " cascade=" +
               std::to_string(config.cascadeResolution) + " cube=" +
               std::to_string(config.pointShadowResolution) + " tick=" +
               std::to_string(options.startTick));
    logMessage("[keys] Space pause, N step, F wireframe, " +
               std::string(1, options.ppmKey) + " PPM, " +
               std::string(1, options.bmpKey) + " BMP, Esc quit");
    logMessage("[export] ppm=" + options.ppmPath + " bmp=" + options.bmpPath +
               " log=" + state.logPath);

    using Clock = std::chrono::steady_clock;
    auto last = Clock::now();
    double accumulator = 0.0;
    constexpr double kFixedDt = 1.0 / 30.0;

    while (state.running) {
        MSG message{};
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
            if (message.message == WM_QUIT) {
                state.running = false;
                break;
            }
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        if (!state.running) {
            break;
        }

        const auto now = Clock::now();
        double elapsed = std::chrono::duration<double>(now - last).count();
        last = now;
        elapsed = std::clamp(elapsed, 0.0, 0.25);
        if (!state.paused) {
            accumulator += elapsed;
            while (accumulator >= kFixedDt) {
                accumulator -= kFixedDt;
                ++state.tick;
                state.dirty = true;
            }
        } else {
            accumulator = 0.0;
        }

        if (state.dirty) {
            renderFrame(state);
            InvalidateRect(window, nullptr, FALSE);
            UpdateWindow(window);
        }
        Sleep(1);
    }

    if (state.logFile) {
        state.logFile.flush();
        state.logFile.close();
    }
    UnregisterClassW(windowClass.lpszClassName, windowClass.hInstance);
    g_state = nullptr;
    return 0;
}

