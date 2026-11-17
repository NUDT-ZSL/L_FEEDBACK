#include "window.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace ct {

#ifdef _WIN32
namespace {

constexpr const char* kWindowClass = "CpuTerrainRendererMessageWindow";

LRESULT CALLBACK windowProcedure(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    switch (message) {
        case WM_CLOSE:
            DestroyWindow(window);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcA(window, message, wparam, lparam);
    }
}

}  // namespace

WindowMessageLoop::WindowMessageLoop() {
    HINSTANCE instance = GetModuleHandleA(nullptr);
    WNDCLASSEXA window_class{};
    window_class.cbSize = sizeof(window_class);
    window_class.lpfnWndProc = windowProcedure;
    window_class.hInstance = instance;
    window_class.lpszClassName = kWindowClass;
    RegisterClassExA(&window_class);
    HWND window = CreateWindowExA(0, kWindowClass, "CPU terrain renderer", 0, 0, 0, 0, 0,
                                  nullptr, nullptr, instance, nullptr);
    if (window == nullptr) {
        quit_ = true;
        return;
    }
    ShowWindow(window, SW_HIDE);
}

WindowMessageLoop::~WindowMessageLoop() {
    if (quit_) {
        return;
    }
    MSG message{};
    while (PeekMessageA(&message, nullptr, 0, 0, PM_REMOVE)) {
        if (message.message == WM_QUIT) {
            quit_ = true;
            break;
        }
        TranslateMessage(&message);
        DispatchMessageA(&message);
    }
}

bool WindowMessageLoop::pump() {
    if (quit_) {
        return false;
    }
    MSG message{};
    while (PeekMessageA(&message, nullptr, 0, 0, PM_REMOVE)) {
        if (message.message == WM_QUIT) {
            quit_ = true;
            return false;
        }
        TranslateMessage(&message);
        DispatchMessageA(&message);
    }
    return true;
}
#else
WindowMessageLoop::WindowMessageLoop() = default;
WindowMessageLoop::~WindowMessageLoop() = default;

bool WindowMessageLoop::pump() {
    return !quit_;
}
#endif

}  // namespace ct
