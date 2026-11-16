#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <directxmath.h>
#include <dxgi.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

template <typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

struct DeviceLostError : std::runtime_error {
    explicit DeviceLostError(HRESULT result)
        : std::runtime_error("Direct3D device was removed or reset"), hr(result) {}

    HRESULT hr;
};

inline std::string HResultText(HRESULT hr) {
    char* buffer = nullptr;
    const DWORD length = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, static_cast<DWORD>(hr), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<char*>(&buffer), 0, nullptr);

    std::ostringstream stream;
    stream << "0x" << std::hex << std::uppercase << static_cast<unsigned long>(hr);
    if (length != 0 && buffer != nullptr) {
        std::string message(buffer, length);
        while (!message.empty() &&
               (message.back() == '\r' || message.back() == '\n' || message.back() == ' ')) {
            message.pop_back();
        }
        stream << " (" << message << ")";
    }
    if (buffer != nullptr) {
        LocalFree(buffer);
    }
    return stream.str();
}

inline bool IsDeviceLostResult(HRESULT hr) {
    return hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET ||
           hr == DXGI_ERROR_DRIVER_INTERNAL_ERROR || hr == DXGI_ERROR_DEVICE_HUNG;
}

inline void ThrowIfFailed(HRESULT hr, const char* operation) {
    if (SUCCEEDED(hr)) {
        return;
    }
    if (IsDeviceLostResult(hr)) {
        throw DeviceLostError(hr);
    }
    throw std::runtime_error(std::string(operation) + " failed: " + HResultText(hr));
}

inline std::string NarrowAscii(std::wstring_view value) {
    std::string result;
    result.reserve(value.size());
    for (wchar_t ch : value) {
        result.push_back(ch >= 0 && ch <= 0x7f ? static_cast<char>(ch) : '?');
    }
    return result;
}

inline std::filesystem::path ExecutableDirectory() {
    std::vector<wchar_t> buffer(32768);
    const DWORD length = GetModuleFileNameW(
        nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        throw std::runtime_error("GetModuleFileNameW failed");
    }
    return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
}

