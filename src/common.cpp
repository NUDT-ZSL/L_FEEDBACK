#include "common.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>

#include <stb_image.h>
#include <stb_image_write.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace cr {
namespace {

bool isPermissionError(const std::error_code& error) {
    if (error == std::errc::permission_denied) {
        return true;
    }
#ifdef _WIN32
    return error.value() == ERROR_ACCESS_DENIED;
#else
    return error.value() == EACCES || error.value() == EPERM;
#endif
}

[[noreturn]] void throwIoError(const std::string& action, const fs::path& path,
                               const std::error_code& error) {
    const char* code = isPermissionError(error) ? E_IO_DENIED : E_IO;
    throw AppError(code, action + " '" + path.string() + "': " + error.message());
}

void durableFlush(const fs::path& path) {
#ifdef _WIN32
    const int descriptor = _open(path.string().c_str(), _O_RDONLY | _O_BINARY);
    if (descriptor < 0) {
        throw AppError(E_IO, "unable to reopen file for flush: " + path.string());
    }
    if (_commit(descriptor) != 0) {
        const int savedErrno = errno;
        _close(descriptor);
        throw AppError(E_IO, "flush failed for '" + path.string() + "': " +
                                 std::strerror(savedErrno));
    }
    _close(descriptor);
#else
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        const int savedErrno = errno;
        throw AppError(savedErrno == EACCES || savedErrno == EPERM ? E_IO_DENIED : E_IO,
                       "unable to reopen file for flush: " + path.string());
    }
    if (::fsync(descriptor) != 0) {
        const int savedErrno = errno;
        ::close(descriptor);
        throw AppError(savedErrno == EACCES || savedErrno == EPERM ? E_IO_DENIED : E_IO,
                       "flush failed for '" + path.string() + "': " +
                           std::strerror(savedErrno));
    }
    ::close(descriptor);
#endif
}

void atomicReplace(const fs::path& source, const fs::path& destination) {
#ifdef _WIN32
    const std::wstring sourceWide = source.wstring();
    const std::wstring destinationWide = destination.wstring();
    if (MoveFileExW(sourceWide.c_str(), destinationWide.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
        const DWORD error = GetLastError();
        const char* code = error == ERROR_ACCESS_DENIED ? E_IO_DENIED : E_IO;
        throw AppError(code, "atomic replace failed for '" + destination.string() + "'");
    }
#else
    std::error_code error;
    fs::rename(source, destination, error);
    if (error) {
        throwIoError("atomic replace failed for", destination, error);
    }
#endif
}

fs::path temporarySibling(const fs::path& path) {
    std::uint64_t nonce = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
#ifdef _WIN32
    nonce ^= static_cast<std::uint64_t>(GetCurrentProcessId()) << 32U;
#else
    nonce ^= static_cast<std::uint64_t>(::getpid()) << 32U;
#endif
    return path.parent_path() /
           (path.filename().string() + ".tmp." + std::to_string(nonce));
}

}  // namespace

AppError::AppError(std::string code, std::string message)
    : std::runtime_error(std::move(message)), code_(std::move(code)) {}

Vec3::Vec3(float xValue, float yValue, float zValue) : x(xValue), y(yValue), z(zValue) {}

Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator-(const Vec3& value) { return {-value.x, -value.y, -value.z}; }
Vec3 operator*(const Vec3& value, float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}
Vec3 operator*(float scalar, const Vec3& value) { return value * scalar; }
Vec3 operator*(const Vec3& a, const Vec3& b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
Vec3 operator/(const Vec3& value, float scalar) {
    return {value.x / scalar, value.y / scalar, value.z / scalar};
}
Vec3& operator+=(Vec3& a, const Vec3& b) {
    a = a + b;
    return a;
}
bool operator==(const Vec3& a, const Vec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
bool operator!=(const Vec3& a, const Vec3& b) { return !(a == b); }

float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float lengthSquared(const Vec3& value) { return dot(value, value); }
float length(const Vec3& value) { return std::sqrt(lengthSquared(value)); }
Vec3 normalize(const Vec3& value) {
    const float magnitude = length(value);
    if (magnitude <= std::numeric_limits<float>::epsilon()) {
        return {};
    }
    return value / magnitude;
}
Vec3 clamp(const Vec3& value, float minimum, float maximum) {
    return {clamp(value.x, minimum, maximum), clamp(value.y, minimum, maximum),
            clamp(value.z, minimum, maximum)};
}
float clamp(float value, float minimum, float maximum) {
    return std::max(minimum, std::min(maximum, value));
}
float mix(float a, float b, float t) { return a * (1.0F - t) + b * t; }
Vec3 mix(const Vec3& a, const Vec3& b, float t) { return a * (1.0F - t) + b * t; }

bool operator==(const Light& a, const Light& b) {
    return a.id == b.id && a.position == b.position && a.color == b.color &&
           a.intensity == b.intensity && a.radius == b.radius;
}
bool operator!=(const Light& a, const Light& b) { return !(a == b); }

Json vec3ToJson(const Vec3& value) { return Json::array({value.x, value.y, value.z}); }

Vec3 vec3FromJson(const Json& value, const std::string& context) {
    if (!value.is_array() || value.size() != 3U) {
        throw AppError(E_DATA_INVALID, context + " must be a three-element array");
    }
    for (const Json& component : value) {
        if (!component.is_number()) {
            throw AppError(E_DATA_INVALID, context + " contains a non-numeric component");
        }
    }
    return {value.at(0).get<float>(), value.at(1).get<float>(),
            value.at(2).get<float>()};
}

Json lightToJson(const Light& light) {
    return Json{{"id", light.id},
                {"position", vec3ToJson(light.position)},
                {"color", vec3ToJson(light.color)},
                {"intensity", light.intensity},
                {"radius", light.radius}};
}

Light lightFromJson(const Json& value, const std::string& context) {
    if (!value.is_object()) {
        throw AppError(E_DATA_INVALID, context + " must be an object");
    }
    Light light;
    light.id = value.at("id").get<std::string>();
    light.position = vec3FromJson(value.at("position"), context + ".position");
    light.color = vec3FromJson(value.at("color"), context + ".color");
    light.intensity = value.at("intensity").get<float>();
    light.radius = value.at("radius").get<float>();
    if (light.id.empty() || light.intensity < 0.0F || light.radius <= 0.0F) {
        throw AppError(E_DATA_INVALID, context + " contains invalid light fields");
    }
    return light;
}

Image::Image(int imageWidth, int imageHeight, int imageChannels)
    : width(imageWidth),
      height(imageHeight),
      channels(imageChannels),
      pixels(static_cast<std::size_t>(imageWidth) * static_cast<std::size_t>(imageHeight) *
             static_cast<std::size_t>(imageChannels)) {}

bool Image::empty() const noexcept { return width <= 0 || height <= 0 || pixels.empty(); }

std::uint8_t* Image::row(int y) {
    return pixels.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(width) *
                               static_cast<std::size_t>(channels);
}

const std::uint8_t* Image::row(int y) const {
    return pixels.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(width) *
                               static_cast<std::size_t>(channels);
}

Image loadPng(const fs::path& path) {
    const std::vector<std::uint8_t> encoded = readBinaryFile(path);
    int width = 0;
    int height = 0;
    int sourceChannels = 0;
    stbi_uc* decoded = stbi_load_from_memory(
        encoded.data(), static_cast<int>(encoded.size()), &width, &height, &sourceChannels, 4);
    if (decoded == nullptr) {
        const char* reason = stbi_failure_reason();
        throw AppError(E_TEX_LOAD, "failed to decode texture '" + path.string() +
                                       "': " + (reason != nullptr ? reason : "unknown error"));
    }

    Image image(width, height, 4);
    std::memcpy(image.pixels.data(), decoded,
                static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4U);
    stbi_image_free(decoded);
    return image;
}

void savePngAtomic(const fs::path& path, const Image& image) {
    if (image.empty() || image.channels != 4) {
        throw AppError(E_IO, "cannot write invalid RGBA image to '" + path.string() + "'");
    }
    std::vector<std::uint8_t> bytes;
    bytes.reserve(static_cast<std::size_t>(image.width) *
                  static_cast<std::size_t>(image.height) * 3U);
    auto append = [](void* context, void* data, int size) {
        if (context == nullptr || data == nullptr || size <= 0) {
            return;
        }
        auto* output = static_cast<std::vector<std::uint8_t>*>(context);
        const auto* begin = static_cast<const std::uint8_t*>(data);
        output->insert(output->end(), begin, begin + size);
    };
    const int result = stbi_write_png_to_func(
        append, &bytes, image.width, image.height, image.channels, image.pixels.data(),
        image.width * image.channels);
    if (result == 0 || bytes.empty()) {
        throw AppError(E_IO, "PNG encoding failed for '" + path.string() + "'");
    }
    writeBinaryAtomic(path, bytes);
}

bool imagesWithinTolerance(const Image& a, const Image& b, int channelTolerance,
                           std::string* detail) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels) {
        if (detail != nullptr) {
            *detail = "dimension mismatch: " + std::to_string(a.width) + "x" +
                      std::to_string(a.height) + " vs " + std::to_string(b.width) + "x" +
                      std::to_string(b.height);
        }
        return false;
    }
    for (std::size_t index = 0; index < a.pixels.size(); ++index) {
        const int difference = std::abs(static_cast<int>(a.pixels[index]) -
                                        static_cast<int>(b.pixels[index]));
        if (difference > channelTolerance) {
            if (detail != nullptr) {
                const std::size_t pixel = index / static_cast<std::size_t>(a.channels);
                const int channel = static_cast<int>(index % static_cast<std::size_t>(a.channels));
                *detail = "pixel (" + std::to_string(pixel % static_cast<std::size_t>(a.width)) +
                          "," + std::to_string(pixel / static_cast<std::size_t>(a.width)) +
                          ") channel " + std::to_string(channel) +
                          " differs by " + std::to_string(difference);
            }
            return false;
        }
    }
    return true;
}

std::string readTextFile(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        std::error_code error;
        const bool exists = fs::exists(path, error);
        if (error) {
            throwIoError("cannot inspect", path, error);
        }
        throw AppError(exists ? E_IO_DENIED : E_IO,
                       "cannot open file '" + path.string() + "'");
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (!stream.good() && !stream.eof()) {
        throw AppError(E_IO, "failed while reading '" + path.string() + "'");
    }
    return buffer.str();
}

std::vector<std::uint8_t> readBinaryFile(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        std::error_code error;
        const bool exists = fs::exists(path, error);
        if (error) {
            throwIoError("cannot inspect", path, error);
        }
        throw AppError(exists ? E_IO_DENIED : E_IO,
                       "cannot open file '" + path.string() + "'");
    }
    stream.seekg(0, std::ios::end);
    const std::streamoff size = stream.tellg();
    if (size < 0) {
        throw AppError(E_IO, "cannot determine size of '" + path.string() + "'");
    }
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()), size);
    }
    if (!stream) {
        throw AppError(E_IO, "failed while reading '" + path.string() + "'");
    }
    return bytes;
}

void writeTextAtomic(const fs::path& path, const std::string& content) {
    writeBinaryAtomic(path, std::vector<std::uint8_t>(content.begin(), content.end()));
}

void writeBinaryAtomic(const fs::path& path, const std::vector<std::uint8_t>& content) {
    ensureDirectory(path.parent_path());
    const fs::path temporary = temporarySibling(path);
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            const int savedErrno = errno;
            std::error_code error;
            fs::remove(temporary, error);
            const bool denied = savedErrno == EACCES || savedErrno == EPERM;
            throw AppError(denied ? E_IO_DENIED : E_IO,
                           "cannot create temporary file '" + temporary.string() + "'");
        }
        if (!content.empty()) {
            stream.write(reinterpret_cast<const char*>(content.data()),
                         static_cast<std::streamsize>(content.size()));
        }
        stream.flush();
        if (!stream) {
            stream.close();
            std::error_code removeError;
            fs::remove(temporary, removeError);
            throw AppError(E_IO, "failed while writing '" + temporary.string() + "'");
        }
    }
    durableFlush(temporary);
    atomicReplace(temporary, path);
}

void appendTextDurable(const fs::path& path, const std::string& content) {
    ensureDirectory(path.parent_path());
    {
        std::ofstream stream(path, std::ios::binary | std::ios::app);
        if (!stream) {
            const int savedErrno = errno;
            const bool denied = savedErrno == EACCES || savedErrno == EPERM;
            throw AppError(denied ? E_IO_DENIED : E_IO,
                           "cannot append commit log '" + path.string() + "'");
        }
        stream.write(content.data(), static_cast<std::streamsize>(content.size()));
        stream.flush();
        if (!stream) {
            throw AppError(E_IO, "failed while appending '" + path.string() + "'");
        }
    }
    durableFlush(path);
}

void ensureDirectory(const fs::path& path) {
    if (path.empty()) {
        return;
    }
    std::error_code error;
    fs::create_directories(path, error);
    if (error && !fs::is_directory(path)) {
        throwIoError("cannot create directory", path, error);
    }
}

bool pathExists(const fs::path& path) {
    std::error_code error;
    const bool exists = fs::exists(path, error);
    if (error) {
        throwIoError("cannot inspect", path, error);
    }
    return exists;
}

bool isRegularFile(const fs::path& path) {
    std::error_code error;
    const bool regular = fs::is_regular_file(path, error);
    if (error) {
        throwIoError("cannot inspect", path, error);
    }
    return regular;
}

std::string utcTimestampNow() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) %
        1000;
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3)
           << std::setfill('0') << milliseconds.count() << 'Z';
    return output.str();
}

std::string zeroPaddedVersion(std::uint64_t version) {
    std::ostringstream output;
    output << std::setw(6) << std::setfill('0') << version;
    return output.str();
}

std::uint64_t fnv1a64(const std::vector<std::uint8_t>& data) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::uint8_t byte : data) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t fnv1a64(const std::string& data) {
    return fnv1a64(std::vector<std::uint8_t>(data.begin(), data.end()));
}

int runProcessExitCodeForError() {
    return 1;
}

}  // namespace cr

