#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace cr {

using Json = nlohmann::json;
namespace fs = std::filesystem;

inline constexpr const char* E_MAT_PARSE = "E_MAT_PARSE";
inline constexpr const char* E_TEX_LOAD = "E_TEX_LOAD";
inline constexpr const char* E_SNAPSHOT_NOT_FOUND = "E_SNAPSHOT_NOT_FOUND";
inline constexpr const char* E_SNAPSHOT_CONFLICT = "E_SNAPSHOT_CONFLICT";
inline constexpr const char* E_IO_DENIED = "E_IO_DENIED";
inline constexpr const char* E_IO = "E_IO";
inline constexpr const char* E_RENDER_NONDETERMINISTIC = "E_RENDER_NONDETERMINISTIC";
inline constexpr const char* E_OPENGL_UNAVAILABLE = "E_OPENGL_UNAVAILABLE";
inline constexpr const char* E_OPENGL = "E_OPENGL";
inline constexpr const char* E_DATA_INVALID = "E_DATA_INVALID";

class AppError final : public std::runtime_error {
public:
    AppError(std::string code, std::string message);

    [[nodiscard]] const std::string& code() const noexcept { return code_; }

private:
    std::string code_;
};

struct Vec3 {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;

    Vec3() = default;
    Vec3(float xValue, float yValue, float zValue);
};

Vec3 operator+(const Vec3& a, const Vec3& b);
Vec3 operator-(const Vec3& a, const Vec3& b);
Vec3 operator-(const Vec3& value);
Vec3 operator*(const Vec3& value, float scalar);
Vec3 operator*(float scalar, const Vec3& value);
Vec3 operator*(const Vec3& a, const Vec3& b);
Vec3 operator/(const Vec3& value, float scalar);
Vec3& operator+=(Vec3& a, const Vec3& b);
bool operator==(const Vec3& a, const Vec3& b);
bool operator!=(const Vec3& a, const Vec3& b);

float dot(const Vec3& a, const Vec3& b);
Vec3 cross(const Vec3& a, const Vec3& b);
float lengthSquared(const Vec3& value);
float length(const Vec3& value);
Vec3 normalize(const Vec3& value);
Vec3 clamp(const Vec3& value, float minimum, float maximum);
float clamp(float value, float minimum, float maximum);
float mix(float a, float b, float t);
Vec3 mix(const Vec3& a, const Vec3& b, float t);

struct Camera {
    int width = 640;
    int height = 480;
    Vec3 position;
    Vec3 target;
    Vec3 up{0.0F, 1.0F, 0.0F};
    float fovYDegrees = 38.0F;
    float nearPlane = 0.1F;
    float farPlane = 20.0F;
};

struct Light {
    std::string id;
    Vec3 position;
    Vec3 color{1.0F, 1.0F, 1.0F};
    float intensity = 1.0F;
    float radius = 8.0F;
};

bool operator==(const Light& a, const Light& b);
bool operator!=(const Light& a, const Light& b);

Json vec3ToJson(const Vec3& value);
Vec3 vec3FromJson(const Json& value, const std::string& context);
Json lightToJson(const Light& light);
Light lightFromJson(const Json& value, const std::string& context);

struct Image {
    int width = 0;
    int height = 0;
    int channels = 4;
    std::vector<std::uint8_t> pixels;

    Image() = default;
    Image(int imageWidth, int imageHeight, int imageChannels);

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::uint8_t* row(int y);
    [[nodiscard]] const std::uint8_t* row(int y) const;
};

Image loadPng(const fs::path& path);
void savePngAtomic(const fs::path& path, const Image& image);
bool imagesWithinTolerance(const Image& a, const Image& b, int channelTolerance,
                           std::string* detail = nullptr);

std::string readTextFile(const fs::path& path);
std::vector<std::uint8_t> readBinaryFile(const fs::path& path);
void writeTextAtomic(const fs::path& path, const std::string& content);
void writeBinaryAtomic(const fs::path& path, const std::vector<std::uint8_t>& content);
void appendTextDurable(const fs::path& path, const std::string& content);
void ensureDirectory(const fs::path& path);
bool pathExists(const fs::path& path);
bool isRegularFile(const fs::path& path);

std::string utcTimestampNow();
std::string zeroPaddedVersion(std::uint64_t version);
std::uint64_t fnv1a64(const std::vector<std::uint8_t>& data);
std::uint64_t fnv1a64(const std::string& data);

int runProcessExitCodeForError();

}  // namespace cr

