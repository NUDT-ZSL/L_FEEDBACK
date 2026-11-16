#pragma once

#include "types.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mv {

class Sha256 {
public:
    Sha256();
    void update(const void* data, std::size_t size);
    void update(std::string_view text);
    std::string hexDigest();

    static std::string hex(const std::string& data);
    static std::string file(const std::filesystem::path& path);

private:
    void transform(const std::uint8_t* block);

    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> buffer_{};
    std::uint64_t bitCount_ = 0;
    std::size_t bufferSize_ = 0;
    bool finalized_ = false;
};

std::string readTextFile(const std::filesystem::path& path);
std::vector<std::uint8_t> readBinaryFile(const std::filesystem::path& path);
void writeBinaryFile(const std::filesystem::path& path, const void* data, std::size_t size);
void writeTextFile(const std::filesystem::path& path, const std::string& text);
void writeTextFileAtomic(const std::filesystem::path& path, const std::string& text);
void writeBinaryFileAtomic(const std::filesystem::path& path, const void* data, std::size_t size);

std::string trim(std::string value);
std::vector<std::string> split(std::string_view value, char delimiter);
std::string join(const std::vector<std::string>& values, std::string_view delimiter);
std::string sanitizeName(std::string value);
std::string formatFloat(float value);
std::string sha256OfStrings(const std::vector<std::string>& values);

std::uint32_t splitmix32(std::uint32_t& state);
std::vector<std::uint8_t> generateProceduralTexture(const TextureSpec& spec);
std::array<std::uint8_t, 4> proceduralTexel(const TextureSpec& spec, int x, int y);

MaterialError makeError(ExitCode code,
                        std::string errorCode,
                        std::string stage,
                        std::string message);

}  // namespace mv
