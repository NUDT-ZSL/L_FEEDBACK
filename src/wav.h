#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "sha256.h"

namespace vr {

constexpr std::uint32_t kSampleRate = 48000;
constexpr std::uint16_t kBitsPerSample = 16;
constexpr std::size_t kWavHeaderSize = 44;

std::array<std::uint8_t, 44> make_wav_header(std::uint64_t frames, std::uint16_t channels);
std::uint64_t wav_size_for_frames(std::uint64_t frames, std::uint16_t channels);

class WavWriter {
public:
    WavWriter() = default;
    ~WavWriter();
    WavWriter(const WavWriter&) = delete;
    WavWriter& operator=(const WavWriter&) = delete;
    WavWriter(WavWriter&& other) noexcept;
    WavWriter& operator=(WavWriter&& other) noexcept;

    void create(const std::filesystem::path& path, std::uint64_t frames, std::uint16_t channels);
    void resume(const std::filesystem::path& path, std::uint64_t frames, std::uint16_t channels,
                std::uint64_t prefix_frames, const std::string& prefix_sha256);

    void write_frames(const std::int16_t* samples, std::uint64_t frames);
    void flush();
    void close();

    std::string prefix_sha256() const { return pcm_hash_.digest_hex(); }
    std::uint64_t written_frames() const noexcept { return written_frames_; }
    const std::filesystem::path& path() const noexcept { return path_; }

private:
    void verify_header();
    void seed_hash_from_prefix(std::uint64_t prefix_frames);

    std::filesystem::path path_;
    std::fstream file_;
    std::uint64_t total_frames_ = 0;
    std::uint64_t written_frames_ = 0;
    std::uint16_t channels_ = 0;
    Sha256 pcm_hash_;
};

}  // namespace vr

