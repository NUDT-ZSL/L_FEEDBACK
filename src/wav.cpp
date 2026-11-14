#include "wav.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <vector>

#include "error.h"

namespace vr {

namespace {

void put_u16(std::uint8_t* out, std::uint16_t value) {
    out[0] = static_cast<std::uint8_t>(value & 0xFFu);
    out[1] = static_cast<std::uint8_t>((value >> 8) & 0xFFu);
}

void put_u32(std::uint8_t* out, std::uint32_t value) {
    out[0] = static_cast<std::uint8_t>(value & 0xFFu);
    out[1] = static_cast<std::uint8_t>((value >> 8) & 0xFFu);
    out[2] = static_cast<std::uint8_t>((value >> 16) & 0xFFu);
    out[3] = static_cast<std::uint8_t>((value >> 24) & 0xFFu);
}

std::uint16_t get_u16(const std::uint8_t* in) {
    return static_cast<std::uint16_t>(in[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(in[1]) << 8);
}

std::uint32_t get_u32(const std::uint8_t* in) {
    return static_cast<std::uint32_t>(in[0]) |
           (static_cast<std::uint32_t>(in[1]) << 8) |
           (static_cast<std::uint32_t>(in[2]) << 16) |
           (static_cast<std::uint32_t>(in[3]) << 24);
}

}  // namespace

std::uint64_t wav_size_for_frames(std::uint64_t frames, std::uint16_t channels) {
    return kWavHeaderSize + frames * channels * 2u;
}

std::array<std::uint8_t, 44> make_wav_header(std::uint64_t frames, std::uint16_t channels) {
    const std::uint64_t data_bytes = frames * channels * 2u;
    if (data_bytes > 0xFFFFFFFFull - 36ull) {
        throw AppError(ErrorCode::MetaInvalid, "WAV output exceeds the RIFF 4 GiB size limit");
    }
    std::array<std::uint8_t, 44> header{};
    std::memcpy(header.data(), "RIFF", 4);
    put_u32(header.data() + 4, static_cast<std::uint32_t>(data_bytes + 36u));
    std::memcpy(header.data() + 8, "WAVE", 4);
    std::memcpy(header.data() + 12, "fmt ", 4);
    put_u32(header.data() + 16, 16u);
    put_u16(header.data() + 20, 1u);
    put_u16(header.data() + 22, channels);
    put_u32(header.data() + 24, kSampleRate);
    put_u32(header.data() + 28, kSampleRate * channels * 2u);
    put_u16(header.data() + 32, static_cast<std::uint16_t>(channels * 2u));
    put_u16(header.data() + 34, kBitsPerSample);
    std::memcpy(header.data() + 36, "data", 4);
    put_u32(header.data() + 40, static_cast<std::uint32_t>(data_bytes));
    return header;
}

WavWriter::~WavWriter() {
    close();
}

WavWriter::WavWriter(WavWriter&& other) noexcept
    : path_(std::move(other.path_)),
      file_(std::move(other.file_)),
      total_frames_(other.total_frames_),
      written_frames_(other.written_frames_),
      channels_(other.channels_),
      pcm_hash_(other.pcm_hash_) {}

WavWriter& WavWriter::operator=(WavWriter&& other) noexcept {
    if (this != &other) {
        close();
        path_ = std::move(other.path_);
        file_ = std::move(other.file_);
        total_frames_ = other.total_frames_;
        written_frames_ = other.written_frames_;
        channels_ = other.channels_;
        pcm_hash_ = other.pcm_hash_;
    }
    return *this;
}

void WavWriter::create(const std::filesystem::path& path, std::uint64_t frames,
                       std::uint16_t channels) {
    close();
    path_ = path;
    total_frames_ = frames;
    written_frames_ = 0;
    channels_ = channels;
    pcm_hash_ = Sha256();

    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        throw AppError(ErrorCode::OutputUnwritable,
                       "cannot create output directory: " + path.parent_path().string());
    }
    {
        std::ofstream create_file(path, std::ios::binary | std::ios::trunc);
        if (!create_file) {
            throw AppError(ErrorCode::OutputUnwritable,
                           "cannot create output file: " + path.string());
        }
        const auto header = make_wav_header(frames, channels);
        create_file.write(reinterpret_cast<const char*>(header.data()),
                          static_cast<std::streamsize>(header.size()));
        if (!create_file) {
            throw AppError(ErrorCode::OutputUnwritable,
                           "cannot write output header: " + path.string());
        }
    }
    file_.open(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file_) {
        throw AppError(ErrorCode::OutputUnwritable, "cannot open output file: " + path.string());
    }
    file_.seekp(static_cast<std::streamoff>(kWavHeaderSize));
}

void WavWriter::verify_header() {
    std::array<std::uint8_t, 44> header{};
    file_.seekg(0);
    file_.read(reinterpret_cast<char*>(header.data()),
               static_cast<std::streamsize>(header.size()));
    if (file_.gcount() != static_cast<std::streamsize>(header.size())) {
        throw AppError(ErrorCode::StateCorrupt,
                       "WAV header is missing or truncated: " + path_.string());
    }
    const auto expected = make_wav_header(total_frames_, channels_);
    if (!std::equal(header.begin(), header.end(), expected.begin())) {
        throw AppError(ErrorCode::StateCorrupt,
                       "WAV header does not match the resume state: " + path_.string());
    }
    if (get_u16(header.data() + 22) != channels_ ||
        get_u32(header.data() + 24) != kSampleRate ||
        get_u16(header.data() + 34) != kBitsPerSample ||
        get_u32(header.data() + 40) !=
            static_cast<std::uint32_t>(total_frames_ * channels_ * 2u)) {
        throw AppError(ErrorCode::StateCorrupt,
                       "invalid WAV format in resume output: " + path_.string());
    }
    file_.clear();
}

void WavWriter::seed_hash_from_prefix(std::uint64_t prefix_frames) {
    const std::uint64_t bytes64 = prefix_frames * channels_ * 2u;
    if (bytes64 > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw AppError(ErrorCode::StateCorrupt, "resume prefix is too large");
    }
    const auto bytes = static_cast<std::streamsize>(bytes64);
    file_.seekg(static_cast<std::streamoff>(kWavHeaderSize));
    std::vector<char> buffer(1024 * 1024);
    std::streamsize remaining = bytes;
    while (remaining > 0) {
        const auto take = std::min<std::streamsize>(remaining,
                                                    static_cast<std::streamsize>(buffer.size()));
        file_.read(buffer.data(), take);
        if (file_.gcount() != take) {
            throw AppError(ErrorCode::StateCorrupt,
                           "output is shorter than the resume state: " + path_.string());
        }
        pcm_hash_.update(buffer.data(), static_cast<std::size_t>(take));
        remaining -= take;
    }
}

void WavWriter::resume(const std::filesystem::path& path, std::uint64_t frames,
                       std::uint16_t channels, std::uint64_t prefix_frames,
                       const std::string& prefix_sha256) {
    close();
    path_ = path;
    total_frames_ = frames;
    written_frames_ = 0;
    channels_ = channels;
    pcm_hash_ = Sha256();

    if (prefix_frames > frames) {
        throw AppError(ErrorCode::StateCorrupt, "resume prefix exceeds total frames");
    }
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        throw AppError(ErrorCode::StateCorrupt,
                       "output file required by resume state is missing: " + path.string());
    }
    const std::uint64_t expected_size = wav_size_for_frames(prefix_frames, channels);
    const std::uint64_t actual_size = std::filesystem::file_size(path, ec);
    if (ec || actual_size < expected_size) {
        throw AppError(ErrorCode::StateCorrupt,
                       "output file is shorter than the resume state: " + path.string());
    }

    std::filesystem::resize_file(path, expected_size, ec);
    if (ec) {
        throw AppError(ErrorCode::OutputUnwritable,
                       "cannot truncate output to resume boundary: " + path.string());
    }

    file_.open(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file_) {
        throw AppError(ErrorCode::OutputUnwritable, "cannot open output file: " + path.string());
    }
    verify_header();
    seed_hash_from_prefix(prefix_frames);
    if (pcm_hash_.digest_hex() != prefix_sha256) {
        throw AppError(ErrorCode::StateCorrupt,
                       "output prefix digest does not match the resume state: " + path.string());
    }
    written_frames_ = prefix_frames;
    file_.clear();
    file_.seekp(static_cast<std::streamoff>(kWavHeaderSize + prefix_frames * channels * 2u));
}

void WavWriter::write_frames(const std::int16_t* samples, std::uint64_t frames) {
    if (written_frames_ + frames > total_frames_) {
        throw AppError(ErrorCode::Internal, "attempt to write past the WAV payload length");
    }
    const std::uint64_t bytes64 = frames * channels_ * 2u;
    if (bytes64 > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw AppError(ErrorCode::Internal, "WAV write is too large");
    }
    const auto bytes = static_cast<std::streamsize>(bytes64);
    file_.write(reinterpret_cast<const char*>(samples), bytes);
    if (!file_) {
        throw AppError(ErrorCode::Io, "failed to write WAV payload: " + path_.string());
    }
    pcm_hash_.update(reinterpret_cast<const char*>(samples), static_cast<std::size_t>(bytes));
    written_frames_ += frames;
}

void WavWriter::flush() {
    if (!file_) {
        return;
    }
    file_.flush();
    if (!file_) {
        throw AppError(ErrorCode::Io, "failed to flush WAV payload: " + path_.string());
    }
}

void WavWriter::close() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

}  // namespace vr

