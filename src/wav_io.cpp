#include "automix/wav_io.h"

#include "automix/error.h"

#include <sndfile.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <system_error>

namespace automix {

namespace {

struct SoundFileCloser {
    void operator()(SNDFILE* file) const {
        if (file != nullptr) {
            sf_close(file);
        }
    }
};

using SoundFilePtr = std::unique_ptr<SNDFILE, SoundFileCloser>;

bool supportedMajorFormat(int major) {
    switch (major) {
        case SF_FORMAT_WAV:
        case SF_FORMAT_WAVEX:
        case SF_FORMAT_RF64:
        case SF_FORMAT_W64:
        case SF_FORMAT_AIFF:
        case SF_FORMAT_AU:
        case SF_FORMAT_FLAC:
            return true;
        default:
            return false;
    }
}

const char* majorFormatName(int major) {
    switch (major) {
        case SF_FORMAT_WAV: return "WAV";
        case SF_FORMAT_WAVEX: return "WAVEX";
        case SF_FORMAT_RF64: return "RF64";
        case SF_FORMAT_W64: return "W64";
        case SF_FORMAT_AIFF: return "AIFF";
        case SF_FORMAT_AU: return "AU";
        case SF_FORMAT_FLAC: return "FLAC";
        default: return "unknown";
    }
}

const char* subtypeName(int subtype) {
    switch (subtype) {
        case SF_FORMAT_PCM_16: return "PCM_16";
        case SF_FORMAT_PCM_24: return "PCM_24";
        case SF_FORMAT_FLOAT: return "FLOAT_32";
        default: return "unsupported";
    }
}

bool supportedSubtype(int subtype) {
    return subtype == SF_FORMAT_PCM_16 || subtype == SF_FORMAT_PCM_24 ||
           subtype == SF_FORMAT_FLOAT;
}

std::uint16_t readU16(const unsigned char* data) {
    return static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8));
}

std::uint32_t readU32(const unsigned char* data) {
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8) |
           (static_cast<std::uint32_t>(data[2]) << 16) |
           (static_cast<std::uint32_t>(data[3]) << 24);
}

void validateRiffContainer(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return;
    }
    std::array<unsigned char, 12> header{};
    input.read(reinterpret_cast<char*>(header.data()),
               static_cast<std::streamsize>(header.size()));
    if (input.gcount() != static_cast<std::streamsize>(header.size()) ||
        std::memcmp(header.data(), "RIFF", 4) != 0 ||
        std::memcmp(header.data() + 8, "WAVE", 4) != 0) {
        return;
    }

    std::error_code ec;
    const std::uint64_t fileSize = std::filesystem::file_size(path, ec);
    if (ec) {
        return;
    }
    const std::uint64_t riffEnd =
        static_cast<std::uint64_t>(readU32(header.data() + 4)) + 8u;
    if (riffEnd > fileSize) {
        throw AppError(ErrorCode::FileCorrupt,
                       "RIFF container is truncated in '" + pathToUtf8(path) + "'");
    }

    std::uint64_t offset = 12;
    std::uint32_t blockAlign = 0;
    bool sawData = false;
    while (offset + 8 <= fileSize) {
        input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        std::array<unsigned char, 8> chunk{};
        input.read(reinterpret_cast<char*>(chunk.data()),
                   static_cast<std::streamsize>(chunk.size()));
        if (input.gcount() != static_cast<std::streamsize>(chunk.size())) {
            throw AppError(ErrorCode::FileCorrupt,
                           "RIFF chunk header is truncated in '" + pathToUtf8(path) +
                               "'");
        }
        const std::uint32_t chunkSize = readU32(chunk.data() + 4);
        const std::uint64_t body = offset + 8;
        const std::uint64_t end = body + chunkSize;
        if (end > fileSize) {
            throw AppError(ErrorCode::FileCorrupt,
                           "RIFF chunk extends beyond the file in '" +
                               pathToUtf8(path) + "'");
        }
        if (std::memcmp(chunk.data(), "fmt ", 4) == 0 && chunkSize >= 14) {
            std::array<unsigned char, 14> format{};
            input.seekg(static_cast<std::streamoff>(body), std::ios::beg);
            input.read(reinterpret_cast<char*>(format.data()),
                       static_cast<std::streamsize>(format.size()));
            if (input.gcount() == static_cast<std::streamsize>(format.size())) {
                blockAlign = readU16(format.data() + 12);
            }
        } else if (std::memcmp(chunk.data(), "data", 4) == 0) {
            sawData = true;
            if (blockAlign > 0 && chunkSize % blockAlign != 0) {
                throw AppError(ErrorCode::FileCorrupt,
                               "WAV data chunk is not frame-aligned in '" +
                                   pathToUtf8(path) + "'");
            }
        }
        offset = end + (chunkSize & 1u);
    }
    if (sawData && offset > fileSize + 1) {
        throw AppError(ErrorCode::FileCorrupt,
                       "WAV chunk padding is truncated in '" + pathToUtf8(path) + "'");
    }
}

std::string formatLabel(const SF_INFO& info) {
    return std::string(majorFormatName(info.format & SF_FORMAT_TYPEMASK)) + "/" +
           subtypeName(info.format & SF_FORMAT_SUBMASK);
}

SNDFILE* openSoundFile(const std::filesystem::path& path, int mode, SF_INFO* info) {
#ifdef _WIN32
    return sf_wchar_open(path.c_str(), mode, info);
#else
    return sf_open(path.string().c_str(), mode, info);
#endif
}

[[noreturn]] void removeAndThrowOutput(const std::filesystem::path& path,
                                       const std::string& message) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    throw AppError(ErrorCode::OutputIo, message);
}

}  // namespace

AudioData readAudioFile(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        throw AppError(ErrorCode::FileMissing,
                       "input file does not exist: " + pathToUtf8(path));
    }
    if (ec || !std::filesystem::is_regular_file(path, ec)) {
        throw AppError(ErrorCode::FileFormat,
                       "input path is not a regular file: " + pathToUtf8(path));
    }
    validateRiffContainer(path);

    SF_INFO info{};
    SoundFilePtr file(openSoundFile(path, SFM_READ, &info));
    if (!file) {
        const int error = sf_error(nullptr);
        if (error == SF_ERR_SYSTEM && errno == ENOENT) {
            throw AppError(ErrorCode::FileMissing,
                           "input file does not exist: " + pathToUtf8(path));
        }
        throw AppError(ErrorCode::FileFormat,
                       "unsupported or unreadable audio file '" + pathToUtf8(path) +
                           "': " + sf_strerror(nullptr));
    }

    const int major = info.format & SF_FORMAT_TYPEMASK;
    const int subtype = info.format & SF_FORMAT_SUBMASK;
    if (!supportedMajorFormat(major) || !supportedSubtype(subtype)) {
        throw AppError(ErrorCode::FileFormat,
                       "unsupported audio format in '" + pathToUtf8(path) + "': " +
                           formatLabel(info));
    }
    if (info.channels != 1 && info.channels != 2) {
        throw AppError(ErrorCode::FileFormat,
                       "unsupported channel count in '" + pathToUtf8(path) + "': " +
                           std::to_string(info.channels));
    }
    if (info.samplerate <= 0 || info.samplerate > 384000) {
        throw AppError(ErrorCode::FileFormat,
                       "unsupported sample rate in '" + pathToUtf8(path) + "': " +
                           std::to_string(info.samplerate));
    }
    if (info.frames <= 0) {
        throw AppError(ErrorCode::FileFormat,
                       "audio file contains no sample frames: " + pathToUtf8(path));
    }
    if (static_cast<std::uint64_t>(info.frames) >
        std::numeric_limits<std::size_t>::max() /
            static_cast<std::size_t>(info.channels)) {
        throw AppError(ErrorCode::FileCorrupt,
                       "audio frame count is too large: " + pathToUtf8(path));
    }

    AudioData audio;
    audio.sampleRate = info.samplerate;
    audio.channels = info.channels;
    audio.frames = static_cast<std::uint64_t>(info.frames);
    audio.formatName = formatLabel(info);
    audio.samples.resize(static_cast<std::size_t>(audio.frames) *
                         static_cast<std::size_t>(audio.channels));

    constexpr sf_count_t kChunkFrames = 4096;
    std::vector<float> chunk(static_cast<std::size_t>(kChunkFrames) *
                             static_cast<std::size_t>(info.channels));
    std::uint64_t writtenFrames = 0;
    while (writtenFrames < audio.frames) {
        const sf_count_t want = static_cast<sf_count_t>(
            std::min<std::uint64_t>(kChunkFrames, audio.frames - writtenFrames));
        const sf_count_t got = sf_readf_float(file.get(), chunk.data(), want);
        if (got <= 0) {
            throw AppError(ErrorCode::FileCorrupt,
                           "sample data ended before the declared frame count in '" +
                               pathToUtf8(path) + "'");
        }
        const std::size_t sampleOffset =
            static_cast<std::size_t>(writtenFrames) *
            static_cast<std::size_t>(info.channels);
        const std::size_t sampleCount =
            static_cast<std::size_t>(got) * static_cast<std::size_t>(info.channels);
        std::copy_n(chunk.data(), sampleCount, audio.samples.data() + sampleOffset);
        writtenFrames += static_cast<std::uint64_t>(got);
    }

    if (writtenFrames != audio.frames) {
        throw AppError(ErrorCode::FileCorrupt,
                       "decoded frame count does not match header in '" + pathToUtf8(path) +
                           "'");
    }
    for (const float sample : audio.samples) {
        if (!std::isfinite(sample) || std::abs(sample) > 64.0f) {
            throw AppError(ErrorCode::FileCorrupt,
                           "invalid floating-point sample in '" + pathToUtf8(path) + "'");
        }
    }
    return audio;
}

void writeFloatWav(const std::filesystem::path& path, const StereoBuffer& buffer) {
    if (buffer.frames == 0 || buffer.interleaved.size() != buffer.frames * 2) {
        throw AppError(ErrorCode::Internal, "attempted to write an invalid stereo buffer");
    }
    if (buffer.sampleRate != 48000) {
        throw AppError(ErrorCode::Internal, "output buffer must be 48000 Hz");
    }

    std::error_code ec;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) {
            throw AppError(ErrorCode::OutputIo,
                           "cannot create output directory '" +
                               pathToUtf8(path.parent_path()) + "': " + ec.message());
        }
    }
    std::filesystem::remove(path, ec);
    ec.clear();

    SF_INFO info{};
    info.samplerate = buffer.sampleRate;
    info.channels = 2;
    info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    SoundFilePtr file(openSoundFile(path, SFM_WRITE, &info));
    if (!file) {
        throw AppError(ErrorCode::OutputIo,
                       "cannot create output WAV '" + pathToUtf8(path) +
                           "': " + sf_strerror(nullptr));
    }
    sf_command(file.get(), SFC_SET_ADD_PEAK_CHUNK, nullptr, SF_FALSE);

    std::uint64_t written = 0;
    while (written < buffer.frames) {
        const sf_count_t want = static_cast<sf_count_t>(
            std::min<std::uint64_t>(4096, buffer.frames - written));
        const float* data = buffer.interleaved.data() + static_cast<std::size_t>(written) * 2;
        const sf_count_t got = sf_writef_float(file.get(), data, want);
        if (got != want) {
            removeAndThrowOutput(path, "failed while writing output WAV '" +
                                           pathToUtf8(path) + "': " +
                                           sf_strerror(file.get()));
        }
        written += static_cast<std::uint64_t>(got);
    }

    SNDFILE* raw = file.release();
    if (sf_close(raw) != 0) {
        removeAndThrowOutput(path,
                             "failed to finalize output WAV '" + pathToUtf8(path) + "'");
    }
}

}  // namespace automix
