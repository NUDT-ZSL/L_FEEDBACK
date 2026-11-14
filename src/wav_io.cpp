#include "wav_io.h"

#include "spatial_audio/audio_engine.h"

#include <sndfile.h>

#include <cmath>
#include <limits>

namespace spatial_audio::detail {
namespace {

std::string sndfileError(const char* prefix) {
    const char* detail = sf_strerror(nullptr);
    std::string message(prefix);
    if (detail != nullptr && detail[0] != '\0') {
        message += ": ";
        message += detail;
    }
    return message;
}

}  // namespace

bool readMonoWav16(const std::string& path,
                   std::vector<float>& samples,
                   std::string& error) {
    samples.clear();

    SF_INFO info{};
    SNDFILE* file = sf_open(path.c_str(), SFM_READ, &info);
    if (file == nullptr) {
        error = sndfileError("cannot open WAV");
        return false;
    }

    const int majorFormat = info.format & SF_FORMAT_TYPEMASK;
    const int subFormat = info.format & SF_FORMAT_SUBMASK;
    if (majorFormat != SF_FORMAT_WAV && majorFormat != SF_FORMAT_WAVEX) {
        error = "unsupported container: expected WAV";
        sf_close(file);
        return false;
    }
    if (subFormat != SF_FORMAT_PCM_16) {
        error = "unsupported sample format: expected 16-bit PCM";
        sf_close(file);
        return false;
    }
    if (info.channels != 1) {
        error = "unsupported channel count: expected mono";
        sf_close(file);
        return false;
    }
    if (info.samplerate != kSampleRate) {
        error = "unsupported sample rate: expected 44100 Hz";
        sf_close(file);
        return false;
    }
    if (info.frames <= 0) {
        error = "WAV contains no audio frames";
        sf_close(file);
        return false;
    }

    std::vector<std::int16_t> pcm(static_cast<std::size_t>(info.frames));
    sf_count_t framesRead = sf_readf_short(file, pcm.data(), info.frames);
    const int closeResult = sf_close(file);
    if (framesRead != info.frames) {
        error = "short read while decoding WAV";
        return false;
    }
    if (closeResult != 0) {
        error = "error while closing WAV";
        return false;
    }

    samples.resize(pcm.size());
    constexpr float scale = 1.0f / 32768.0f;
    for (std::size_t i = 0; i < pcm.size(); ++i) {
        samples[i] = static_cast<float>(pcm[i]) * scale;
    }
    return true;
}

bool writeMonoWav16(const std::string& path,
                    const std::vector<std::int16_t>& samples,
                    std::string& error) {
    SF_INFO info{};
    info.samplerate = kSampleRate;
    info.channels = 1;
    info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* file = sf_open(path.c_str(), SFM_WRITE, &info);
    if (file == nullptr) {
        error = sndfileError("cannot create WAV");
        return false;
    }

    sf_count_t written = 0;
    if (!samples.empty()) {
        written = sf_writef_short(file, samples.data(),
                                  static_cast<sf_count_t>(samples.size()));
    }
    const int closeResult = sf_close(file);
    if (written != static_cast<sf_count_t>(samples.size())) {
        error = "short write while encoding WAV";
        return false;
    }
    if (closeResult != 0) {
        error = "error while closing WAV";
        return false;
    }
    return true;
}

bool writeStereoWav16(const std::string& path,
                      const std::vector<std::int16_t>& interleavedStereo,
                      std::string& error) {
    if (interleavedStereo.size() % 2 != 0) {
        error = "stereo sample vector has an odd length";
        return false;
    }

    SF_INFO info{};
    info.samplerate = kSampleRate;
    info.channels = 2;
    info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* file = sf_open(path.c_str(), SFM_WRITE, &info);
    if (file == nullptr) {
        error = sndfileError("cannot create WAV");
        return false;
    }

    const sf_count_t frames =
        static_cast<sf_count_t>(interleavedStereo.size() / 2);
    sf_count_t written = 0;
    if (frames > 0) {
        written = sf_writef_short(file, interleavedStereo.data(), frames);
    }
    const int closeResult = sf_close(file);
    if (written != frames) {
        error = "short write while encoding WAV";
        return false;
    }
    if (closeResult != 0) {
        error = "error while closing WAV";
        return false;
    }
    return true;
}

bool inspectStereoWav16(const std::string& path,
                        std::uint64_t expectedFrames,
                        std::string& error) {
    SF_INFO info{};
    SNDFILE* file = sf_open(path.c_str(), SFM_READ, &info);
    if (file == nullptr) {
        error = sndfileError("cannot reopen output WAV");
        return false;
    }
    const int closeResult = sf_close(file);

    const int majorFormat = info.format & SF_FORMAT_TYPEMASK;
    const int subFormat = info.format & SF_FORMAT_SUBMASK;
    if (majorFormat != SF_FORMAT_WAV && majorFormat != SF_FORMAT_WAVEX) {
        error = "output is not a WAV file";
        return false;
    }
    if (subFormat != SF_FORMAT_PCM_16) {
        error = "output is not 16-bit PCM";
        return false;
    }
    if (info.channels != 2) {
        error = "output is not stereo";
        return false;
    }
    if (info.samplerate != kSampleRate) {
        error = "output is not 44100 Hz";
        return false;
    }
    if (info.frames < 0 ||
        static_cast<std::uint64_t>(info.frames) != expectedFrames) {
        error = "output frame count mismatch";
        return false;
    }
    if (closeResult != 0) {
        error = "error while closing output WAV";
        return false;
    }
    return true;
}

}  // namespace spatial_audio::detail
