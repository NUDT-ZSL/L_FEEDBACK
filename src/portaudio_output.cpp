#include "portaudio_output.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace spatial_audio {

PortAudioOutput::PortAudioOutput(AudioEngine& engine) noexcept : engine_(engine) {}

PortAudioOutput::~PortAudioOutput() {
    stop();
}

ErrorCode PortAudioOutput::start(std::string& error) {
#if defined(SPATIAL_AUDIO_HAS_PORTAUDIO) && SPATIAL_AUDIO_HAS_PORTAUDIO
    stop();

    PaError result = Pa_Initialize();
    if (result != paNoError) {
        error = std::string("Pa_Initialize: ") + Pa_GetErrorText(result);
        return ErrorCode::PortAudioInitFailed;
    }
    initialized_ = true;

    const PaDeviceIndex device = Pa_GetDefaultOutputDevice();
    if (device == paNoDevice) {
        error = "Pa_GetDefaultOutputDevice: no default output device";
        Pa_Terminate();
        initialized_ = false;
        return ErrorCode::PortAudioStreamOpenFailed;
    }
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(device);
    if (deviceInfo == nullptr) {
        error = "Pa_GetDeviceInfo returned null";
        Pa_Terminate();
        initialized_ = false;
        return ErrorCode::PortAudioStreamOpenFailed;
    }

    PaStreamParameters outputParameters{};
    outputParameters.device = device;
    outputParameters.channelCount = static_cast<int>(kChannels);
    outputParameters.sampleFormat = paInt16;
    outputParameters.suggestedLatency = deviceInfo->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = nullptr;

    result = Pa_OpenStream(
        &stream_, nullptr, &outputParameters, kSampleRate,
        static_cast<unsigned long>(kBlockFrames), paClipOff, &audioCallback,
        this);
    if (result != paNoError) {
        error = std::string("Pa_OpenStream: ") + Pa_GetErrorText(result);
        stream_ = nullptr;
        Pa_Terminate();
        initialized_ = false;
        return ErrorCode::PortAudioStreamOpenFailed;
    }

    result = Pa_StartStream(stream_);
    if (result != paNoError) {
        error = std::string("Pa_StartStream: ") + Pa_GetErrorText(result);
        Pa_CloseStream(stream_);
        stream_ = nullptr;
        Pa_Terminate();
        initialized_ = false;
        return ErrorCode::PortAudioStreamStartFailed;
    }
    return ErrorCode::Ok;
#else
    error = "this build was configured without PortAudio "
            "(-DSPATIAL_AUDIO_BUILD_REALTIME=OFF)";
    return ErrorCode::PortAudioInitFailed;
#endif
}

void PortAudioOutput::stop() noexcept {
#if defined(SPATIAL_AUDIO_HAS_PORTAUDIO) && SPATIAL_AUDIO_HAS_PORTAUDIO
    if (stream_ != nullptr) {
        if (Pa_IsStreamActive(stream_) == 1) {
            Pa_StopStream(stream_);
        }
        Pa_CloseStream(stream_);
        stream_ = nullptr;
    }
    if (initialized_) {
        Pa_Terminate();
        initialized_ = false;
    }
#endif
}

#if defined(SPATIAL_AUDIO_HAS_PORTAUDIO) && SPATIAL_AUDIO_HAS_PORTAUDIO
int PortAudioOutput::audioCallback(
    const void* input,
    void* output,
    unsigned long frameCount,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void* userData) noexcept {
    (void)input;
    (void)timeInfo;
    (void)statusFlags;

    auto* self = static_cast<PortAudioOutput*>(userData);
    auto* destination = static_cast<std::int16_t*>(output);
    std::array<std::int16_t, kBlockFrames * kChannels> rendered{};

    unsigned long remaining = frameCount;
    while (remaining > 0) {
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<unsigned long>(remaining, kBlockFrames));
        self->engine_.processBlock(rendered.data(), chunk);
        std::memcpy(destination, rendered.data(),
                    chunk * kChannels * sizeof(std::int16_t));
        destination += chunk * kChannels;
        remaining -= static_cast<unsigned long>(chunk);
    }
    return paContinue;
}
#endif

}  // namespace spatial_audio

