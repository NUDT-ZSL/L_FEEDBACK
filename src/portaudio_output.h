#pragma once

#include "spatial_audio/audio_engine.h"

#include <string>

#if defined(SPATIAL_AUDIO_HAS_PORTAUDIO) && SPATIAL_AUDIO_HAS_PORTAUDIO
#include <portaudio.h>
#endif

namespace spatial_audio {

class PortAudioOutput {
public:
    explicit PortAudioOutput(AudioEngine& engine) noexcept;
    ~PortAudioOutput();

    PortAudioOutput(const PortAudioOutput&) = delete;
    PortAudioOutput& operator=(const PortAudioOutput&) = delete;

    // Returns ErrorCode::Ok on success. On failure, error contains the
    // PortAudio diagnostic and the main program must return a non-zero code.
    ErrorCode start(std::string& error);
    void stop() noexcept;

private:
#if defined(SPATIAL_AUDIO_HAS_PORTAUDIO) && SPATIAL_AUDIO_HAS_PORTAUDIO
    static int audioCallback(const void* input,
                             void* output,
                             unsigned long frameCount,
                             const PaStreamCallbackTimeInfo* timeInfo,
                             PaStreamCallbackFlags statusFlags,
                             void* userData) noexcept;

    PaStream* stream_ = nullptr;
    bool initialized_ = false;
#endif
    AudioEngine& engine_;
};

}  // namespace spatial_audio

