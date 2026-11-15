#pragma once

#include "automix/stereo.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace automix {

struct AudioData {
    int sampleRate = 0;
    int channels = 0;
    std::uint64_t frames = 0;
    std::string formatName;
    std::vector<float> samples;
};

AudioData readAudioFile(const std::filesystem::path& path);
void writeFloatWav(const std::filesystem::path& path, const StereoBuffer& buffer);

}  // namespace automix
