#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace automix {

struct StereoBuffer {
    int sampleRate = 48000;
    std::size_t frames = 0;
    std::vector<float> interleaved;

    StereoBuffer() = default;
    StereoBuffer(int rate, std::size_t frameCount)
        : sampleRate(rate), frames(frameCount), interleaved(frameCount * 2, 0.0f) {}

    float& left(std::size_t n) { return interleaved[n * 2]; }
    float& right(std::size_t n) { return interleaved[n * 2 + 1]; }
    float left(std::size_t n) const { return interleaved[n * 2]; }
    float right(std::size_t n) const { return interleaved[n * 2 + 1]; }
};

struct LevelStats {
    double peak = 0.0;
    double rms = 0.0;
    double peakDbfs = 0.0;
    double rmsDbfs = 0.0;
};

double linearToDb(double value);
double dbToLinear(double db);
LevelStats measureLevels(const StereoBuffer& buffer);
std::string pathToUtf8(const std::filesystem::path& path);

}  // namespace automix
