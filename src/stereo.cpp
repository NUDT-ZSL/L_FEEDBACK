#include "automix/stereo.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace automix {

double linearToDb(double value) {
    if (value <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    return 20.0 * std::log10(value);
}

double dbToLinear(double db) {
    return std::pow(10.0, db / 20.0);
}

LevelStats measureLevels(const StereoBuffer& buffer) {
    LevelStats stats;
    if (buffer.frames == 0) {
        stats.peakDbfs = -std::numeric_limits<double>::infinity();
        stats.rmsDbfs = -std::numeric_limits<double>::infinity();
        return stats;
    }

    double sumSquares = 0.0;
    for (const float sample : buffer.interleaved) {
        const double value = static_cast<double>(sample);
        stats.peak = std::max(stats.peak, std::abs(value));
        sumSquares += value * value;
    }
    stats.rms = std::sqrt(sumSquares / static_cast<double>(buffer.interleaved.size()));
    stats.peakDbfs = linearToDb(stats.peak);
    stats.rmsDbfs = linearToDb(stats.rms);
    return stats;
}

std::string pathToUtf8(const std::filesystem::path& path) {
    const auto utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

}  // namespace automix
