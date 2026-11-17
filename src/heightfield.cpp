#include "heightfield.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ct {
namespace {

std::uint64_t splitMix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

double unitFromBits(std::uint64_t bits) {
    return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

}  // namespace

HeightField::HeightField(const HeightfieldConfig& config) : config_(config), size_(config.size) {
    heights_.resize(static_cast<std::size_t>(size_) * static_cast<std::size_t>(size_));

    std::uint64_t state = static_cast<std::uint64_t>(config.seed) ^ 0xd1b54a32d192ed03ULL;
    struct Wave {
        double frequency;
        double angle;
        double phase;
        double amplitude;
        double cross_frequency;
        double cross_phase;
    };
    std::vector<Wave> waves;
    waves.reserve(static_cast<std::size_t>(config.octaves));
    double normalization = 0.0;
    for (int octave = 0; octave < config.octaves; ++octave) {
        Wave wave;
        wave.frequency = config.base_frequency * std::pow(2.0, octave);
        wave.angle = unitFromBits(splitMix64(state)) * 2.0 * kPi;
        wave.phase = unitFromBits(splitMix64(state)) * 2.0 * kPi;
        wave.amplitude = std::pow(0.46, octave);
        wave.cross_frequency = wave.frequency * (1.31 + 0.17 * octave);
        wave.cross_phase = unitFromBits(splitMix64(state)) * 2.0 * kPi;
        normalization += wave.amplitude * 1.4;
        waves.push_back(wave);
    }

    minimum_ = std::numeric_limits<double>::infinity();
    maximum_ = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (int z = 0; z < size_; ++z) {
        for (int x = 0; x < size_; ++x) {
            double value = 0.0;
            for (const Wave& wave : waves) {
                const double projection =
                    static_cast<double>(x) * std::cos(wave.angle) +
                    static_cast<double>(z) * std::sin(wave.angle);
                const double cross_projection =
                    static_cast<double>(x) * std::sin(wave.angle) -
                    static_cast<double>(z) * std::cos(wave.angle);
                value += wave.amplitude *
                         std::sin(2.0 * kPi * wave.frequency * projection + wave.phase);
                value += wave.amplitude * 0.32 *
                         std::cos(2.0 * kPi * wave.cross_frequency * cross_projection +
                                  wave.cross_phase);
            }
            value = config.amplitude * value / normalization;
            heights_[static_cast<std::size_t>(z) * size_ + static_cast<std::size_t>(x)] = value;
            minimum_ = std::min(minimum_, value);
            maximum_ = std::max(maximum_, value);
            sum += value;
        }
    }
    mean_ = sum / static_cast<double>(size_) / static_cast<double>(size_);
}

double HeightField::at(int x, int z) const {
    x = std::clamp(x, 0, size_ - 1);
    z = std::clamp(z, 0, size_ - 1);
    return heights_[static_cast<std::size_t>(z) * size_ + static_cast<std::size_t>(x)];
}

double HeightField::sampleBilinear(double x, double z) const {
    const double max_coordinate = static_cast<double>(size_ - 1);
    x = std::clamp(x, 0.0, max_coordinate);
    z = std::clamp(z, 0.0, max_coordinate);
    const int x0 = std::min(static_cast<int>(std::floor(x)), size_ - 1);
    const int z0 = std::min(static_cast<int>(std::floor(z)), size_ - 1);
    const int x1 = std::min(x0 + 1, size_ - 1);
    const int z1 = std::min(z0 + 1, size_ - 1);
    const double tx = x - static_cast<double>(x0);
    const double tz = z - static_cast<double>(z0);
    const double h00 = at(x0, z0);
    const double h10 = at(x1, z0);
    const double h01 = at(x0, z1);
    const double h11 = at(x1, z1);
    const double a = h00 * (1.0 - tx) + h10 * tx;
    const double b = h01 * (1.0 - tx) + h11 * tx;
    return a * (1.0 - tz) + b * tz;
}

Vec3 HeightField::normalAt(double x, double z) const {
    constexpr double kStep = 0.65;
    const double left = sampleBilinear(x - kStep, z);
    const double right = sampleBilinear(x + kStep, z);
    const double back = sampleBilinear(x, z - kStep);
    const double front = sampleBilinear(x, z + kStep);
    return normalize({
        (left - right) / (2.0 * kStep),
        1.0,
        (back - front) / (2.0 * kStep),
    });
}

std::uint32_t HeightField::materialAt(double x, double z) const {
    const double h = sampleBilinear(x, z);
    const double normalized = h / config_.amplitude;
    const Vec3 normal = normalAt(x, z);

    if (normalized > 0.53 || (normal.y < 0.53 && normalized > 0.33)) {
        return 3;
    }
    if (normal.y < 0.70) {
        return 2;
    }
    if (normalized < -0.18) {
        return 0;
    }
    return 1;
}

}  // namespace ct
