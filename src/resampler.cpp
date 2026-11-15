#include "automix/resampler.h"

#include "automix/error.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace automix {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

double besselI0(double value) {
    const double x = value * value / 4.0;
    double term = 1.0;
    double sum = 1.0;
    for (int k = 1; k < 64; ++k) {
        term *= x / static_cast<double>(k * k);
        sum += term;
        if (std::abs(term) < 1e-16 * std::abs(sum)) {
            break;
        }
    }
    return sum;
}

double sinc(double value) {
    if (std::abs(value) < 1e-12) {
        return 1.0;
    }
    const double angle = kPi * value;
    return std::sin(angle) / angle;
}

struct KernelDesign {
    int radius = 1;
    double cutoff = 0.25;
    double beta = 8.96;
    double denominator = 1.0;
};

KernelDesign designKernel(int sourceRate, int targetRate) {
    const double stop = 0.5 * static_cast<double>(std::min(sourceRate, targetRate));
    const double pass = std::min(20000.0, stop * 0.95);
    const double transition = std::max(1.0, stop - pass);
    constexpr double kAttenuationDb = 90.0;
    const double normalizedTransition = transition / static_cast<double>(sourceRate);
    const double estimatedTaps =
        (kAttenuationDb - 7.95) / (14.36 * normalizedTransition);
    int radius = static_cast<int>(std::ceil(estimatedTaps / 2.0)) + 2;
    radius = std::max(radius, 8);

    KernelDesign design;
    design.radius = radius;
    design.cutoff = ((pass + stop) * 0.5) / static_cast<double>(sourceRate);
    design.beta = 0.1102 * (kAttenuationDb - 8.7);
    design.denominator = besselI0(design.beta);
    return design;
}

double kernelValue(const KernelDesign& design, double distanceInSourceSamples) {
    const double x = distanceInSourceSamples / static_cast<double>(design.radius);
    if (std::abs(x) >= 1.0) {
        return 0.0;
    }
    const double window =
        besselI0(design.beta * std::sqrt(std::max(0.0, 1.0 - x * x))) /
        design.denominator;
    return 2.0 * design.cutoff * sinc(2.0 * design.cutoff * distanceInSourceSamples) *
           window;
}

std::uint64_t outputFrameCount(std::uint64_t sourceFrames, int sourceRate,
                               int targetRate) {
    const long double exact =
        static_cast<long double>(sourceFrames) * static_cast<long double>(targetRate) /
        static_cast<long double>(sourceRate);
    const long double rounded = std::ceil(exact - 1e-12L);
    if (rounded < 1.0L ||
        rounded > static_cast<long double>(std::numeric_limits<std::size_t>::max() / 2)) {
        throw AppError(ErrorCode::Internal, "resampled audio length is out of range");
    }
    return static_cast<std::uint64_t>(rounded);
}

}  // namespace

double BandlimitedResampler::passbandLimitForRates(int sourceRate, int targetRate) {
    const double stop = 0.5 * static_cast<double>(std::min(sourceRate, targetRate));
    return std::min(20000.0, stop * 0.95);
}

AudioData BandlimitedResampler::convert(const AudioData& input, int targetSampleRate) const {
    if (input.sampleRate <= 0 || targetSampleRate <= 0 || input.channels <= 0) {
        throw AppError(ErrorCode::Internal, "invalid sample rate or channel count");
    }
    if (input.sampleRate == targetSampleRate) {
        return input;
    }
    if (input.frames > std::numeric_limits<std::size_t>::max() /
                           static_cast<std::size_t>(input.channels)) {
        throw AppError(ErrorCode::Internal, "input audio is too large");
    }

    const int divisor = std::gcd(input.sampleRate, targetSampleRate);
    const std::uint64_t up = static_cast<std::uint64_t>(targetSampleRate / divisor);
    const std::uint64_t down = static_cast<std::uint64_t>(input.sampleRate / divisor);
    if (up > 65536) {
        throw AppError(ErrorCode::FileFormat,
                       "sample-rate ratio is too complex to resample deterministically");
    }

    const KernelDesign design = designKernel(input.sampleRate, targetSampleRate);
    const int tapCount = design.radius * 2 + 1;

    std::vector<double> coefficients(
        static_cast<std::size_t>(up) * static_cast<std::size_t>(tapCount));
    for (std::uint64_t phase = 0; phase < up; ++phase) {
        const double offset =
            static_cast<double>(phase) / static_cast<double>(up);
        double* row = coefficients.data() + static_cast<std::size_t>(phase) *
                                                static_cast<std::size_t>(tapCount);
        double sum = 0.0;
        for (int tap = 0; tap < tapCount; ++tap) {
            const int j = tap - design.radius;
            const double coefficient = kernelValue(design, offset - static_cast<double>(j));
            row[tap] = coefficient;
            sum += coefficient;
        }
        if (std::abs(sum) < 1e-12) {
            throw AppError(ErrorCode::Internal, "resampler kernel normalization failed");
        }
        for (int tap = 0; tap < tapCount; ++tap) {
            row[tap] /= sum;
        }
    }

    const std::uint64_t outFrames =
        outputFrameCount(input.frames, input.sampleRate, targetSampleRate);
    AudioData output;
    output.sampleRate = targetSampleRate;
    output.channels = input.channels;
    output.frames = outFrames;
    output.formatName = input.formatName;
    output.samples.resize(static_cast<std::size_t>(outFrames) *
                          static_cast<std::size_t>(input.channels));

    const std::int64_t sourceFrames = static_cast<std::int64_t>(input.frames);
    for (std::uint64_t outputIndex = 0; outputIndex < outFrames; ++outputIndex) {
        const std::uint64_t numerator = outputIndex * down;
        const std::uint64_t base = numerator / up;
        const std::uint64_t phase = numerator % up;
        const double* row =
            coefficients.data() +
            static_cast<std::size_t>(phase) * static_cast<std::size_t>(tapCount);
        for (int channel = 0; channel < input.channels; ++channel) {
            double accumulator = 0.0;
            for (int tap = 0; tap < tapCount; ++tap) {
                const std::int64_t sourceIndex =
                    static_cast<std::int64_t>(base) + (tap - design.radius);
                if (sourceIndex < 0 || sourceIndex >= sourceFrames) {
                    continue;
                }
                const float sample =
                    input.samples[static_cast<std::size_t>(sourceIndex) *
                                      static_cast<std::size_t>(input.channels) +
                                  static_cast<std::size_t>(channel)];
                accumulator += static_cast<double>(sample) * row[tap];
            }
            output.samples[static_cast<std::size_t>(outputIndex) *
                               static_cast<std::size_t>(input.channels) +
                           static_cast<std::size_t>(channel)] =
                static_cast<float>(accumulator);
        }
    }
    return output;
}

}  // namespace automix
