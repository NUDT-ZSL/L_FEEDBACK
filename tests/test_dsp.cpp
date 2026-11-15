#include "test_support.h"

#include "automix/resampler.h"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

automix::AudioData makeTone(int sampleRate, double frequency, double amplitude,
                            double seconds) {
    automix::AudioData audio;
    audio.sampleRate = sampleRate;
    audio.channels = 1;
    audio.frames = static_cast<std::uint64_t>(
        std::llround(static_cast<double>(sampleRate) * seconds));
    audio.formatName = "TEST";
    audio.samples.resize(static_cast<std::size_t>(audio.frames));
    for (std::size_t frame = 0; frame < audio.samples.size(); ++frame) {
        audio.samples[frame] = static_cast<float>(
            amplitude * std::sin(2.0 * kPi * frequency *
                                 static_cast<double>(frame) /
                                 static_cast<double>(sampleRate)));
    }
    return audio;
}

double centralRms(const automix::AudioData& audio) {
    const std::size_t begin =
        std::min<std::size_t>(audio.samples.size() / 5, audio.samples.size());
    const std::size_t end =
        audio.samples.size() - std::min<std::size_t>(audio.samples.size() / 10,
                                                     audio.samples.size() - begin);
    double sum = 0.0;
    for (std::size_t index = begin; index < end; ++index) {
        sum += static_cast<double>(audio.samples[index]) *
               static_cast<double>(audio.samples[index]);
    }
    return std::sqrt(sum / static_cast<double>(end - begin));
}

double toneAmplitude(const automix::AudioData& audio, double frequency) {
    constexpr std::size_t kFftSize = 65536;
    const std::size_t begin = audio.samples.size() / 4;
    test::require(begin + kFftSize <= audio.samples.size(),
                  "not enough samples for FFT");

    std::vector<float> input(kFftSize);
    double windowSum = 0.0;
    for (std::size_t index = 0; index < kFftSize; ++index) {
        const double window =
            0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(index) /
                                 static_cast<double>(kFftSize - 1));
        input[index] =
            static_cast<float>(static_cast<double>(audio.samples[begin + index]) * window);
        windowSum += window;
    }

    fftwf_complex* output = fftwf_alloc_complex(kFftSize / 2 + 1);
    test::require(output != nullptr, "FFTW allocation failed");
    fftwf_plan plan =
        fftwf_plan_dft_r2c_1d(static_cast<int>(kFftSize), input.data(), output,
                              FFTW_ESTIMATE);
    test::require(plan != nullptr, "FFTW plan creation failed");
    fftwf_execute(plan);

    const std::size_t bin = static_cast<std::size_t>(
        std::llround(frequency * static_cast<double>(kFftSize) /
                     static_cast<double>(audio.sampleRate)));
    const double magnitude =
        std::hypot(output[bin][0], output[bin][1]) * 2.0 / windowSum;
    fftwf_destroy_plan(plan);
    fftwf_free(output);
    return magnitude;
}

}  // namespace

void testResampler() {
    automix::BandlimitedResampler resampler;
    const std::vector<double> passband = {
        1000.0, 5000.0, 10000.0, 15000.0, 18000.0, 19500.0, 20000.0};
    double worstDeviation = 0.0;
    for (const double frequency : passband) {
        const auto input = makeTone(44100, frequency, 0.5, 1.2);
        const auto output = resampler.convert(input, 48000);
        const double measured = centralRms(output);
        const double reference = 0.5 / std::sqrt(2.0);
        const double deviation =
            20.0 * std::log10(measured / reference);
        worstDeviation = std::max(worstDeviation, std::abs(deviation));
        test::require(std::abs(deviation) <= 0.1,
                      "resampler passband deviation exceeds 0.1 dB at " +
                          std::to_string(frequency) + " Hz: " +
                          std::to_string(deviation) + " dB");
    }
    std::cout << "METRIC resampler_worst_passband_deviation_db " << worstDeviation
              << '\n';

    const auto input = makeTone(96000, 30000.0, 0.5, 2.0);
    const auto output = resampler.convert(input, 48000);
    const double aliasAmplitude = toneAmplitude(output, 18000.0);
    const double aliasSuppression =
        20.0 * std::log10(0.5 / std::max(aliasAmplitude, 1e-12));
    std::cout << "METRIC resampler_alias_suppression_db " << aliasSuppression
              << '\n';
    test::require(aliasSuppression >= 60.0,
                  "resampler alias suppression is below 60 dB");

    const auto again = resampler.convert(input, 48000);
    test::require(output.samples.size() == again.samples.size(),
                  "deterministic resampler length mismatch");
    for (std::size_t index = 0; index < output.samples.size(); ++index) {
        test::require(output.samples[index] == again.samples[index],
                      "resampler output is not sample-deterministic");
    }
}
