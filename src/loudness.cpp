#include "automix/loudness.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace automix {

namespace {

struct Biquad {
    double b0 = 1.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
    double x1 = 0.0;
    double x2 = 0.0;
    double y1 = 0.0;
    double y2 = 0.0;

    double process(double input) {
        const double output =
            b0 * input + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = input;
        y2 = y1;
        y1 = output;
        return output;
    }
};

Biquad kWeightingShelf() {
    Biquad filter;
    filter.b0 = 1.53512485958697;
    filter.b1 = -2.69169618940638;
    filter.b2 = 1.19839281085285;
    filter.a1 = -1.69065929318241;
    filter.a2 = 0.73248077421585;
    return filter;
}

Biquad kWeightingHighPass() {
    Biquad filter;
    filter.b0 = 1.0;
    filter.b1 = -2.0;
    filter.b2 = 1.0;
    filter.a1 = -1.99004745483398;
    filter.a2 = 0.99007225036621;
    return filter;
}

constexpr double kTruePeakCoefficients[12][4] = {
    {0.0017089843750, -0.0291748046875, -0.0189208984375, -0.0083007812500},
    {0.0109863281250, 0.0292968750000, 0.0330810546875, 0.0148925781250},
    {-0.0196533203125, -0.0517578125000, -0.0582275390625, -0.0266113281250},
    {0.0332031250000, 0.0891113281250, 0.1015625000000, 0.0476074218750},
    {-0.0594482421875, -0.1665039062500, -0.2003173828125, -0.1022949218750},
    {0.1373291015625, 0.4650878906250, 0.7797851562500, 0.9721679687500},
    {0.9721679687500, 0.7797851562500, 0.4650878906250, 0.1373291015625},
    {-0.1022949218750, -0.2003173828125, -0.1665039062500, -0.0594482421875},
    {0.0476074218750, 0.1015625000000, 0.0891113281250, 0.0332031250000},
    {-0.0266113281250, -0.0582275390625, -0.0517578125000, -0.0196533203125},
    {0.0148925781250, 0.0330810546875, 0.0292968750000, 0.0109863281250},
    {-0.0083007812500, -0.0189208984375, -0.0291748046875, 0.0017089843750},
};

double truePeakChannel(const StereoBuffer& buffer, int channel) {
    double peak = 0.0;
    const std::size_t frames = buffer.frames;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const double sample = channel == 0 ? buffer.left(frame) : buffer.right(frame);
        peak = std::max(peak, std::abs(sample));
    }

    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (int phase = 0; phase < 4; ++phase) {
            double interpolated = 0.0;
            for (int tap = 0; tap < 12; ++tap) {
                const std::int64_t source =
                    static_cast<std::int64_t>(frame) + tap - 5;
                if (source < 0 || source >= static_cast<std::int64_t>(frames)) {
                    continue;
                }
                const double sample =
                    channel == 0
                        ? buffer.left(static_cast<std::size_t>(source))
                        : buffer.right(static_cast<std::size_t>(source));
                interpolated += kTruePeakCoefficients[tap][phase] * sample;
            }
            peak = std::max(peak, std::abs(interpolated));
        }
    }
    return peak;
}

}  // namespace

LoudnessResult measureBs1770(const StereoBuffer& buffer) {
    LoudnessResult result;
    if (buffer.sampleRate != 48000 || buffer.frames == 0) {
        result.integratedLufs = -std::numeric_limits<double>::infinity();
        result.relativeGateLufs = -std::numeric_limits<double>::infinity();
        result.truePeakDbtp = -std::numeric_limits<double>::infinity();
        return result;
    }

    const double leftTruePeak = truePeakChannel(buffer, 0);
    const double rightTruePeak = truePeakChannel(buffer, 1);
    result.truePeakLinear = std::max(leftTruePeak, rightTruePeak);
    result.truePeakDbtp =
        result.truePeakLinear > 0.0
            ? 20.0 * std::log10(result.truePeakLinear)
            : -std::numeric_limits<double>::infinity();

    Biquad shelfLeft = kWeightingShelf();
    Biquad shelfRight = kWeightingShelf();
    Biquad highPassLeft = kWeightingHighPass();
    Biquad highPassRight = kWeightingHighPass();
    std::vector<double> weightedLeft(buffer.frames);
    std::vector<double> weightedRight(buffer.frames);
    for (std::size_t frame = 0; frame < buffer.frames; ++frame) {
        weightedLeft[frame] = highPassLeft.process(shelfLeft.process(buffer.left(frame)));
        weightedRight[frame] = highPassRight.process(shelfRight.process(buffer.right(frame)));
    }

    const std::size_t blockFrames =
        static_cast<std::size_t>(std::llround(0.4 * buffer.sampleRate));
    const std::size_t stepFrames =
        static_cast<std::size_t>(std::llround(0.1 * buffer.sampleRate));
    if (buffer.frames < blockFrames) {
        result.totalBlocks = 0;
        result.integratedLufs = -std::numeric_limits<double>::infinity();
        result.relativeGateLufs = -std::numeric_limits<double>::infinity();
        return result;
    }

    struct Block {
        double energy = 0.0;
        double loudness = 0.0;
    };
    std::vector<Block> blocks;
    for (std::size_t start = 0; start + blockFrames <= buffer.frames;
         start += stepFrames) {
        double energy = 0.0;
        for (std::size_t frame = start; frame < start + blockFrames; ++frame) {
            energy += weightedLeft[frame] * weightedLeft[frame] +
                      weightedRight[frame] * weightedRight[frame];
        }
        energy /= static_cast<double>(blockFrames);
        Block block;
        block.energy = energy;
        block.loudness =
            energy > 0.0 ? -0.691 + 10.0 * std::log10(energy)
                         : -std::numeric_limits<double>::infinity();
        blocks.push_back(block);
    }
    result.totalBlocks = blocks.size();

    double absoluteEnergySum = 0.0;
    std::size_t absoluteCount = 0;
    for (const Block& block : blocks) {
        if (block.loudness > -70.0) {
            absoluteEnergySum += block.energy;
            ++absoluteCount;
        }
    }
    if (absoluteCount == 0) {
        result.integratedLufs = -std::numeric_limits<double>::infinity();
        result.relativeGateLufs = -std::numeric_limits<double>::infinity();
        return result;
    }
    const double absoluteMean = absoluteEnergySum / static_cast<double>(absoluteCount);
    result.relativeGateLufs =
        -0.691 + 10.0 * std::log10(absoluteMean) - 10.0;

    double gatedEnergySum = 0.0;
    std::uint64_t gatedCount = 0;
    for (const Block& block : blocks) {
        if (block.loudness > -70.0 && block.loudness > result.relativeGateLufs) {
            gatedEnergySum += block.energy;
            ++gatedCount;
        }
    }
    if (gatedCount > 0) {
        const double gatedMean = gatedEnergySum / static_cast<double>(gatedCount);
        result.integratedLufs = -0.691 + 10.0 * std::log10(gatedMean);
        result.gatedBlocks = gatedCount;
        result.hasIntegratedValue = true;
    } else {
        result.integratedLufs = -std::numeric_limits<double>::infinity();
    }
    return result;
}

}  // namespace automix
