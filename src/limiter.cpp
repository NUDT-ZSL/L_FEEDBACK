#include "automix/limiter.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <vector>

namespace automix {

LimiterStats processLimiter(StereoBuffer& buffer, const LimiterConfig& config) {
    LimiterStats stats;
    if (!config.enabled || buffer.frames == 0) {
        stats.outputPeakDbfs = measureLevels(buffer).peakDbfs;
        return stats;
    }

    const double sampleRate = static_cast<double>(buffer.sampleRate);
    std::uint64_t lookahead = static_cast<std::uint64_t>(
        std::llround(config.lookaheadMs * sampleRate / 1000.0));
    lookahead = std::max<std::uint64_t>(lookahead, 1);
    stats.lookaheadSamples = lookahead;

    const double threshold = dbToLinear(config.thresholdDbfs);
    const double safeThreshold = threshold * (1.0 - 1e-9);
    std::vector<double> target(buffer.frames, 1.0);
    for (std::size_t frame = 0; frame < buffer.frames; ++frame) {
        const double peak = std::max(
            std::abs(static_cast<double>(buffer.left(frame))),
            std::abs(static_cast<double>(buffer.right(frame))));
        if (peak > safeThreshold) {
            target[frame] = safeThreshold / peak;
            ++stats.triggeredSamples;
        }
    }

    std::vector<double> envelope(buffer.frames, 1.0);
    std::deque<std::size_t> minimumQueue;
    const std::size_t initialEnd =
        std::min<std::size_t>(static_cast<std::size_t>(lookahead), buffer.frames - 1);
    for (std::size_t index = 0; index <= initialEnd; ++index) {
        while (!minimumQueue.empty() && target[minimumQueue.back()] >= target[index]) {
            minimumQueue.pop_back();
        }
        minimumQueue.push_back(index);
    }
    for (std::size_t frame = 0; frame < buffer.frames; ++frame) {
        if (frame > 0 && frame + lookahead < buffer.frames) {
            const std::size_t index = frame + static_cast<std::size_t>(lookahead);
            while (!minimumQueue.empty() && target[minimumQueue.back()] >= target[index]) {
                minimumQueue.pop_back();
            }
            minimumQueue.push_back(index);
        }
        while (!minimumQueue.empty() && minimumQueue.front() < frame) {
            minimumQueue.pop_front();
        }
        envelope[frame] = target[minimumQueue.front()];
    }

    const double releaseSeconds = std::max(config.releaseMs, 0.001) / 1000.0;
    const double releaseCoefficient = std::exp(-1.0 / (releaseSeconds * sampleRate));
    double gain = 1.0;
    double minimumGain = 1.0;
    for (std::size_t frame = 0; frame < buffer.frames; ++frame) {
        const double released = releaseCoefficient * gain + (1.0 - releaseCoefficient);
        gain = std::min(envelope[frame], released);
        gain = std::clamp(gain, 0.0, 1.0);
        minimumGain = std::min(minimumGain, gain);
        if (gain < 1.0 - 1e-12) {
            ++stats.affectedSamples;
        }

        double left = static_cast<double>(buffer.left(frame)) * gain;
        double right = static_cast<double>(buffer.right(frame)) * gain;
        left = std::clamp(left, -safeThreshold, safeThreshold);
        right = std::clamp(right, -safeThreshold, safeThreshold);
        buffer.left(frame) = static_cast<float>(left);
        buffer.right(frame) = static_cast<float>(right);
    }

    stats.maxAttenuationDb =
        minimumGain > 0.0 ? -20.0 * std::log10(minimumGain)
                          : std::numeric_limits<double>::infinity();
    stats.outputPeakDbfs = measureLevels(buffer).peakDbfs;
    return stats;
}

}  // namespace automix
