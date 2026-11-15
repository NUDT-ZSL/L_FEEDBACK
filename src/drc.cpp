#include "automix/drc.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace automix {

double drcStaticReductionDb(double levelDbfs, const DrcConfig& config) {
    if (!config.enabled || config.ratio <= 1.0) {
        return 0.0;
    }
    const double threshold = config.thresholdDb;
    const double slope = 1.0 - 1.0 / config.ratio;
    if (config.kneeDb <= 0.0) {
        return levelDbfs <= threshold ? 0.0 : slope * (levelDbfs - threshold);
    }
    const double lower = threshold - config.kneeDb * 0.5;
    const double upper = threshold + config.kneeDb * 0.5;
    if (levelDbfs <= lower) {
        return 0.0;
    }
    if (levelDbfs >= upper) {
        return slope * (levelDbfs - threshold);
    }
    const double position = levelDbfs - lower;
    return slope * position * position / (2.0 * config.kneeDb);
}

DrcStats processDrc(StereoBuffer& buffer,
                    const DrcConfig& config,
                    std::vector<double>* gainReductionDb) {
    DrcStats stats;
    if (gainReductionDb != nullptr) {
        gainReductionDb->assign(buffer.frames, 0.0);
    }
    if (!config.enabled || config.ratio <= 1.0 || buffer.frames == 0) {
        return stats;
    }

    const double sampleRate = static_cast<double>(buffer.sampleRate);
    const double attackSeconds = std::max(config.attackMs, 0.001) / 1000.0;
    const double releaseSeconds = std::max(config.releaseMs, 0.001) / 1000.0;
    const double attackCoefficient = std::exp(-1.0 / (attackSeconds * sampleRate));
    const double releaseCoefficient = std::exp(-1.0 / (releaseSeconds * sampleRate));

    double gainReduction = 0.0;
    double totalReduction = 0.0;
    for (std::size_t frame = 0; frame < buffer.frames; ++frame) {
        const double peak = std::max(
            std::abs(static_cast<double>(buffer.left(frame))),
            std::abs(static_cast<double>(buffer.right(frame))));
        const double levelDbfs =
            peak > 0.0 ? 20.0 * std::log10(peak)
                       : -std::numeric_limits<double>::infinity();
        const double target = drcStaticReductionDb(levelDbfs, config);
        const double coefficient =
            target > gainReduction ? attackCoefficient : releaseCoefficient;
        gainReduction = coefficient * gainReduction + (1.0 - coefficient) * target;
        if (gainReduction < 1e-12) {
            gainReduction = 0.0;
        }

        const double gain = std::pow(10.0, -gainReduction / 20.0);
        buffer.left(frame) = static_cast<float>(buffer.left(frame) * gain);
        buffer.right(frame) = static_cast<float>(buffer.right(frame) * gain);

        if (gainReductionDb != nullptr) {
            (*gainReductionDb)[frame] = gainReduction;
        }
        stats.maxReductionDb = std::max(stats.maxReductionDb, gainReduction);
        totalReduction += gainReduction;
        if (gainReduction > 1e-9) {
            ++stats.activeSamples;
        }
    }
    stats.meanReductionDb =
        totalReduction / static_cast<double>(buffer.frames);
    return stats;
}

}  // namespace automix
