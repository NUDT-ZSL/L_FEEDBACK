#include "core/adaptive.hpp"

#include <algorithm>
#include <cmath>

namespace uid166 {

const char* qualityActionName(QualityAction action) {
    switch (action) {
    case QualityAction::none: return "none";
    case QualityAction::degrade: return "degrade";
    case QualityAction::restore: return "restore";
    }
    return "unknown";
}

AdaptiveQuality::AdaptiveQuality(AdaptiveConfig config) : config_(config) {
    if (config_.renderScales.size() != config_.shadowScales.size() ||
        config_.renderScales.empty() || config_.historyLimit == 0) {
        config_.renderScales = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
        config_.shadowScales = {1.0f, 0.85f, 0.72f, 0.6f, 0.5f, 0.42f};
        config_.historyLimit = 120;
    }
    state_.renderScale = config_.renderScales[0];
    state_.shadowScale = config_.shadowScales[0];
    history_.reserve(config_.historyLimit);
}

QualityState AdaptiveQuality::update(uint64_t frame, float gpuFrameMs) {
    const float sample = std::clamp(std::isfinite(gpuFrameMs) ? gpuFrameMs : config_.targetFrameMs,
                                    0.05f, 1000.0f);
    state_.smoothedFrameMs = state_.smoothedFrameMs <= 0.0f
        ? sample
        : state_.smoothedFrameMs * 0.90f + sample * 0.10f;
    state_.action = QualityAction::none;
    const char* reason = "within hysteresis band";
    if (!config_.enabled) {
        state_.level = 0;
        state_.renderScale = config_.renderScales[0];
        state_.shadowScale = config_.shadowScales[0];
        reason = "adaptive quality disabled";
        highCount_ = 0;
        lowCount_ = 0;
    } else {
        if (cooldown_ != 0) --cooldown_;

        if (state_.smoothedFrameMs > config_.degradeThresholdMs) {
            ++highCount_;
            lowCount_ = 0;
        } else if (state_.smoothedFrameMs < config_.restoreThresholdMs) {
            ++lowCount_;
            highCount_ = 0;
        } else {
            highCount_ = 0;
            lowCount_ = 0;
        }

        const uint32_t maxLevel = static_cast<uint32_t>(config_.renderScales.size() - 1);
        if (cooldown_ == 0 && highCount_ >= config_.degradeFrames &&
            state_.level < maxLevel) {
            applyLevel(state_.level + 1, QualityAction::degrade, frame, sample,
                       "smoothed GPU frame time above high threshold");
            highCount_ = 0;
            lowCount_ = 0;
            cooldown_ = config_.cooldownFrames;
            reason = "smoothed GPU frame time above high threshold";
        } else if (cooldown_ == 0 && lowCount_ >= config_.restoreFrames &&
                   state_.level > 0) {
            applyLevel(state_.level - 1, QualityAction::restore, frame, sample,
                       "smoothed GPU frame time below low threshold");
            highCount_ = 0;
            lowCount_ = 0;
            cooldown_ = config_.cooldownFrames;
            reason = "smoothed GPU frame time below low threshold";
        } else if (state_.level == 0 && (history_.empty() ||
                   history_.back().level != 0 || history_.back().action != QualityAction::none)) {
            reason = "default quality active";
        }
    }
    if (history_.size() == config_.historyLimit) {
        history_.erase(history_.begin());
    }
    history_.push_back({frame, sample, state_.smoothedFrameMs, state_.level,
                        state_.renderScale, state_.shadowScale, state_.action, reason});
    return state_;
}

void AdaptiveQuality::applyLevel(uint32_t level, QualityAction action, uint64_t frame,
                                 float gpuFrameMs, const char* reason) {
    const uint32_t maxLevel = static_cast<uint32_t>(config_.renderScales.size() - 1);
    level = std::min(level, maxLevel);
    state_.level = level;
    state_.renderScale = config_.renderScales[level];
    state_.shadowScale = config_.shadowScales[level];
    state_.action = action;
    (void)frame;
    (void)gpuFrameMs;
    (void)reason;
}

} // namespace uid166

