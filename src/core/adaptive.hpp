#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace uid166 {

struct AdaptiveConfig {
    bool enabled = true;
    float targetFrameMs = 1000.0f / 60.0f;
    float degradeThresholdMs = 18.2f;
    float restoreThresholdMs = 13.6f;
    uint32_t degradeFrames = 5;
    uint32_t restoreFrames = 12;
    uint32_t cooldownFrames = 8;
    uint32_t historyLimit = 120;
    std::array<float, 6> renderScales{1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
    std::array<float, 6> shadowScales{1.0f, 0.85f, 0.72f, 0.6f, 0.5f, 0.42f};
};

enum class QualityAction { none, degrade, restore };

struct QualityState {
    uint32_t level = 0;
    float renderScale = 1.0f;
    float shadowScale = 1.0f;
    float smoothedFrameMs = 0.0f;
    QualityAction action = QualityAction::none;
};

struct AdaptiveHistoryEntry {
    uint64_t frame = 0;
    float gpuFrameMs = 0.0f;
    float smoothedFrameMs = 0.0f;
    uint32_t level = 0;
    float renderScale = 1.0f;
    float shadowScale = 1.0f;
    QualityAction action = QualityAction::none;
    std::string reason;
};

class AdaptiveQuality {
public:
    explicit AdaptiveQuality(AdaptiveConfig config = {});

    [[nodiscard]] QualityState update(uint64_t frame, float gpuFrameMs);
    void setEnabled(bool enabled) { config_.enabled = enabled; }
    [[nodiscard]] const std::vector<AdaptiveHistoryEntry>& history() const { return history_; }
    [[nodiscard]] const AdaptiveConfig& config() const { return config_; }
    [[nodiscard]] const QualityState& state() const { return state_; }

private:
    void applyLevel(uint32_t level, QualityAction action, uint64_t frame, float gpuFrameMs,
                    const char* reason);

    AdaptiveConfig config_;
    QualityState state_{};
    uint32_t highCount_ = 0;
    uint32_t lowCount_ = 0;
    uint32_t cooldown_ = 0;
    std::vector<AdaptiveHistoryEntry> history_;
};

[[nodiscard]] const char* qualityActionName(QualityAction action);

} // namespace uid166

