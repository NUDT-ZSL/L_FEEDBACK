#pragma once

#include <string>
#include <vector>

namespace lipsync {

// 音素时间区间（秒），phoneme 为该语言音素集中的音素符号。
struct PhonemeInterval {
    std::string phoneme;
    double start = 0.0;
    double end = 0.0;
};

struct PhonemeTrack {
    std::string language;
    std::vector<PhonemeInterval> intervals;
};

// 每帧的一组 viseme 权重，下标与 viseme 名称列表一一对应。
struct VisemeFrame {
    std::vector<float> weights;
};

struct VisemeCurve {
    std::vector<std::string> viseme_names;
    double fps = 30.0;
    std::vector<VisemeFrame> frames;
};

} // namespace lipsync
