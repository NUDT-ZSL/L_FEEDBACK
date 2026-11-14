#pragma once

#include <string>

#include "error_codes.h"
#include "types.h"
#include "viseme_mapper.h"

namespace lipsync {

struct LipSyncResult {
    std::string output_curve_path;
    int frame_count = 0;
    std::vector<MappingWarning> warnings;
};

// 依据音素时间戳生成 fps 帧率的 viseme 权重曲线并序列化为 JSON。
// 帧 i 覆盖 [i/fps, (i+1)/fps)；按与音素区间的重叠比例分配权重，
// 未被任何音素覆盖的比例计入静默 viseme，最终权重和严格归一化为 1。
// 曲线从 0 时刻开始，帧数覆盖至音频结束（ceil(duration * fps)）。
ProcessError generate_lip_sync(const PhonemeTrack& track,
                               double audio_duration_seconds,
                               double fps,
                               const VisemeMapper& mapper,
                               const std::string& output_curve_path,
                               LipSyncResult& result);

} // namespace lipsync
