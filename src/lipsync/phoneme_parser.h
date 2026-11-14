#pragma once

#include <string>

#include "error_codes.h"
#include "types.h"

namespace lipsync {

// 解析侧车 JSON：
// { "language": "en",
//   "phonemes": [ {"phoneme":"AA","start":0.0,"end":0.12}, ... ] }
// 文件损坏/结构非法返回 E_PARSE_JSON。
ProcessError parse_phoneme_file(const std::string& json_path, PhonemeTrack& track);

// 校验区间合法性：start<=end、不重叠、不得超出音频时长超过 tolerance_ms。
// 超出时返回 E_PHONEME_OVERFLOW（可恢复：跳过唇形生成但保留音频产物）。
ProcessError validate_phoneme_track(const PhonemeTrack& track,
                                    double audio_duration_seconds,
                                    double tolerance_ms);

} // namespace lipsync
