#pragma once

#include <string>
#include <vector>

#include "error_codes.h"

namespace lipsync {

struct AudioResult {
    std::string output_wav_path;
    double source_duration_seconds = 0.0; // 原始音频时长（毫秒精度）
    int source_sample_rate = 0;
    int source_channels = 0;
    int target_sample_rate = 0;
    int target_channels = 0;
};

// 解码 WAV/OGG 输入，重采样并转换声道，输出严格匹配配置的 PCM16 WAV。
// 原始时长按输入采样率下的总样本数计算，外部四舍五入到毫秒。
ProcessError process_audio(const std::string& input_path,
                           const std::string& output_wav_path,
                           int target_sample_rate,
                           int target_channels,
                           AudioResult& result);

// 读取 WAV 头以校验输出格式（测试用）。
struct WavInfo {
    int sample_rate = 0;
    int channels = 0;
    int bits_per_sample = 0;
    size_t data_bytes = 0;
};
ProcessError read_wav_info(const std::string& wav_path, WavInfo& info);

} // namespace lipsync
