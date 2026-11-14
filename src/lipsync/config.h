#pragma once

#include <string>
#include <vector>

#include "error_codes.h"

namespace lipsync {

struct AppConfig {
    int target_sample_rate = 48000; // Hz
    int target_channels = 1;        // 1=mono, 2=stereo
    double fps = 30.0;              // 口型曲线帧率
    int max_workers = 2;            // 并行文件数上限
    std::vector<std::string> languages;
    std::string viseme_map_path;    // 可选：外部音素->viseme 映射 JSON
    double overflow_tolerance_ms = 100.0;
};

// 解析配置 JSON。字段非法时返回 E_CONFIG。
ProcessError load_config(const std::string& config_path, AppConfig& config);

} // namespace lipsync
