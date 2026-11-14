#pragma once

#include <string>
#include <vector>

#include "config.h"
#include "error_codes.h"
#include "viseme_mapper.h"

namespace lipsync {

struct FileReport {
    std::string input_path;          // 相对输出目录（报告要求）
    std::string output_audio_path;   // 相对输出目录，空表示无产物
    std::string output_curve_path;   // 相对输出目录，空表示无产物
    std::string status;              // success / skipped / failed
    std::string error_code;          // OK / E_PHONEME_OVERFLOW / ...
    std::string error_message;
    long long elapsed_ms = 0;        // 单调时钟耗时
    int frame_count = 0;
    double source_duration_ms = 0.0;
    std::vector<MappingWarning> warnings;
};

struct PipelineReport {
    std::vector<FileReport> files;
    int total = 0;
    int success_count = 0;
    int skipped_count = 0;
    int failed_count = 0;
};

// 扫描输入目录中的音频文件，按配置上限并行处理，单个文件失败不影响其他文件。
// 报告写入 <output_dir>/report.json，内部路径均相对输出目录。
ProcessError run_pipeline(const std::string& input_dir,
                          const std::string& output_dir,
                          const AppConfig& config,
                          const VisemeMapper& mapper,
                          PipelineReport& report);

// 将报告序列化为 JSON 文本。
std::string render_report_json(const PipelineReport& report, const AppConfig& config);

} // namespace lipsync
