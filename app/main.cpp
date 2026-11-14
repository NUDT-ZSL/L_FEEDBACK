#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "lipsync/config.h"
#include "lipsync/error_codes.h"
#include "lipsync/pipeline.h"
#include "lipsync/viseme_mapper.h"

namespace {

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " --input <dir> --output <dir> --config <json>\n"
              << "  --input   directory containing .wav/.ogg audio plus sidecar .json\n"
              << "  --output  directory for resampled audio, lip curves and report.json\n"
              << "  --config  JSON config: target_sample_rate, target_channels, fps,\n"
              << "            languages, max_workers, optional viseme_map_path\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string input_dir;
    std::string output_dir;
    std::string config_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_value = [&](std::string& target) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << arg << "\n";
                return false;
            }
            target = argv[++i];
            return true;
        };
        if (arg == "--input") {
            if (!next_value(input_dir)) return static_cast<int>(lipsync::ErrorCode::E_CONFIG);
        } else if (arg == "--output") {
            if (!next_value(output_dir)) return static_cast<int>(lipsync::ErrorCode::E_CONFIG);
        } else if (arg == "--config") {
            if (!next_value(config_path)) return static_cast<int>(lipsync::ErrorCode::E_CONFIG);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return static_cast<int>(lipsync::ErrorCode::E_CONFIG);
        }
    }

    if (input_dir.empty() || output_dir.empty() || config_path.empty()) {
        print_usage(argv[0]);
        return static_cast<int>(lipsync::ErrorCode::E_CONFIG);
    }

    try {
        lipsync::AppConfig config;
        if (lipsync::ProcessError error = lipsync::load_config(config_path, config)) {
            std::cerr << "config error [" << lipsync::error_code_name(error.code)
                      << "]: " << error.message << "\n";
            return static_cast<int>(error.code);
        }

        lipsync::VisemeMapper mapper;
        if (lipsync::ProcessError error =
                lipsync::VisemeMapper::load(config.viseme_map_path, mapper)) {
            std::cerr << "viseme map error [" << lipsync::error_code_name(error.code)
                      << "]: " << error.message << "\n";
            return static_cast<int>(error.code);
        }

        lipsync::PipelineReport report;
        if (lipsync::ProcessError error =
                lipsync::run_pipeline(input_dir, output_dir, config, mapper, report)) {
            std::cerr << "pipeline error [" << lipsync::error_code_name(error.code)
                      << "]: " << error.message << "\n";
            return static_cast<int>(error.code);
        }

        std::cout << "processed " << report.total
                  << " file(s): success=" << report.success_count
                  << " skipped=" << report.skipped_count
                  << " failed=" << report.failed_count << "\n";
        // 单文件失败是可恢复的业务结果；仅当整体管线无法运行时才返回非 0。
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "fatal error: " << ex.what() << "\n";
        return static_cast<int>(lipsync::ErrorCode::E_INTERNAL);
    }
}
