#include "pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

#include "audio_processor.h"
#include "json.h"
#include "lip_sync.h"
#include "phoneme_parser.h"

namespace lipsync {
namespace fs = std::filesystem;
namespace {

bool has_audio_extension(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".wav" || ext == ".ogg";
}

std::string relative_to(const fs::path& base, const fs::path& target) {
    std::error_code ec;
    fs::path rel = fs::relative(target, base, ec);
    return ec ? target.string() : rel.generic_string();
}

void assign_error(FileReport& file_report, const ProcessError& error) {
    file_report.error_code = std::string(error_code_name(error.code));
    file_report.error_message = error.message;
    if (error.code == ErrorCode::E_PHONEME_OVERFLOW) {
        file_report.status = "skipped";
    } else {
        file_report.status = "failed";
    }
}

} // namespace

ProcessError run_pipeline(const std::string& input_dir,
                          const std::string& output_dir,
                          const AppConfig& config,
                          const VisemeMapper& mapper,
                          PipelineReport& report) {
    std::error_code ec;
    if (!fs::exists(input_dir, ec) || !fs::is_directory(input_dir, ec)) {
        return ProcessError(ErrorCode::E_IO, "input directory not found: " + input_dir);
    }
    fs::create_directories(output_dir, ec);
    fs::path audio_out_dir = fs::path(output_dir) / "audio";
    fs::path curve_out_dir = fs::path(output_dir) / "lipsync";
    fs::create_directories(audio_out_dir, ec);
    fs::create_directories(curve_out_dir, ec);

    std::vector<fs::path> inputs;
    for (const fs::directory_entry& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file() && has_audio_extension(entry.path())) {
            inputs.push_back(entry.path());
        }
    }
    std::sort(inputs.begin(), inputs.end());

    const size_t worker_count =
        static_cast<size_t>(std::max(1, std::min(config.max_workers,
                                                 static_cast<int>(inputs.size()))));

    std::atomic<size_t> next_index{0};
    std::vector<FileReport> reports(inputs.size());
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    // 测试探针：LIPSYNC_PARALLEL_PROBE=1 时记录实际同时在处理文件的线程数
    // 峰值，写入 LIPSYNC_PARALLEL_MAX 指向的文件。生产环境不设置即零开销。
    const bool parallel_probe =
        std::getenv("LIPSYNC_PARALLEL_PROBE") != nullptr;
    std::atomic<int> active_workers{0};
    std::atomic<int> peak_workers{0};

    auto worker = [&]() {
        while (true) {
            size_t index = next_index.fetch_add(1);
            if (index >= inputs.size()) return;
            const fs::path& audio_path = inputs[index];
            FileReport& file_report = reports[index];

            if (parallel_probe) {
                int active = active_workers.fetch_add(1) + 1;
                int peak = peak_workers.load();
                while (active > peak &&
                       !peak_workers.compare_exchange_weak(peak, active)) {
                }
            }

            const auto started = std::chrono::steady_clock::now();
            file_report.input_path =
                relative_to(fs::path(output_dir), audio_path);

            fs::path sidecar = audio_path;
            sidecar.replace_extension(".json");

            fs::path out_wav =
                audio_out_dir / fs::path(audio_path.stem().string() + ".wav");
            fs::path out_curve =
                curve_out_dir / fs::path(audio_path.stem().string() + "_lipsync.json");

            // 1) 音素侧车文件先解析，损坏 JSON 直接归类为可恢复失败。
            PhonemeTrack track;
            ProcessError error = parse_phoneme_file(sidecar.string(), track);
            if (error) {
                assign_error(file_report, error);
            }

            // 2) 音频解码与重采样始终尝试，保证其他产物尽量完整。
            AudioResult audio_result;
            if (!error) {
                error = process_audio(audio_path.string(), out_wav.string(),
                                      config.target_sample_rate,
                                      config.target_channels,
                                      audio_result);
            }
            if (error) {
                assign_error(file_report, error);
            } else {
                file_report.output_audio_path =
                    relative_to(fs::path(output_dir), out_wav);
                file_report.source_duration_ms =
                    std::llround(audio_result.source_duration_seconds * 1000.0);
            }

            // 3) 时长校验：越界超过容差则跳过唇形生成（音频产物保留）。
            if (!error) {
                error = validate_phoneme_track(track,
                                               audio_result.source_duration_seconds,
                                               config.overflow_tolerance_ms);
                if (error) {
                    assign_error(file_report, error);
                }
            }

            // 4) 生成口型曲线。
            if (!error) {
                LipSyncResult lip_result;
                error = generate_lip_sync(track,
                                          audio_result.source_duration_seconds,
                                          config.fps, mapper,
                                          out_curve.string(), lip_result);
                if (error) {
                    assign_error(file_report, error);
                } else {
                    file_report.status = "success";
                    file_report.error_code = "OK";
                    file_report.frame_count = lip_result.frame_count;
                    file_report.output_curve_path =
                        relative_to(fs::path(output_dir), out_curve);
                    file_report.warnings = std::move(lip_result.warnings);
                }
            }

            const auto ended = std::chrono::steady_clock::now();
            file_report.elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    ended - started).count();
            if (parallel_probe) {
                active_workers.fetch_sub(1);
            }
        }
    };

    for (size_t i = 0; i < worker_count; ++i) {
        workers.emplace_back(worker);
    }
    for (std::thread& thread : workers) {
        thread.join();
    }

    if (parallel_probe) {
        if (const char* probe_path = std::getenv("LIPSYNC_PARALLEL_MAX")) {
            std::ofstream probe_out(probe_path, std::ios::trunc);
            probe_out << peak_workers.load();
        }
    }

    report.files = std::move(reports);
    report.total = static_cast<int>(report.files.size());
    for (const FileReport& file_report : report.files) {
        if (file_report.status == "success") ++report.success_count;
        else if (file_report.status == "skipped") ++report.skipped_count;
        else ++report.failed_count;
    }

    fs::path report_path = fs::path(output_dir) / "report.json";
    std::string io_error;
    if (!write_text_file(report_path.string(), render_report_json(report, config),
                         &io_error)) {
        return ProcessError(ErrorCode::E_IO, io_error);
    }
    return ProcessError{};
}

std::string render_report_json(const PipelineReport& report, const AppConfig& config) {
    JsonValue root;
    root.type = JsonValue::Type::Object;

    auto add_number = [&](const char* key, double value) {
        JsonValue v;
        v.type = JsonValue::Type::Number;
        v.number = value;
        root.object.emplace(key, std::move(v));
    };

    add_number("target_sample_rate", config.target_sample_rate);
    add_number("target_channels", config.target_channels);
    add_number("fps", config.fps);
    add_number("max_workers", config.max_workers);
    add_number("total", report.total);
    add_number("success", report.success_count);
    add_number("skipped", report.skipped_count);
    add_number("failed", report.failed_count);

    JsonValue files_array;
    files_array.type = JsonValue::Type::Array;
    for (const FileReport& file_report : report.files) {
        JsonValue item;
        item.type = JsonValue::Type::Object;

        auto put_string = [&](const char* key, const std::string& value) {
            JsonValue v;
            v.type = JsonValue::Type::String;
            v.str = value;
            item.object.emplace(key, std::move(v));
        };
        auto put_number = [&](const char* key, double value) {
            JsonValue v;
            v.type = JsonValue::Type::Number;
            v.number = value;
            item.object.emplace(key, std::move(v));
        };

        put_string("input_path", file_report.input_path);
        put_string("output_audio_path", file_report.output_audio_path);
        put_string("output_curve_path", file_report.output_curve_path);
        put_string("status", file_report.status);
        put_string("error_code", file_report.error_code);
        put_string("error_message", file_report.error_message);
        put_number("elapsed_ms", static_cast<double>(file_report.elapsed_ms));
        put_number("frame_count", file_report.frame_count);
        put_number("source_duration_ms", file_report.source_duration_ms);

        JsonValue warnings;
        warnings.type = JsonValue::Type::Array;
        for (const MappingWarning& warning : file_report.warnings) {
            JsonValue warning_object;
            warning_object.type = JsonValue::Type::Object;
            JsonValue language;
            language.type = JsonValue::Type::String;
            language.str = warning.language;
            warning_object.object.emplace("language", std::move(language));
            JsonValue phoneme;
            phoneme.type = JsonValue::Type::String;
            phoneme.str = warning.phoneme;
            warning_object.object.emplace("phoneme", std::move(phoneme));
            JsonValue message;
            message.type = JsonValue::Type::String;
            message.str = warning.message;
            warning_object.object.emplace("message", std::move(message));
            warnings.array.push_back(std::move(warning_object));
        }
        item.object.emplace("warnings", std::move(warnings));
        files_array.array.push_back(std::move(item));
    }
    root.object.emplace("files", std::move(files_array));
    return json_dump(root);
}

} // namespace lipsync
