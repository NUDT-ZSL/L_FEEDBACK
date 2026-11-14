#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#else
#include <process.h>
#endif

#include "lipsync/audio_processor.h"
#include "lipsync/config.h"
#include "lipsync/json.h"
#include "lipsync/pipeline.h"
#include "lipsync/viseme_mapper.h"

namespace fs = std::filesystem;
using namespace lipsync;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const std::string& label) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::cerr << "FAIL: " << label << "\n";
    }
}

void write_silence_wav(const fs::path& path) {
    const int sample_rate = 8000;
    const int frames = sample_rate; // 1 秒静音 PCM16
    std::ofstream out(path, std::ios::binary);
    auto u32 = [&](uint32_t v) {
        char b[4] = {char(v), char(v >> 8), char(v >> 16), char(v >> 24)};
        out.write(b, 4);
    };
    auto u16 = [&](uint16_t v) {
        char b[2] = {char(v), char(v >> 8)};
        out.write(b, 2);
    };
    const uint32_t data_bytes = frames * 2;
    out.write("RIFF", 4);
    u32(36 + data_bytes);
    out.write("WAVEfmt ", 8);
    u32(16);
    u16(1);
    u16(1);
    u32(sample_rate);
    u32(sample_rate * 2);
    u16(2);
    u16(16);
    out.write("data", 4);
    u32(data_bytes);
    std::string zeros(data_bytes, '\0');
    out.write(zeros.data(), data_bytes);
}

} // namespace

// 有界并行验证：6 个文件、max_workers=2。
// 处理逻辑内部用固定大小线程池 + 原子索引派发任务，此处通过重复运行管线
// 并核对全部成功来验证上限配置被严格遵守（无死锁、无任务丢失）。
int main() {
#if defined(_WIN32)
    const int pid = _getpid();
#else
    const int pid = ::getpid();
#endif
    fs::path root = fs::temp_directory_path() /
                    ("lipsync_concurrency_" + std::to_string(pid));
    fs::path input = root / "input";
    fs::path output = root / "output";
    fs::remove_all(root);
    fs::create_directories(input);

    constexpr int kFileCount = 6;
    for (int i = 0; i < kFileCount; ++i) {
        std::string base = "line_" + std::to_string(i);
        write_silence_wav(input / (base + ".wav"));
        JsonValue root_json;
        root_json.type = JsonValue::Type::Object;
        JsonValue language;
        language.type = JsonValue::Type::String;
        language.str = "en";
        root_json.object.emplace("language", std::move(language));
        JsonValue phonemes;
        phonemes.type = JsonValue::Type::Array;
        JsonValue interval;
        interval.type = JsonValue::Type::Object;
        for (const char* key : {"phoneme"}) {
            JsonValue v;
            v.type = JsonValue::Type::String;
            v.str = "AA";
            interval.object.emplace(key, std::move(v));
        }
        for (auto kv : std::vector<std::pair<const char*, double>>{
                 {"start", 0.05}, {"end", 0.9}}) {
            JsonValue v;
            v.type = JsonValue::Type::Number;
            v.number = kv.second;
            interval.object.emplace(kv.first, std::move(v));
        }
        phonemes.array.push_back(std::move(interval));
        root_json.object.emplace("phonemes", std::move(phonemes));
        std::string error;
        check(write_text_file((input / (base + ".json")).string(),
                              json_dump(root_json), &error),
              "sidecar written: " + base);
    }

    AppConfig config;
    config.target_sample_rate = 16000;
    config.target_channels = 1;
    config.fps = 30.0;
    config.max_workers = 2;
    config.languages = {"en", "ja", "es"};

    VisemeMapper mapper;
    check(!VisemeMapper::load("", mapper), "mapper loads");

    fs::path peak_file = root / "peak.txt";
#if defined(_WIN32)
    _putenv_s("LIPSYNC_PARALLEL_PROBE", "1");
    _putenv_s("LIPSYNC_PARALLEL_MAX", peak_file.string().c_str());
#else
    setenv("LIPSYNC_PARALLEL_PROBE", "1", 1);
    setenv("LIPSYNC_PARALLEL_MAX", peak_file.string().c_str(), 1);
#endif

    PipelineReport report;
    check(!run_pipeline(input.string(), output.string(), config, mapper, report),
          "bounded pipeline runs to completion");
    check(report.total == kFileCount, "all files scheduled");
    check(report.success_count == kFileCount, "all files succeed with max_workers=2");
    check(report.failed_count == 0 && report.skipped_count == 0,
          "no failures under bounded parallelism");

    std::ifstream peak_in(peak_file);
    int peak = -1;
    peak_in >> peak;
    check(peak == 2, "observed peak concurrency equals max_workers (2)");

    // 串行配置 max_workers=1 必须同样完整处理。
    fs::path serial_output = root / "serial_output";
    config.max_workers = 1;
    PipelineReport serial_report;
    check(!run_pipeline(input.string(), serial_output.string(), config, mapper,
                        serial_report),
          "serial pipeline runs to completion");
    check(serial_report.success_count == kFileCount,
          "all files succeed with max_workers=1");

    fs::remove_all(root);
    std::cout << "concurrency checks: " << g_checks
              << ", failures: " << g_failures << "\n";
    return g_failures == 0 ? 0 : 1;
}
