#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
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

void put_u32(std::vector<uint8_t>& buf, uint32_t v) {
    for (int shift = 0; shift < 32; shift += 8) buf.push_back((v >> shift) & 0xFF);
}

void put_u16(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(v & 0xFF);
    buf.push_back((v >> 8) & 0xFF);
}

// 生成确定性的 16kHz 单声道 PCM16 WAV（2 秒）。
void write_test_wav(const fs::path& path, int sample_rate, double seconds) {
    const int total = static_cast<int>(sample_rate * seconds);
    std::vector<uint8_t> pcm;
    pcm.reserve(static_cast<size_t>(total) * 2);
    for (int i = 0; i < total; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        int16_t sample = static_cast<int16_t>(
            12000.0 * std::sin(2.0 * 3.14159265358979323846 * 220.0 * t));
        pcm.push_back(static_cast<uint8_t>(sample & 0xFF));
        pcm.push_back(static_cast<uint8_t>((sample >> 8) & 0xFF));
    }

    std::vector<uint8_t> header;
    header.insert(header.end(), {'R', 'I', 'F', 'F'});
    put_u32(header, 36u + static_cast<uint32_t>(pcm.size()));
    header.insert(header.end(), {'W', 'A', 'V', 'E'});
    header.insert(header.end(), {'f', 'm', 't', ' '});
    put_u32(header, 16);
    put_u16(header, 1);
    put_u16(header, 1);
    put_u32(header, sample_rate);
    put_u32(header, static_cast<uint32_t>(sample_rate * 2));
    put_u16(header, 2);
    put_u16(header, 16);
    header.insert(header.end(), {'d', 'a', 't', 'a'});
    put_u32(header, static_cast<uint32_t>(pcm.size()));

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(header.data()),
              static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(pcm.data()),
              static_cast<std::streamsize>(pcm.size()));
}

void write_text(const fs::path& path, const std::string& content) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << content;
}

std::string phoneme_doc(const std::string& language,
                        const std::vector<std::tuple<std::string, double, double>>& items) {
    std::string text = "{\"language\":\"" + language + "\",\"phonemes\":[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i) text += ",";
        text += "{\"phoneme\":\"" + std::get<0>(items[i]) + "\",\"start\":" +
                std::to_string(std::get<1>(items[i])) + ",\"end\":" +
                std::to_string(std::get<2>(items[i])) + "}";
    }
    text += "]}";
    return text;
}

} // namespace

int main() {
#if defined(_WIN32)
    const int pid = _getpid();
#else
    const int pid = ::getpid();
#endif
    fs::path root = fs::temp_directory_path() /
                    ("lipsync_integration_" + std::to_string(pid));
    fs::path input_dir = root / "input";
    fs::path output_dir = root / "output";
    fs::remove_all(root);
    fs::create_directories(input_dir);

    // 三个语言、五种输入场景，全部为 16kHz/2s WAV（目标转换为 48kHz mono）。
    write_test_wav(input_dir / "en_good.wav", 16000, 2.0);
    write_text(input_dir / "en_good.json",
               phoneme_doc("en", {{"HH", 0.05, 0.3}, {"AA", 0.3, 0.8},
                                  {"SIL", 0.8, 1.0}, {"D", 1.0, 1.7}}));

    write_test_wav(input_dir / "ja_good.wav", 16000, 2.0);
    write_text(input_dir / "ja_good.json",
               phoneme_doc("ja", {{"K", 0.1, 0.4}, {"O", 0.4, 0.9},
                                  {"N", 0.9, 1.3}, {"I", 1.3, 1.85}}));

    write_test_wav(input_dir / "es_good.wav", 16000, 2.0);
    write_text(input_dir / "es_good.json",
               phoneme_doc("es", {{"B", 0.1, 0.35}, {"U", 0.35, 0.7},
                                  {"E", 0.7, 1.2}, {"S", 1.2, 1.9}}));

    write_test_wav(input_dir / "es_overflow.wav", 16000, 2.0);
    write_text(input_dir / "es_overflow.json",
               phoneme_doc("es", {{"K", 0.1, 0.4}, {"A", 0.4, 2.6}}));

    write_test_wav(input_dir / "ja_broken.wav", 16000, 2.0);
    write_text(input_dir / "ja_broken.json",
               "{\"language\":\"ja\", \"phonemes\": [ {\"phoneme\":\"A\",}, ]}");

    AppConfig config;
    config.target_sample_rate = 48000;
    config.target_channels = 1;
    config.fps = 30.0;
    config.max_workers = 2; // 验证有界并行：5 个文件、2 个槽位
    config.languages = {"en", "ja", "es"};
    config.overflow_tolerance_ms = 100.0;

    VisemeMapper mapper;
    check(!VisemeMapper::load("", mapper), "mapper initializes");

    PipelineReport report;
    ProcessError pipeline_error =
        run_pipeline(input_dir.string(), output_dir.string(), config, mapper, report);
    check(!pipeline_error, "pipeline completes despite per-file errors");

    check(report.total == 5, "all 5 files were scheduled");
    check(report.success_count == 3, "three files succeed");
    check(report.skipped_count == 1, "one file is skipped");
    check(report.failed_count == 1, "one file fails");

    JsonParseResult report_json = json_parse_file((output_dir / "report.json").string());
    check(!report_json.error, "report.json is valid JSON");
    const JsonValue* files = report_json.root.find("files");
    check(files->array.size() == 5, "report contains one entry per file");

    const JsonValue* overflow_entry = nullptr;
    const JsonValue* broken_entry = nullptr;
    const JsonValue* good_entry = nullptr;
    for (const JsonValue& item : files->array) {
        const std::string& input = item.find("input_path")->str;
        if (input.find("es_overflow") != std::string::npos) overflow_entry = &item;
        if (input.find("ja_broken") != std::string::npos) broken_entry = &item;
        if (input.find("en_good") != std::string::npos) good_entry = &item;
    }

    check(overflow_entry &&
              overflow_entry->find("error_code")->str == "E_PHONEME_OVERFLOW" &&
              overflow_entry->find("status")->str == "skipped",
          "overflow file reports E_PHONEME_OVERFLOW and skipped");
    check(broken_entry &&
              broken_entry->find("error_code")->str == "E_PARSE_JSON" &&
              broken_entry->find("status")->str == "failed",
          "corrupt JSON reports E_PARSE_JSON and failed");
    check(good_entry && good_entry->find("error_code")->str == "OK" &&
              good_entry->find("status")->str == "success",
          "well-formed file reports OK/success");
    check(good_entry->find("elapsed_ms")->number >= 0.0,
          "elapsed time measured with monotonic clock");

    // 越界文件：唇形跳过但音频产物保留。
    check(overflow_entry->find("output_curve_path")->str.empty(),
          "overflow file has no lip curve output");
    check(!overflow_entry->find("output_audio_path")->str.empty(),
          "overflow file still produces resampled audio");
    check(broken_entry->find("output_audio_path")->str.empty(),
          "broken-JSON file produces no audio (parse happens first)");

    // 校验输出 WAV 格式严格匹配配置。
    for (const char* name : {"en_good", "ja_good", "es_good", "es_overflow"}) {
        WavInfo info;
        fs::path wav = output_dir / "audio" / (std::string(name) + ".wav");
        check(!read_wav_info(wav.string(), info), "wav readable: " + std::string(name));
        check(info.sample_rate == 48000, "48kHz output: " + std::string(name));
        check(info.channels == 1, "mono output: " + std::string(name));
        check(info.bits_per_sample == 16, "PCM16 output: " + std::string(name));
        const double measured_seconds =
            static_cast<double>(info.data_bytes) /
            static_cast<double>(info.sample_rate * info.channels *
                                (info.bits_per_sample / 8));
        check(std::fabs(measured_seconds - 2.0) < 0.02,
              "resampled duration ~2.0s: " + std::string(name));
    }

    // 校验口型帧数 = ceil(2.0 * 30) = 60，且每帧权重和为 1。
    for (const char* name : {"en_good", "ja_good", "es_good"}) {
        fs::path curve = output_dir / "lipsync" /
                         (std::string(name) + "_lipsync.json");
        JsonParseResult curve_json = json_parse_file(curve.string());
        check(!curve_json.error, "curve parses: " + std::string(name));
        check(curve_json.root.find("frame_count")->number == 60.0,
              "60 frames for 2s @30fps: " + std::string(name));
        const JsonValue* curve_frames = curve_json.root.find("frames");
        check(curve_frames->array.size() == 60,
              "serialized 60 frames: " + std::string(name));
        const JsonValue* visemes = curve_json.root.find("visemes");
        for (const JsonValue& frame : curve_frames->array) {
            double sum = 0.0;
            const JsonValue& weights = *frame.find("weights");
            check(weights.array.size() == visemes->array.size(),
                  "frame carries every viseme weight");
            for (const JsonValue& weight : weights.array) sum += weight.number;
            check(std::fabs(sum - 1.0) < 1e-5,
                  "normalized weights: " + std::string(name));
        }
    }

    // 报告中的所有路径均相对输出目录。
    for (const JsonValue& item : files->array) {
        for (const char* key :
             {"input_path", "output_audio_path", "output_curve_path"}) {
            const std::string& value = item.find(key)->str;
            if (!value.empty()) {
                check(value.front() != '/' &&
                          value.find(':') == std::string::npos,
                      std::string("relative report path: ") + key);
            }
        }
    }

    fs::remove_all(root);
    std::cout << "integration checks: " << g_checks
              << ", failures: " << g_failures << "\n";
    return g_failures == 0 ? 0 : 1;
}
