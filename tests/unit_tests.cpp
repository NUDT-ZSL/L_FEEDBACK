#include <cmath>
#include <cstdlib>
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

#include "lipsync/config.h"
#include "lipsync/json.h"
#include "lipsync/lip_sync.h"
#include "lipsync/phoneme_parser.h"
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

bool nearly_equal(double a, double b, double eps = 1e-4) {
    return std::fabs(a - b) <= eps;
}

fs::path make_temp_dir() {
#if defined(_WIN32)
    const int pid = _getpid();
#else
    const int pid = ::getpid();
#endif
    fs::path dir = fs::temp_directory_path() /
                   ("lipsync_unit_" + std::to_string(pid));
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

void test_json_roundtrip_and_errors() {
    JsonParseResult ok = json_parse(R"({"a":[1,2.5,true,null,"x"],"b":{"k":"v"}})");
    check(!ok.error, "valid JSON parses");
    check(ok.root.is_object(), "root is object");
    const JsonValue* arr = ok.root.find("a");
    check(arr && arr->is_array() && arr->array.size() == 5, "nested array size");
    check((*arr).array[1].number == 2.5, "number value");
    check(ok.root.find("b")->find("k")->str == "v", "nested string");

    JsonParseResult bad = json_parse(R"({"a": [1, 2,], })");
    check(bad.error && bad.error.code == ErrorCode::E_PARSE_JSON,
          "corrupt JSON yields E_PARSE_JSON");

    std::string dumped = json_dump(ok.root, false);
    JsonParseResult reparsed = json_parse(dumped);
    check(!reparsed.error, "reparsed dump is valid JSON");
    check(reparsed.root.find("a")->array.size() == 5, "roundtrip keeps array size");
}

void test_config_loading(const fs::path& dir) {
    fs::path path = dir / "config.json";
    {
        std::ofstream out(path);
        out << R"({
          "target_sample_rate": 48000,
          "target_channels": 1,
          "fps": 25,
          "languages": ["en", "ja", "es"],
          "max_workers": 2
        })";
    }
    AppConfig config;
    check(!load_config(path.string(), config), "config loads");
    check(config.target_sample_rate == 48000 && config.fps == 25.0 &&
              config.max_workers == 2 && config.languages.size() == 3,
          "config field values");

    fs::path bad_path = dir / "bad_config.json";
    {
        std::ofstream out(bad_path);
        out << R"({"target_sample_rate": -1})";
    }
    AppConfig bad_config;
    ProcessError error = load_config(bad_path.string(), bad_config);
    check(error && error.code == ErrorCode::E_CONFIG, "invalid config yields E_CONFIG");
}

void test_viseme_mapping() {
    VisemeMapper mapper;
    check(!VisemeMapper::load("", mapper), "default mapping loads");
    check(mapper.viseme_names().size() == 15, "15 canonical visemes");
    check(mapper.has_language("en") && mapper.has_language("ja") &&
              mapper.has_language("es"),
          "three built-in languages");

    std::vector<MappingWarning> warnings;
    int aa = mapper.map_phoneme("en", "aa", warnings); // 大小写不敏感
    check(aa >= 0 && mapper.viseme_names()[aa] == "AA",
          "English AA maps to viseme AA case-insensitively");
    int ja_a = mapper.map_phoneme("ja", "A", warnings);
    check(mapper.viseme_names()[ja_a] == "AA", "Japanese A maps to viseme AA");
    int es_b = mapper.map_phoneme("es", "B", warnings);
    check(mapper.viseme_names()[es_b] == "PP", "Spanish B maps to viseme PP");
    check(warnings.empty(), "known phonemes produce no warnings");

    int unknown = mapper.map_phoneme("en", "ZZZ", warnings);
    check(unknown == mapper.silence_index(),
          "unknown phoneme falls back to silence viseme");
    check(warnings.size() == 1, "unknown phoneme records one warning");

    int unknown_lang = mapper.map_phoneme("fr", "A", warnings);
    check(unknown_lang == mapper.silence_index() && warnings.size() == 2,
          "unknown language also falls back with warning, no interruption");
}

void test_phoneme_validation() {
    PhonemeTrack track;
    track.language = "en";
    track.intervals = {
        {"AA", 0.0, 0.5},
        {"B", 0.5, 2.0},
    };
    check(!validate_phoneme_track(track, 2.0, 100.0),
          "in-range phonemes pass validation");

    PhonemeTrack overflow = track;
    overflow.intervals.back().end = 2.2; // 超出 200ms
    ProcessError error = validate_phoneme_track(overflow, 2.0, 100.0);
    check(error && error.code == ErrorCode::E_PHONEME_OVERFLOW,
          "overflow beyond 100ms yields E_PHONEME_OVERFLOW");

    PhonemeTrack within_tolerance = track;
    within_tolerance.intervals.back().end = 2.08; // 超出 80ms，容差内
    check(!validate_phoneme_track(within_tolerance, 2.0, 100.0),
          "overflow within tolerance is accepted");

    PhonemeTrack overlapping = track;
    overlapping.intervals[1].start = 0.4;
    error = validate_phoneme_track(overlapping, 2.0, 100.0);
    check(error && error.code == ErrorCode::E_PARSE_JSON,
          "overlapping intervals reported as parse-time error");
}

void test_curve_generation(const fs::path& dir) {
    VisemeMapper mapper;
    VisemeMapper::load("", mapper);

    PhonemeTrack track;
    track.language = "en";
    track.intervals = {
        {"P", 0.0, 0.48},
        {"AA", 0.48, 1.0},
        {"SIL", 1.0, 1.5},
    };
    const double duration = 2.0;
    const double fps = 30.0;

    LipSyncResult result;
    fs::path curve_path = dir / "curve.json";
    ProcessError error = generate_lip_sync(track, duration, fps, mapper,
                                           curve_path.string(), result);
    check(!error, "curve generation succeeds");
    const int expected_frames = static_cast<int>(std::ceil(duration * fps));
    check(result.frame_count == expected_frames,
          "frame count equals ceil(duration * fps) = 60");

    JsonParseResult parsed = json_parse_file(curve_path.string());
    check(!parsed.error, "curve JSON parses");
    const JsonValue* frames = parsed.root.find("frames");
    check(frames && frames->array.size() == static_cast<size_t>(expected_frames),
          "serialized frame count matches");
    const JsonValue* visemes = parsed.root.find("visemes");
    const size_t viseme_count = visemes->array.size();

    for (size_t i = 0; i < frames->array.size(); ++i) {
        const JsonValue& weights = *frames->array[i].find("weights");
        check(weights.array.size() == viseme_count,
              "every frame contains all viseme weights");
        double sum = 0.0;
        for (const JsonValue& w : weights.array) sum += w.number;
        check(nearly_equal(sum, 1.0, 1e-5), "per-frame weights sum to 1");
        check(nearly_equal(frames->array[i].find("t")->number, i / fps, 1e-9),
              "frame timestamps start at zero and advance by 1/fps");
    }

    // 最后一帧位于 1.9667s，音素只覆盖到 1.5s -> 静默权重必须为 1。
    const JsonValue& last_frame = frames->array.back();
    const JsonValue& last_weights = *last_frame.find("weights");
    check(nearly_equal(last_weights.array[mapper.silence_index()].number, 1.0),
          "uncovered frame is pure silence");

    // 首帧完全落在 P 区间内 -> P 对应 viseme PP 权重为 1。
    const JsonValue& first_weights = *frames->array.front().find("weights");
    std::vector<MappingWarning> probe_warnings;
    int p_viseme = mapper.map_phoneme("en", "P", probe_warnings);
    check(nearly_equal(first_weights.array[p_viseme].number, 1.0),
          "fully covered frame is one-hot on the mapped viseme");

    // 边界帧应出现两个 viseme 的混合权重：
    // 帧 14 覆盖 [0.4667, 0.5)，跨在 P(PP, 结束于 0.48)/AA 分界上。
    const JsonValue& boundary = frames->array[14];
    double nonzero = 0.0;
    for (const JsonValue& w : boundary.find("weights")->array) {
        if (w.number > 0.0) ++nonzero;
    }
    check(nonzero == 2.0, "phoneme boundary frame blends two visemes");

    // 未知音素不中断生成，仅产生警告。
    PhonemeTrack weird = track;
    weird.intervals.push_back({"ZZZ", 1.5, 1.8});
    LipSyncResult weird_result;
    check(!generate_lip_sync(weird, duration, fps, mapper,
                             (dir / "curve_weird.json").string(), weird_result),
          "unknown phoneme does not interrupt curve generation");
    check(weird_result.warnings.size() == 1, "unknown phoneme warning captured");
}

} // namespace

int main() {
    fs::path dir = make_temp_dir();
    test_json_roundtrip_and_errors();
    test_config_loading(dir);
    test_viseme_mapping();
    test_phoneme_validation();
    test_curve_generation(dir);
    fs::remove_all(dir);

    std::cout << "unit checks: " << g_checks
              << ", failures: " << g_failures << "\n";
    return g_failures == 0 ? 0 : 1;
}
