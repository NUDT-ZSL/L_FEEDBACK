#include "lip_sync.h"

#include <algorithm>
#include <cmath>

#include "json.h"

namespace lipsync {
namespace {

constexpr double kEpsilon = 1e-9;

} // namespace

ProcessError generate_lip_sync(const PhonemeTrack& track,
                               double audio_duration_seconds,
                               double fps,
                               const VisemeMapper& mapper,
                               const std::string& output_curve_path,
                               LipSyncResult& result) {
    const auto& names = mapper.viseme_names();
    const int viseme_count = static_cast<int>(names.size());
    const int silence = mapper.silence_index();

    // 先把每个音素区间解析成 viseme 下标，未知音素只警告一次。
    std::vector<int> interval_visemes;
    interval_visemes.reserve(track.intervals.size());
    for (const PhonemeInterval& interval : track.intervals) {
        interval_visemes.push_back(
            mapper.map_phoneme(track.language, interval.phoneme, result.warnings));
    }

    const int frame_count = std::max(
        1, static_cast<int>(std::ceil(audio_duration_seconds * fps - kEpsilon)));
    const double frame_len = 1.0 / fps;

    VisemeCurve curve;
    curve.fps = fps;
    curve.viseme_names = names;
    curve.frames.reserve(frame_count);

    for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
        const double frame_start = frame_index * frame_len;
        const double frame_end = std::min(audio_duration_seconds,
                                          (frame_index + 1) * frame_len);
        const double effective_len = frame_end - frame_start;

        std::vector<double> masses(viseme_count, 0.0);
        double covered = 0.0;
        for (size_t i = 0; i < track.intervals.size(); ++i) {
            const PhonemeInterval& interval = track.intervals[i];
            const double lo = std::max(frame_start, interval.start);
            const double hi = std::min(frame_end, interval.end);
            if (hi > lo) {
                const double overlap = hi - lo;
                masses[interval_visemes[i]] += overlap;
                covered += overlap;
            }
        }

        // 没有音素覆盖的部分计入静默 viseme（包含完全静默的帧）。
        const double silence_mass = std::max(0.0, effective_len - covered);
        masses[silence] += silence_mass;

        VisemeFrame frame;
        frame.weights.resize(viseme_count);
        double total = 0.0;
        for (double mass : masses) total += mass;
        if (total <= kEpsilon) {
            // 理论上不会发生；兜底保证静默帧权重合法。
            for (int j = 0; j < viseme_count; ++j) frame.weights[j] = 0.0f;
            frame.weights[silence] = 1.0f;
        } else {
            for (int j = 0; j < viseme_count; ++j) {
                frame.weights[j] = static_cast<float>(masses[j] / total);
            }
        }
        curve.frames.push_back(std::move(frame));
    }

    JsonValue root;
    root.type = JsonValue::Type::Object;

    JsonValue language_value;
    language_value.type = JsonValue::Type::String;
    language_value.str = track.language;
    root.object.emplace("language", std::move(language_value));

    JsonValue fps_value;
    fps_value.type = JsonValue::Type::Number;
    fps_value.number = fps;
    root.object.emplace("fps", std::move(fps_value));

    JsonValue duration_value;
    duration_value.type = JsonValue::Type::Number;
    duration_value.number = audio_duration_seconds;
    root.object.emplace("duration_seconds", std::move(duration_value));

    JsonValue frame_count_value;
    frame_count_value.type = JsonValue::Type::Number;
    frame_count_value.number = static_cast<double>(frame_count);
    root.object.emplace("frame_count", std::move(frame_count_value));

    JsonValue viseme_array;
    viseme_array.type = JsonValue::Type::Array;
    for (const std::string& name : names) {
        JsonValue item;
        item.type = JsonValue::Type::String;
        item.str = name;
        viseme_array.array.push_back(std::move(item));
    }
    root.object.emplace("visemes", std::move(viseme_array));

    JsonValue frames_array;
    frames_array.type = JsonValue::Type::Array;
    for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
        JsonValue frame_object;
        frame_object.type = JsonValue::Type::Object;

        JsonValue time_value;
        time_value.type = JsonValue::Type::Number;
        time_value.number = frame_index * frame_len;
        frame_object.object.emplace("t", std::move(time_value));

        JsonValue weights_array;
        weights_array.type = JsonValue::Type::Array;
        for (float weight : curve.frames[frame_index].weights) {
            JsonValue weight_value;
            weight_value.type = JsonValue::Type::Number;
            weight_value.number = static_cast<double>(weight);
            weights_array.array.push_back(std::move(weight_value));
        }
        frame_object.object.emplace("weights", std::move(weights_array));
        frames_array.array.push_back(std::move(frame_object));
    }
    root.object.emplace("frames", std::move(frames_array));

    std::string io_error;
    if (!write_text_file(output_curve_path, json_dump(root), &io_error)) {
        return ProcessError(ErrorCode::E_IO, io_error);
    }

    result.output_curve_path = output_curve_path;
    result.frame_count = frame_count;
    return ProcessError{};
}

} // namespace lipsync
