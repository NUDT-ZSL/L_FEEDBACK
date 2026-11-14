#include "phoneme_parser.h"

#include <sstream>

#include "json.h"

namespace lipsync {

ProcessError parse_phoneme_file(const std::string& json_path, PhonemeTrack& track) {
    JsonParseResult parsed = json_parse_file(json_path);
    if (parsed.error) return parsed.error; // E_IO 或 E_PARSE_JSON 原样上抛

    const JsonValue& root = parsed.root;
    if (!root.is_object()) {
        return ProcessError(ErrorCode::E_PARSE_JSON, "phoneme JSON root must be an object");
    }
    const JsonValue* language = root.find("language");
    if (!language || !language->is_string() || language->str.empty()) {
        return ProcessError(ErrorCode::E_PARSE_JSON, "phoneme JSON requires string 'language'");
    }
    const JsonValue* phonemes = root.find("phonemes");
    if (!phonemes || !phonemes->is_array()) {
        return ProcessError(ErrorCode::E_PARSE_JSON, "phoneme JSON requires array 'phonemes'");
    }

    PhonemeTrack result;
    result.language = language->str;
    result.intervals.reserve(phonemes->array.size());
    for (size_t i = 0; i < phonemes->array.size(); ++i) {
        const JsonValue& item = phonemes->array[i];
        if (!item.is_object()) {
            return ProcessError(ErrorCode::E_PARSE_JSON,
                                "phonemes[" + std::to_string(i) + "] must be an object");
        }
        const JsonValue* ph = item.find("phoneme");
        const JsonValue* start = item.find("start");
        const JsonValue* end = item.find("end");
        if (!ph || !ph->is_string() || ph->str.empty() ||
            !start || !start->is_number() ||
            !end || !end->is_number()) {
            return ProcessError(ErrorCode::E_PARSE_JSON,
                                "phonemes[" + std::to_string(i) +
                                    "] requires string phoneme and numeric start/end");
        }
        PhonemeInterval interval;
        interval.phoneme = ph->str;
        interval.start = start->number;
        interval.end = end->number;
        if (interval.start < 0.0 || interval.end < interval.start) {
            return ProcessError(ErrorCode::E_PARSE_JSON,
                                "phonemes[" + std::to_string(i) +
                                    "] has invalid time range");
        }
        result.intervals.push_back(std::move(interval));
    }
    track = std::move(result);
    return ProcessError{};
}

ProcessError validate_phoneme_track(const PhonemeTrack& track,
                                    double audio_duration_seconds,
                                    double tolerance_ms) {
    const double tolerance = tolerance_ms / 1000.0;
    double prev_end = 0.0;
    for (size_t i = 0; i < track.intervals.size(); ++i) {
        const PhonemeInterval& interval = track.intervals[i];
        if (i > 0 && interval.start < prev_end) {
            std::ostringstream oss;
            oss << "overlapping phoneme intervals at index " << i;
            return ProcessError(ErrorCode::E_PARSE_JSON, oss.str());
        }
        prev_end = interval.end;
        if (interval.end > audio_duration_seconds + tolerance) {
            std::ostringstream oss;
            oss.precision(3);
            oss << "phoneme end " << interval.end
                << "s exceeds audio duration " << audio_duration_seconds
                << "s beyond tolerance " << tolerance_ms << "ms";
            return ProcessError(ErrorCode::E_PHONEME_OVERFLOW, oss.str());
        }
    }
    return ProcessError{};
}

} // namespace lipsync
