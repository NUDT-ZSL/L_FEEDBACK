#include "config.h"

#include "json.h"

#include <algorithm>
#include <thread>

namespace lipsync {

ProcessError load_config(const std::string& config_path, AppConfig& config) {
    JsonParseResult parsed = json_parse_file(config_path);
    if (parsed.error) return parsed.error;

    const JsonValue& root = parsed.root;
    if (!root.is_object()) {
        return ProcessError(ErrorCode::E_CONFIG, "config root must be a JSON object");
    }

    if (const JsonValue* v = root.find("target_sample_rate")) {
        if (!v->is_number() || v->number <= 0) {
            return ProcessError(ErrorCode::E_CONFIG, "target_sample_rate must be a positive number");
        }
        config.target_sample_rate = static_cast<int>(v->number);
    }
    if (const JsonValue* v = root.find("target_channels")) {
        if (!v->is_number() || (v->number != 1 && v->number != 2)) {
            return ProcessError(ErrorCode::E_CONFIG, "target_channels must be 1 or 2");
        }
        config.target_channels = static_cast<int>(v->number);
    }
    if (const JsonValue* v = root.find("fps")) {
        if (!v->is_number() || v->number <= 0) {
            return ProcessError(ErrorCode::E_CONFIG, "fps must be a positive number");
        }
        config.fps = v->number;
    }
    if (const JsonValue* v = root.find("max_workers")) {
        if (!v->is_number() || v->number < 1) {
            return ProcessError(ErrorCode::E_CONFIG, "max_workers must be >= 1");
        }
        config.max_workers = static_cast<int>(v->number);
    } else {
        config.max_workers = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }
    if (const JsonValue* v = root.find("overflow_tolerance_ms")) {
        if (!v->is_number() || v->number < 0) {
            return ProcessError(ErrorCode::E_CONFIG, "overflow_tolerance_ms must be >= 0");
        }
        config.overflow_tolerance_ms = v->number;
    }
    if (const JsonValue* v = root.find("viseme_map_path")) {
        if (!v->is_string() || v->str.empty()) {
            return ProcessError(ErrorCode::E_CONFIG, "viseme_map_path must be a non-empty string");
        }
        config.viseme_map_path = v->str;
    }

    config.languages.clear();
    if (const JsonValue* v = root.find("languages")) {
        if (!v->is_array()) {
            return ProcessError(ErrorCode::E_CONFIG, "languages must be an array");
        }
        for (const JsonValue& lang : v->array) {
            if (!lang.is_string() || lang.str.empty()) {
                return ProcessError(ErrorCode::E_CONFIG, "language entries must be non-empty strings");
            }
            config.languages.push_back(lang.str);
        }
    }
    if (config.languages.empty()) {
        config.languages = {"en", "ja", "es"};
    }
    return ProcessError{};
}

} // namespace lipsync
