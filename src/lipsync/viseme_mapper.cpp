#include "viseme_mapper.h"

#include <algorithm>
#include <cctype>

#include "json.h"

namespace lipsync {
namespace {

std::string upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return value;
}

// 内置默认映射：15 个 canonical viseme（SIL 为静默），覆盖 en/ja/es。
// 与语言标识解耦：解析逻辑对所有语言一视同仁，仅按表查找。
constexpr const char* kDefaultMapping = R"JSON(
{
  "silence_viseme": "SIL",
  "visemes": [
    "SIL", "PP", "FF", "TH", "DD", "KK",
    "CH", "SS", "NN", "RR", "AA", "E",
    "IH", "OH", "OU"
  ],
  "languages": {
    "en": {
      "AA": "AA", "AE": "AA", "AH": "AA", "AO": "OH",
      "AW": "OU", "AY": "AA",
      "B": "PP", "CH": "CH", "D": "DD", "DH": "TH",
      "EH": "E", "ER": "RR", "EY": "E",
      "F": "FF", "G": "KK", "HH": "AA", "IH": "IH",
      "IY": "IH", "JH": "CH", "K": "KK", "L": "NN",
      "M": "PP", "N": "NN", "NG": "KK", "OW": "OU",
      "OY": "OH", "P": "PP", "R": "RR", "S": "SS",
      "SH": "SS", "T": "DD", "TH": "TH", "UH": "OU",
      "UW": "OU", "V": "FF", "W": "OU", "Y": "IH",
      "Z": "SS", "ZH": "SS", "SIL": "SIL", "SP": "SIL"
    },
    "ja": {
      "A": "AA", "I": "IH", "U": "OU", "E": "E", "O": "OH",
      "N": "NN", "Q": "SIL",
      "K": "KK", "G": "KK", "S": "SS", "Z": "SS",
      "T": "DD", "D": "DD", "CH": "CH", "J": "CH",
      "H": "AA", "F": "FF", "B": "PP", "P": "PP",
      "M": "PP", "Y": "IH", "R": "RR", "W": "OU",
      "SH": "SS", "TS": "SS", "SIL": "SIL"
    },
    "es": {
      "A": "AA", "E": "E", "I": "IH", "O": "OH", "U": "OU",
      "B": "PP", "P": "PP", "M": "PP",
      "T": "DD", "D": "DD",
      "K": "KK", "G": "KK", "CH": "CH", "LL": "NN",
      "N": "NN", "NY": "NN",
      "F": "FF", "S": "SS", "Z": "SS", "X": "KK",
      "R": "RR", "RR": "RR", "L": "NN", "Y": "IH",
      "W": "OU", "GN": "NN",
      "SIL": "SIL"
    }
  }
}
)JSON";

} // namespace

ProcessError VisemeMapper::load(const std::string& path, VisemeMapper& mapper) {
    std::string text = kDefaultMapping;
    if (!path.empty()) {
        JsonParseResult file = json_parse_file(path);
        if (file.error) return file.error;
        text = json_dump(file.root, false);
    }
    return mapper.build_from_json(text);
}

ProcessError VisemeMapper::build_from_json(const std::string& text) {
    JsonParseResult parsed = json_parse(text);
    if (parsed.error) return parsed.error;
    const JsonValue& root = parsed.root;
    if (!root.is_object()) {
        return ProcessError(ErrorCode::E_CONFIG, "viseme map root must be an object");
    }
    const JsonValue* visemes = root.find("visemes");
    if (!visemes || !visemes->is_array() || visemes->array.empty()) {
        return ProcessError(ErrorCode::E_CONFIG, "viseme map requires non-empty 'visemes' array");
    }
    viseme_names_.clear();
    for (const JsonValue& v : visemes->array) {
        if (!v.is_string() || v.str.empty()) {
            return ProcessError(ErrorCode::E_CONFIG, "viseme names must be non-empty strings");
        }
        viseme_names_.push_back(v.str);
    }
    const JsonValue* silence = root.find("silence_viseme");
    std::string silence_name = silence ? silence->get_string("SIL") : "SIL";
    auto sil_it = std::find(viseme_names_.begin(), viseme_names_.end(), silence_name);
    if (sil_it == viseme_names_.end()) {
        return ProcessError(ErrorCode::E_CONFIG,
                            "silence_viseme not found in visemes list: " + silence_name);
    }
    silence_index_ = static_cast<int>(std::distance(viseme_names_.begin(), sil_it));

    const JsonValue* languages = root.find("languages");
    if (!languages || !languages->is_object()) {
        return ProcessError(ErrorCode::E_CONFIG, "viseme map requires 'languages' object");
    }

    std::unordered_map<std::string, int> viseme_index;
    for (size_t i = 0; i < viseme_names_.size(); ++i) {
        viseme_index[viseme_names_[i]] = static_cast<int>(i);
    }

    tables_.clear();
    for (const auto& lang_kv : languages->object) {
        if (!lang_kv.second.is_object()) {
            return ProcessError(ErrorCode::E_CONFIG,
                                "language table must be an object: " + lang_kv.first);
        }
        std::unordered_map<std::string, int> table;
        for (const auto& kv : lang_kv.second.object) {
            auto vit = viseme_index.find(kv.second.str);
            if (vit == viseme_index.end()) {
                return ProcessError(ErrorCode::E_CONFIG,
                                    "unknown viseme '" + kv.second.str +
                                    "' referenced by phoneme '" + kv.first + "'");
            }
            table[upper(kv.first)] = vit->second; // 单值映射：一个音素仅一个 viseme
        }
        tables_[lang_kv.first] = std::move(table);
    }
    return ProcessError{};
}

bool VisemeMapper::has_language(const std::string& language) const {
    return tables_.find(language) != tables_.end();
}

int VisemeMapper::map_phoneme(const std::string& language,
                              const std::string& phoneme,
                              std::vector<MappingWarning>& warnings) const {
    auto lang_it = tables_.find(language);
    std::string key = upper(phoneme);
    if (lang_it != tables_.end()) {
        auto ph_it = lang_it->second.find(key);
        if (ph_it != lang_it->second.end()) {
            return ph_it->second;
        }
    }
    warnings.push_back(MappingWarning{
        language, phoneme,
        "unknown phoneme '" + phoneme + "' for language '" + language +
            "', falling back to silence viseme"});
    return silence_index_;
}

} // namespace lipsync
