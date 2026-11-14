#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "error_codes.h"

namespace lipsync {

struct MappingWarning {
    std::string language;
    std::string phoneme;
    std::string message;
};

// 数据驱动的音素 -> viseme 映射器。
// 逻辑不感知任何具体语言标识：语言只是映射表的键，可通过 JSON 任意扩展。
// 每个音素只对应一个 viseme；未知音素回退静默 viseme 并记录警告。
class VisemeMapper {
public:
    // 从 JSON 文件加载；path 为空时使用内置 en/ja/es 映射。
    static ProcessError load(const std::string& path, VisemeMapper& mapper);

    const std::vector<std::string>& viseme_names() const { return viseme_names_; }

    int silence_index() const { return silence_index_; }

    // 返回 viseme 下标；未知音素返回静默 viseme 下标并追加警告，绝不抛异常。
    int map_phoneme(const std::string& language,
                    const std::string& phoneme,
                    std::vector<MappingWarning>& warnings) const;

    bool has_language(const std::string& language) const;

private:
    std::vector<std::string> viseme_names_;
    int silence_index_ = 0;
    // language -> (大写音素符号 -> viseme 下标)
    std::unordered_map<std::string,
                       std::unordered_map<std::string, int>> tables_;

    ProcessError build_from_json(const std::string& text);
};

} // namespace lipsync
