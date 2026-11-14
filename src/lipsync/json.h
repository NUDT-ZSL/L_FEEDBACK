#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "error_codes.h"

namespace lipsync {

class JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    JsonArray array;
    JsonObject object;

    JsonValue() = default;

    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }
    bool is_string() const { return type == Type::String; }
    bool is_number() const { return type == Type::Number; }

    const JsonValue* find(std::string_view key) const {
        if (type != Type::Object) return nullptr;
        auto it = object.find(std::string(key));
        return it == object.end() ? nullptr : &it->second;
    }

    std::string get_string(std::string_view fallback) const {
        return type == Type::String ? str : std::string(fallback);
    }

    double get_number(double fallback) const {
        return type == Type::Number ? number : fallback;
    }
};

struct JsonParseResult {
    JsonValue root;
    ProcessError error;
};

// 解析 UTF-8 JSON 文本。失败时返回 E_PARSE_JSON，并在 message 中给出行列号。
JsonParseResult json_parse(const std::string& text);

// 读取并解析文件；IO 失败返回 E_IO，解析失败返回 E_PARSE_JSON。
JsonParseResult json_parse_file(const std::string& path);

// 紧凑序列化（确定性：对象按 key 字典序输出）。
std::string json_dump(const JsonValue& value, bool pretty = true);

bool write_text_file(const std::string& path, const std::string& content, std::string* error_message);

} // namespace lipsync
