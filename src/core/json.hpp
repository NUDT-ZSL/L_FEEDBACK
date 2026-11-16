#pragma once

#include "core/error.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace uid166 {

class JsonWriter {
public:
    explicit JsonWriter(bool pretty = true) : pretty_(pretty) {}

    void beginObject();
    void endObject();
    void beginArray();
    void endArray();
    void key(std::string_view value);
    void string(std::string_view value);
    void number(double value);
    void integer(int64_t value);
    void boolean(bool value);
    void null();

    [[nodiscard]] const std::string& str() const { return out_; }
    [[nodiscard]] std::string take() { return std::move(out_); }

private:
    enum class Container { object, array };
    void beforeValue();
    void newline();
    void appendEscaped(std::string_view value);

    bool pretty_ = true;
    std::string out_;
    std::vector<Container> stack_;
    std::vector<size_t> counts_;
    bool afterKey_ = false;
};

void writeTextFile(const std::filesystem::path& path, std::string_view text);
[[nodiscard]] std::string readTextFile(const std::filesystem::path& path,
                                       ErrorCode missingCode, ErrorCode readCode,
                                       size_t maxBytes = 64u * 1024u * 1024u);

enum class JsonType { nullValue, booleanValue, numberValue, stringValue, arrayValue, objectValue };

struct JsonValue {
    JsonType type = JsonType::nullValue;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object;

    [[nodiscard]] const JsonValue* find(std::string_view name) const;
};

[[nodiscard]] JsonValue parseJson(std::string_view text, ErrorCode parseCode,
                                  ErrorCode schemaCode);
[[nodiscard]] JsonValue parseJsonFile(const std::filesystem::path& path,
                                      ErrorCode missingCode, ErrorCode readCode,
                                      ErrorCode parseCode, ErrorCode schemaCode);

[[nodiscard]] const JsonValue& requireObject(const JsonValue& value, std::string_view where,
                                             ErrorCode schemaCode);
[[nodiscard]] const JsonValue& requireArray(const JsonValue& value, std::string_view where,
                                            ErrorCode schemaCode);
[[nodiscard]] const JsonValue& requireMember(const JsonValue& object, std::string_view name,
                                             ErrorCode schemaCode);
[[nodiscard]] double requireNumber(const JsonValue& value, std::string_view where,
                                   ErrorCode schemaCode);
[[nodiscard]] int64_t requireInteger(const JsonValue& value, std::string_view where,
                                     ErrorCode schemaCode);
[[nodiscard]] std::string requireString(const JsonValue& value, std::string_view where,
                                        ErrorCode schemaCode);

} // namespace uid166

