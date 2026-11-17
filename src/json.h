#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace softraster {

class JsonValue {
public:
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue, std::less<>>;

    JsonValue();
    explicit JsonValue(std::nullptr_t);
    explicit JsonValue(bool value);
    explicit JsonValue(double value);
    explicit JsonValue(std::string value);
    explicit JsonValue(Array value);
    explicit JsonValue(Object value);

    [[nodiscard]] bool isNull() const;
    [[nodiscard]] bool isBool() const;
    [[nodiscard]] bool isNumber() const;
    [[nodiscard]] bool isString() const;
    [[nodiscard]] bool isArray() const;
    [[nodiscard]] bool isObject() const;

    [[nodiscard]] bool asBool() const;
    [[nodiscard]] double asNumber() const;
    [[nodiscard]] const std::string& asString() const;
    [[nodiscard]] const Array& asArray() const;
    [[nodiscard]] Array& asArray();
    [[nodiscard]] const Object& asObject() const;
    [[nodiscard]] Object& asObject();

    [[nodiscard]] const JsonValue* find(std::string_view key) const;

    using Storage = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;
    [[nodiscard]] const Storage& storage() const;

private:
    Storage value_;
};

struct JsonParseResult {
    std::optional<JsonValue> value;
    std::string error;
    std::size_t errorOffset = 0;
};

[[nodiscard]] JsonParseResult parseJson(std::string_view text);
[[nodiscard]] std::string writeJson(const JsonValue& value, bool pretty = true);

}  // namespace softraster
