#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace automix {

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    JsonValue() = default;
    explicit JsonValue(bool value) : type_(Type::Bool), bool_(value) {}
    explicit JsonValue(double value) : type_(Type::Number), number_(value) {}
    explicit JsonValue(std::string value) : type_(Type::String), string_(std::move(value)) {}

    static JsonValue array(std::vector<JsonValue> values);
    static JsonValue object(std::map<std::string, JsonValue> values);

    Type type() const noexcept { return type_; }
    bool isNull() const noexcept { return type_ == Type::Null; }
    bool isBool() const noexcept { return type_ == Type::Bool; }
    bool isNumber() const noexcept { return type_ == Type::Number; }
    bool isString() const noexcept { return type_ == Type::String; }
    bool isArray() const noexcept { return type_ == Type::Array; }
    bool isObject() const noexcept { return type_ == Type::Object; }

    bool asBool() const { return bool_; }
    double asNumber() const { return number_; }
    const std::string& asString() const { return string_; }
    const std::vector<JsonValue>& asArray() const { return array_; }
    const std::map<std::string, JsonValue>& asObject() const { return object_; }

    const JsonValue* find(const std::string& key) const;

private:
    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<JsonValue> array_;
    std::map<std::string, JsonValue> object_;
};

struct JsonParseError {
    std::size_t line = 1;
    std::size_t column = 1;
    std::string message;
};

JsonValue parseJson(const std::string& text, JsonParseError& error);
std::string jsonQuote(const std::string& text);

}  // namespace automix
