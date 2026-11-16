#pragma once

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilegen {

class JsonError : public std::runtime_error {
public:
    JsonError(const std::string& message, std::size_t line, std::size_t column);

    std::size_t line() const noexcept { return line_; }
    std::size_t column() const noexcept { return column_; }

private:
    std::size_t line_;
    std::size_t column_;
};

class JsonValue {
public:
    enum class Type { Null, Boolean, Number, String, Array, Object };

    JsonValue() = default;
    explicit JsonValue(bool value);
    explicit JsonValue(double value);
    explicit JsonValue(std::string value);
    explicit JsonValue(std::vector<JsonValue> value);
    explicit JsonValue(std::map<std::string, JsonValue> value);

    Type type() const noexcept { return type_; }
    bool isNull() const noexcept { return type_ == Type::Null; }
    bool isBool() const noexcept { return type_ == Type::Boolean; }
    bool isNumber() const noexcept { return type_ == Type::Number; }
    bool isString() const noexcept { return type_ == Type::String; }
    bool isArray() const noexcept { return type_ == Type::Array; }
    bool isObject() const noexcept { return type_ == Type::Object; }

    bool asBool() const;
    double asNumber() const;
    const std::string& asString() const;
    const std::vector<JsonValue>& asArray() const;
    const std::map<std::string, JsonValue>& asObject() const;

    const JsonValue* find(const std::string& key) const;
    bool contains(const std::string& key) const;

private:
    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<JsonValue> array_;
    std::map<std::string, JsonValue> object_;
};

class JsonParser {
public:
    JsonValue parse(const std::string& text);
    JsonValue parseFile(const std::string& path);

private:
    JsonValue parseValue(std::size_t depth);
    JsonValue parseObject(std::size_t depth);
    JsonValue parseArray(std::size_t depth);
    JsonValue parseString();
    JsonValue parseNumber();
    void parseLiteral(const char* literal, const JsonValue& value);
    void skipWhitespace();
    char peek() const;
    char take();
    bool consume(char expected);
    void expect(char expected, const std::string& message);
    [[noreturn]] void error(const std::string& message) const;
    static void appendUtf8(std::string& out, unsigned int codepoint);
    std::size_t lineAt(std::size_t position) const;
    std::size_t columnAt(std::size_t position) const;

    std::string text_;
    std::size_t position_ = 0;
};

std::string readTextFile(const std::string& path);
std::string jsonPath(const std::string& base, const std::string& key);
std::string jsonPath(const std::string& base, std::size_t index);

}  // namespace tilegen
