#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace preview::json {

class ParseError final : public std::runtime_error {
public:
    explicit ParseError(const std::string& message) : std::runtime_error(message) {}
};

struct Value {
    enum class Type { Null, Boolean, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;

    bool isNull() const noexcept { return type == Type::Null; }
    bool isBoolean() const noexcept { return type == Type::Boolean; }
    bool isNumber() const noexcept { return type == Type::Number; }
    bool isString() const noexcept { return type == Type::String; }
    bool isArray() const noexcept { return type == Type::Array; }
    bool isObject() const noexcept { return type == Type::Object; }
};

Value parse(const std::string& text);

} // namespace preview::json
