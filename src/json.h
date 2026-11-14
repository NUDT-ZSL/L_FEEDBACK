#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace cloud::json {

class Value {
public:
    using Array = std::vector<Value>;
    using Object = std::map<std::string, Value>;

    Value() : value_(nullptr) {}
    Value(std::nullptr_t) : value_(nullptr) {}
    Value(bool value) : value_(value) {}
    Value(int value) : value_(static_cast<double>(value)) {}
    Value(unsigned value) : value_(static_cast<double>(value)) {}
    Value(std::int64_t value) : value_(static_cast<double>(value)) {}
    Value(std::uint64_t value) : value_(static_cast<double>(value)) {}
    Value(float value) : value_(static_cast<double>(value)) {}
    Value(double value) : value_(value) {}
    Value(const char* value) : value_(std::string(value)) {}
    Value(std::string value) : value_(std::move(value)) {}
    Value(Array value) : value_(std::move(value)) {}
    Value(Object value) : value_(std::move(value)) {}

    bool isNull() const { return std::holds_alternative<std::nullptr_t>(value_); }
    bool isBool() const { return std::holds_alternative<bool>(value_); }
    bool isNumber() const { return std::holds_alternative<double>(value_); }
    bool isString() const { return std::holds_alternative<std::string>(value_); }
    bool isArray() const { return std::holds_alternative<Array>(value_); }
    bool isObject() const { return std::holds_alternative<Object>(value_); }

    bool asBool() const;
    double asNumber() const;
    std::int64_t asInt64() const;
    std::uint64_t asUInt64() const;
    const std::string& asString() const;
    const Array& asArray() const;
    const Object& asObject() const;
    Array& asArray();
    Object& asObject();

    const Value* find(const std::string& key) const;
    Value& operator[](const std::string& key);

private:
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value_;
};

class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string& message) : std::runtime_error(message) {}
};

Value parse(const std::string& text);
std::string stringify(const Value& value, bool pretty = false);

}  // namespace cloud::json

