#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfb::json {

class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string& message) : std::runtime_error(message) {}
};

class Value {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Value() = default;
    explicit Value(bool value) : type_(Type::Bool), bool_(value) {}
    explicit Value(double value) : type_(Type::Number), number_(value) {}
    explicit Value(std::string value) : type_(Type::String), string_(std::move(value)) {}
    explicit Value(std::vector<Value> value) : type_(Type::Array), array_(std::move(value)) {}
    explicit Value(std::vector<std::pair<std::string, Value>> value)
        : type_(Type::Object), object_(std::move(value)) {}

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
    const std::vector<Value>& asArray() const { return array_; }
    const std::vector<std::pair<std::string, Value>>& asObject() const { return object_; }

    const Value* find(std::string_view key) const;
    bool has(std::string_view key) const { return find(key) != nullptr; }
    void set(std::string key, Value value);

private:
    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<Value> array_;
    std::vector<std::pair<std::string, Value>> object_;
};

Value parse(std::string_view input);
std::string typeName(Value::Type type);

}  // namespace lfb::json
