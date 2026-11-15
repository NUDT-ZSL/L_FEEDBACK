#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace vlr::json {

class Value {
public:
    using Array = std::vector<Value>;
    using Object = std::map<std::string, Value>;

    enum class Type { Null, Bool, Number, String, Array, Object };

    Value() = default;
    explicit Value(bool value) : value_(value) {}
    explicit Value(double value) : value_(value) {}
    explicit Value(std::string value) : value_(std::move(value)) {}
    explicit Value(Array value) : value_(std::move(value)) {}
    explicit Value(Object value) : value_(std::move(value)) {}

    [[nodiscard]] Type type() const noexcept { return static_cast<Type>(value_.index()); }
    [[nodiscard]] bool isNull() const noexcept { return type() == Type::Null; }
    [[nodiscard]] bool isBool() const noexcept { return type() == Type::Bool; }
    [[nodiscard]] bool isNumber() const noexcept { return type() == Type::Number; }
    [[nodiscard]] bool isString() const noexcept { return type() == Type::String; }
    [[nodiscard]] bool isArray() const noexcept { return type() == Type::Array; }
    [[nodiscard]] bool isObject() const noexcept { return type() == Type::Object; }

    [[nodiscard]] bool asBool() const { return std::get<bool>(value_); }
    [[nodiscard]] double asNumber() const { return std::get<double>(value_); }
    [[nodiscard]] const std::string& asString() const { return std::get<std::string>(value_); }
    [[nodiscard]] const Array& asArray() const { return std::get<Array>(value_); }
    [[nodiscard]] const Object& asObject() const { return std::get<Object>(value_); }

    [[nodiscard]] const Value* find(std::string_view key) const;

private:
    std::variant<std::monostate, bool, double, std::string, Array, Object> value_;
};

Value parse(std::string_view text);
std::string stringify(const Value& value, int indent = 2);
std::string escape(std::string_view text);

} // namespace vlr::json
