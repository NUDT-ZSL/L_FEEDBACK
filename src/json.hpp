#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace orv {

class JsonValue {
 public:
  using Number = double;
  using Array = std::vector<JsonValue>;
  using Object = std::map<std::string, JsonValue>;

  enum class Type { Null, Bool, Number, String, Array, Object };

  JsonValue() : value_(nullptr) {}
  explicit JsonValue(bool value) : value_(value) {}
  explicit JsonValue(double value) : value_(value) {}
  explicit JsonValue(std::string value) : value_(std::move(value)) {}
  explicit JsonValue(Array value) : value_(std::move(value)) {}
  explicit JsonValue(Object value) : value_(std::move(value)) {}

  Type type() const { return static_cast<Type>(value_.index()); }
  bool isNull() const { return type() == Type::Null; }
  bool isBool() const { return type() == Type::Bool; }
  bool isNumber() const { return type() == Type::Number; }
  bool isString() const { return type() == Type::String; }
  bool isArray() const { return type() == Type::Array; }
  bool isObject() const { return type() == Type::Object; }

  bool asBool() const { return std::get<bool>(value_); }
  double asNumber() const { return std::get<double>(value_); }
  const std::string& asString() const { return std::get<std::string>(value_); }
  const Array& asArray() const { return std::get<Array>(value_); }
  const Object& asObject() const { return std::get<Object>(value_); }

  const JsonValue* find(const std::string& key) const {
    if (!isObject()) {
      return nullptr;
    }
    const auto& object = asObject();
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
  }

 private:
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value_;
};

struct JsonParseError {
  std::string message;
  std::size_t line = 1;
  std::size_t column = 1;
};

bool parseJson(const std::string& text, JsonValue& value, JsonParseError& error);

std::string jsonEscape(const std::string& input);
std::string jsonNumber(double value, int precision = 9);

}  // namespace orv
