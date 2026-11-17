#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raster {

class JsonError : public std::runtime_error {
public:
    explicit JsonError(const std::string& message) : std::runtime_error(message) {}
};

class JsonValue {
public:
    enum class Type {
        Null,
        Boolean,
        Number,
        String,
        Array,
        Object,
    };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object;

    bool isNull() const { return type == Type::Null; }
    bool isBoolean() const { return type == Type::Boolean; }
    bool isNumber() const { return type == Type::Number; }
    bool isString() const { return type == Type::String; }
    bool isArray() const { return type == Type::Array; }
    bool isObject() const { return type == Type::Object; }

    const JsonValue* find(std::string_view key) const {
        if (type != Type::Object) {
            return nullptr;
        }
        for (const auto& entry : object) {
            if (entry.first == key) {
                return &entry.second;
            }
        }
        return nullptr;
    }
};

JsonValue parseJson(std::string_view source);

}  // namespace raster

