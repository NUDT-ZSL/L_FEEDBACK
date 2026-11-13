#pragma once

#include <map>
#include <string>
#include <vector>

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;

    JsonValue() = default;
    explicit JsonValue(bool value) : type_(Type::Bool), bool_(value) {}
    explicit JsonValue(double value) : type_(Type::Number), number_(value) {}
    explicit JsonValue(std::string value) : type_(Type::String), string_(std::move(value)) {}
    explicit JsonValue(Array value) : type_(Type::Array), array_(std::move(value)) {}
    explicit JsonValue(Object value) : type_(Type::Object), object_(std::move(value)) {}

    Type type() const { return type_; }
    bool isNull() const { return type_ == Type::Null; }
    bool isBool() const { return type_ == Type::Bool; }
    bool isNumber() const { return type_ == Type::Number; }
    bool isString() const { return type_ == Type::String; }
    bool isArray() const { return type_ == Type::Array; }
    bool isObject() const { return type_ == Type::Object; }

    bool asBool() const { return bool_; }
    double asNumber() const { return number_; }
    const std::string& asString() const { return string_; }
    const Array& asArray() const { return array_; }
    const Object& asObject() const { return object_; }
    Array& asArray() { return array_; }
    Object& asObject() { return object_; }

    const JsonValue* find(const std::string& key) const;
    bool contains(const std::string& key) const;
    void set(const std::string& key, JsonValue value);
    void push(JsonValue value);

    static JsonValue parse(const std::string& text);
    static JsonValue parseFile(const std::string& path);

private:
    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    Array array_;
    Object object_;

    friend class JsonParser;
};

std::string jsonEscape(const std::string& text);
