#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace ct::json {

class Value {
public:
    enum class Type {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object,
    };

    Value() = default;

    static Value parse(std::string_view text, std::string& error);

    Type type() const { return type_; }
    bool isNull() const { return type_ == Type::Null; }
    bool isBool() const { return type_ == Type::Bool; }
    bool isNumber() const { return type_ == Type::Number; }
    bool isString() const { return type_ == Type::String; }
    bool isArray() const { return type_ == Type::Array; }
    bool isObject() const { return type_ == Type::Object; }

    bool asBool() const { return bool_value_; }
    double asNumber() const { return number_value_; }
    const std::string& asString() const { return string_value_; }
    const std::vector<Value>& asArray() const { return array_value_; }
    const std::map<std::string, Value>& asObject() const { return object_value_; }

    const Value* find(const std::string& key) const {
        if (type_ != Type::Object) {
            return nullptr;
        }
        const auto it = object_value_.find(key);
        return it == object_value_.end() ? nullptr : &it->second;
    }

private:
    friend class Parser;

    Type type_ = Type::Null;
    bool bool_value_ = false;
    double number_value_ = 0.0;
    std::string string_value_;
    std::vector<Value> array_value_;
    std::map<std::string, Value> object_value_;
};

}  // namespace ct::json
