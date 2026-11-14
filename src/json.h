#pragma once

#include <initializer_list>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace vr {

class Json {
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
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;

    static Json make_null();
    static Json make_boolean(bool value);
    static Json make_number(double value);
    static Json make_string(std::string value);
    static Json make_array();
    static Json make_object();

    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }
    bool is_string() const { return type == Type::String; }
    bool is_number() const { return type == Type::Number; }
    bool is_boolean() const { return type == Type::Boolean; }

    const Json* find(std::string_view key) const;
    const Json* find_any(std::initializer_list<std::string_view> keys) const;
    void append(std::pair<std::string, Json> value);
};

bool parse_json(std::string_view text, Json& out, std::string& error);
std::string json_quote(std::string_view value);
std::string json_number(double value);

}  // namespace vr

