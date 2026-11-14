#include "json.h"

#include <cmath>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace vr {

namespace {

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    bool parse(Json& out, std::string& error) {
        skip_ws();
        if (!parse_value(out, error)) {
            return false;
        }
        skip_ws();
        if (pos_ != text_.size()) {
            fail("unexpected trailing content");
            error = error_;
            return false;
        }
        return true;
    }

private:
    void fail(const std::string& message) {
        std::ostringstream os;
        os << message << " at byte " << pos_;
        error_ = os.str();
    }

    void skip_ws() {
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    bool consume(char expected) {
        skip_ws();
        if (pos_ >= text_.size() || text_[pos_] != expected) {
            fail(std::string("expected '") + expected + "'");
            return false;
        }
        ++pos_;
        return true;
    }

    bool parse_value(Json& out, std::string& error) {
        skip_ws();
        if (pos_ >= text_.size()) {
            fail("expected a JSON value");
            error = error_;
            return false;
        }
        const char c = text_[pos_];
        if (c == '{') {
            return parse_object(out, error);
        }
        if (c == '[') {
            return parse_array(out, error);
        }
        if (c == '"') {
            out = Json::make_string("");
            if (!parse_string(out.string, error)) {
                return false;
            }
            out.type = Json::Type::String;
            return true;
        }
        if (c == 't' || c == 'f' || c == 'n') {
            return parse_literal(out, error);
        }
        if (c == '-' || (c >= '0' && c <= '9') || c == 'N' || c == 'I') {
            return parse_number(out, error);
        }
        fail("expected a JSON value");
        error = error_;
        return false;
    }

    bool parse_literal(Json& out, std::string& error) {
        if (text_.substr(pos_, 4) == "true") {
            pos_ += 4;
            out = Json::make_boolean(true);
            return true;
        }
        if (text_.substr(pos_, 5) == "false") {
            pos_ += 5;
            out = Json::make_boolean(false);
            return true;
        }
        if (text_.substr(pos_, 4) == "null") {
            pos_ += 4;
            out = Json::make_null();
            return true;
        }
        fail("invalid literal");
        error = error_;
        return false;
    }

    bool parse_number(Json& out, std::string& error) {
        const std::size_t begin = pos_;
        if (text_.substr(pos_, 3) == "NaN") {
            pos_ += 3;
            out = Json::make_number(std::numeric_limits<double>::quiet_NaN());
            return true;
        }
        if (text_.substr(pos_, 8) == "Infinity") {
            pos_ += 8;
            out = Json::make_number(std::numeric_limits<double>::infinity());
            return true;
        }
        if (text_.substr(pos_, 9) == "-Infinity") {
            pos_ += 9;
            out = Json::make_number(-std::numeric_limits<double>::infinity());
            return true;
        }
        if (text_[pos_] == '-') {
            ++pos_;
        }
        if (pos_ >= text_.size()) {
            fail("invalid number");
            error = error_;
            return false;
        }
        if (text_[pos_] == '0') {
            ++pos_;
        } else if (text_[pos_] >= '1' && text_[pos_] <= '9') {
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
                ++pos_;
            }
        } else {
            fail("invalid number");
            error = error_;
            return false;
        }
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
                fail("invalid fraction");
                error = error_;
                return false;
            }
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
                ++pos_;
            }
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
                fail("invalid exponent");
                error = error_;
                return false;
            }
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
                ++pos_;
            }
        }

        const std::string token(text_.substr(begin, pos_ - begin));
        errno = 0;
        char* end = nullptr;
        const double value = std::strtod(token.c_str(), &end);
        if (end == nullptr || *end != '\0') {
            fail("invalid number");
            error = error_;
            return false;
        }
        out = Json::make_number(value);
        return true;
    }

    bool parse_string(std::string& out, std::string& error) {
        if (!consume('"')) {
            error = error_;
            return false;
        }
        out.clear();
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') {
                return true;
            }
            if (static_cast<unsigned char>(c) < 0x20) {
                fail("control character in string");
                error = error_;
                return false;
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (pos_ >= text_.size()) {
                fail("unterminated escape");
                error = error_;
                return false;
            }
            const char esc = text_[pos_++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    unsigned code = 0;
                    for (int i = 0; i < 4; ++i) {
                        if (pos_ >= text_.size()) {
                            fail("short unicode escape");
                            error = error_;
                            return false;
                        }
                        const char h = text_[pos_++];
                        code <<= 4;
                        if (h >= '0' && h <= '9') {
                            code |= static_cast<unsigned>(h - '0');
                        } else if (h >= 'a' && h <= 'f') {
                            code |= static_cast<unsigned>(h - 'a' + 10);
                        } else if (h >= 'A' && h <= 'F') {
                            code |= static_cast<unsigned>(h - 'A' + 10);
                        } else {
                            fail("invalid unicode escape");
                            error = error_;
                            return false;
                        }
                    }
                    if (code >= 0xD800 && code <= 0xDFFF) {
                        fail("surrogate code points are not supported");
                        error = error_;
                        return false;
                    }
                    if (code < 0x80) {
                        out.push_back(static_cast<char>(code));
                    } else if (code < 0x800) {
                        out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    } else {
                        out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    }
                    break;
                }
                default:
                    fail("invalid escape");
                    error = error_;
                    return false;
            }
        }
        fail("unterminated string");
        error = error_;
        return false;
    }

    bool parse_array(Json& out, std::string& error) {
        if (!consume('[')) {
            error = error_;
            return false;
        }
        out = Json::make_array();
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == ']') {
            ++pos_;
            return true;
        }
        while (true) {
            Json value;
            if (!parse_value(value, error)) {
                return false;
            }
            out.array.push_back(std::move(value));
            skip_ws();
            if (pos_ >= text_.size()) {
                fail("unterminated array");
                error = error_;
                return false;
            }
            if (text_[pos_] == ']') {
                ++pos_;
                return true;
            }
            if (text_[pos_] != ',') {
                fail("expected ',' or ']'");
                error = error_;
                return false;
            }
            ++pos_;
            skip_ws();
        }
    }

    bool parse_object(Json& out, std::string& error) {
        if (!consume('{')) {
            error = error_;
            return false;
        }
        out = Json::make_object();
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == '}') {
            ++pos_;
            return true;
        }
        while (true) {
            std::string key;
            if (!parse_string(key, error)) {
                return false;
            }
            if (!consume(':')) {
                error = error_;
                return false;
            }
            Json value;
            if (!parse_value(value, error)) {
                return false;
            }
            for (const auto& existing : out.object) {
                if (existing.first == key) {
                    fail("duplicate object key");
                    error = error_;
                    return false;
                }
            }
            out.object.emplace_back(std::move(key), std::move(value));
            skip_ws();
            if (pos_ >= text_.size()) {
                fail("unterminated object");
                error = error_;
                return false;
            }
            if (text_[pos_] == '}') {
                ++pos_;
                return true;
            }
            if (text_[pos_] != ',') {
                fail("expected ',' or '}'");
                error = error_;
                return false;
            }
            ++pos_;
            skip_ws();
        }
    }

    std::string_view text_;
    std::size_t pos_ = 0;
    std::string error_;
};

}  // namespace

Json Json::make_null() {
    Json value;
    value.type = Type::Null;
    return value;
}

Json Json::make_boolean(bool input) {
    Json value;
    value.type = Type::Boolean;
    value.boolean = input;
    return value;
}

Json Json::make_number(double input) {
    Json value;
    value.type = Type::Number;
    value.number = input;
    return value;
}

Json Json::make_string(std::string input) {
    Json value;
    value.type = Type::String;
    value.string = std::move(input);
    return value;
}

Json Json::make_array() {
    Json value;
    value.type = Type::Array;
    return value;
}

Json Json::make_object() {
    Json value;
    value.type = Type::Object;
    return value;
}

const Json* Json::find(std::string_view key) const {
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

const Json* Json::find_any(std::initializer_list<std::string_view> keys) const {
    for (const auto key : keys) {
        if (const Json* value = find(key)) {
            return value;
        }
    }
    return nullptr;
}

void Json::append(std::pair<std::string, Json> value) {
    type = Type::Object;
    object.push_back(std::move(value));
}

bool parse_json(std::string_view text, Json& out, std::string& error) {
    Parser parser(text);
    return parser.parse(out, error);
}

std::string json_quote(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (const char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04X",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
    return out;
}

std::string json_number(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    if (value == 0.0) {
        return "0";
    }
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                      std::chars_format::general, 17);
    if (result.ec != std::errc()) {
        return "null";
    }
    std::string text(buffer, result.ptr);
    if (text.find_first_of(".eE") == std::string::npos) {
        text += ".0";
    }
    return text;
}

}  // namespace vr

