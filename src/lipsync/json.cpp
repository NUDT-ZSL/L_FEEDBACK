#include "json.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace lipsync {
namespace {

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text) {}

    JsonParseResult run() {
        JsonParseResult result;
        skip_ws();
        if (pos_ >= text_.size()) {
            result.error = ProcessError(ErrorCode::E_PARSE_JSON, "empty JSON input");
            return result;
        }
        result.root = parse_value();
        if (failed_) {
            result.error = ProcessError(ErrorCode::E_PARSE_JSON, error_);
            return result;
        }
        skip_ws();
        if (pos_ != text_.size()) {
            result.error = ProcessError(ErrorCode::E_PARSE_JSON,
                                        fail_here("trailing characters after JSON value"));
        }
        return result;
    }

private:
    const std::string& text_;
    size_t pos_ = 0;
    bool failed_ = false;
    std::string error_;

    [[noreturn]] void fail(const std::string& msg) {
        failed_ = true;
        error_ = msg;
        throw std::runtime_error(msg);
    }

    std::string fail_here(const std::string& msg) const {
        size_t line = 1;
        size_t col = 1;
        for (size_t i = 0; i < pos_ && i < text_.size(); ++i) {
            if (text_[i] == '\n') {
                ++line;
                col = 1;
            } else {
                ++col;
            }
        }
        std::ostringstream oss;
        oss << msg << " at line " << line << " column " << col;
        return oss.str();
    }

    char peek() const {
        return pos_ < text_.size() ? text_[pos_] : '\0';
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

    void expect(char c) {
        if (pos_ >= text_.size() || text_[pos_] != c) {
            fail(fail_here(std::string("expected '") + c + "'"));
        }
        ++pos_;
    }

    JsonValue parse_value() {
        skip_ws();
        char c = peek();
        switch (c) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': {
                JsonValue v;
                v.type = JsonValue::Type::String;
                v.str = parse_string();
                return v;
            }
            case 't':
            case 'f': return parse_bool();
            case 'n': return parse_null();
            default:
                if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
                fail(fail_here("unexpected character"));
        }
    }

    JsonValue parse_object() {
        JsonValue v;
        v.type = JsonValue::Type::Object;
        expect('{');
        skip_ws();
        if (peek() == '}') {
            ++pos_;
            return v;
        }
        while (true) {
            skip_ws();
            if (peek() != '"') fail(fail_here("expected string key in object"));
            std::string key = parse_string();
            skip_ws();
            expect(':');
            JsonValue child = parse_value();
            v.object.emplace(std::move(key), std::move(child));
            skip_ws();
            char c = peek();
            if (c == ',') {
                ++pos_;
                continue;
            }
            if (c == '}') {
                ++pos_;
                return v;
            }
            fail(fail_here("expected ',' or '}' in object"));
        }
    }

    JsonValue parse_array() {
        JsonValue v;
        v.type = JsonValue::Type::Array;
        expect('[');
        skip_ws();
        if (peek() == ']') {
            ++pos_;
            return v;
        }
        while (true) {
            v.array.push_back(parse_value());
            skip_ws();
            char c = peek();
            if (c == ',') {
                ++pos_;
                continue;
            }
            if (c == ']') {
                ++pos_;
                return v;
            }
            fail(fail_here("expected ',' or ']' in array"));
        }
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (pos_ < text_.size()) {
            char c = text_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= text_.size()) break;
                char esc = text_[pos_++];
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
                        if (pos_ + 4 > text_.size()) fail(fail_here("bad unicode escape"));
                        unsigned code = 0;
                        for (int i = 0; i < 4; ++i) {
                            char h = text_[pos_++];
                            code <<= 4;
                            if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
                            else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
                            else fail(fail_here("bad hex digit in unicode escape"));
                        }
                        append_utf8(out, code);
                        break;
                    }
                    default: fail(fail_here("invalid escape sequence"));
                }
            } else {
                if (static_cast<unsigned char>(c) < 0x20) {
                    fail(fail_here("unescaped control character in string"));
                }
                out.push_back(c);
            }
        }
        fail(fail_here("unterminated string"));
    }

    static void append_utf8(std::string& out, unsigned code) {
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
    }

    JsonValue parse_bool() {
        if (text_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            JsonValue v;
            v.type = JsonValue::Type::Bool;
            v.boolean = true;
            return v;
        }
        if (text_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            JsonValue v;
            v.type = JsonValue::Type::Bool;
            return v;
        }
        fail(fail_here("invalid literal"));
    }

    JsonValue parse_null() {
        if (text_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            return JsonValue();
        }
        fail(fail_here("invalid literal"));
    }

    JsonValue parse_number() {
        size_t start = pos_;
        if (peek() == '-') ++pos_;
        consume_digits();
        if (peek() == '.') {
            ++pos_;
            consume_digits();
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') ++pos_;
            consume_digits();
        }
        std::string token = text_.substr(start, pos_ - start);
        JsonValue v;
        v.type = JsonValue::Type::Number;
        try {
            v.number = std::stod(token);
        } catch (...) {
            fail(fail_here("invalid number"));
        }
        return v;
    }

    void consume_digits() {
        size_t start = pos_;
        while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        if (pos_ == start) fail(fail_here("expected digits"));
    }
};

void append_escaped(std::string& out, const std::string& str) {
    out.push_back('"');
    for (char raw : str) {
        auto c = static_cast<unsigned char>(raw);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out.push_back('"');
}

void dump_number(std::string& out, double value) {
    if (std::isfinite(value) && value == std::floor(value) &&
        std::fabs(value) < 1e15) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld",
                      static_cast<long long>(value));
        out += buf;
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.10g", value);
        out += buf;
    }
}

void dump_value(std::string& out, const JsonValue& value, bool pretty, int depth) {
    auto indent = [&](int d) {
        if (pretty) {
            out.push_back('\n');
            out.append(static_cast<size_t>(d) * 2, ' ');
        }
    };
    switch (value.type) {
        case JsonValue::Type::Null: out += "null"; break;
        case JsonValue::Type::Bool: out += value.boolean ? "true" : "false"; break;
        case JsonValue::Type::Number: dump_number(out, value.number); break;
        case JsonValue::Type::String: append_escaped(out, value.str); break;
        case JsonValue::Type::Array: {
            if (value.array.empty()) {
                out += "[]";
                break;
            }
            out.push_back('[');
            for (size_t i = 0; i < value.array.size(); ++i) {
                if (i) out.push_back(',');
                indent(depth + 1);
                dump_value(out, value.array[i], pretty, depth + 1);
            }
            indent(depth);
            out.push_back(']');
            break;
        }
        case JsonValue::Type::Object: {
            if (value.object.empty()) {
                out += "{}";
                break;
            }
            out.push_back('{');
            size_t i = 0;
            for (const auto& kv : value.object) {
                if (i++) out.push_back(',');
                indent(depth + 1);
                append_escaped(out, kv.first);
                out.push_back(':');
                if (pretty) out.push_back(' ');
                dump_value(out, kv.second, pretty, depth + 1);
            }
            indent(depth);
            out.push_back('}');
            break;
        }
    }
}

} // namespace

JsonParseResult json_parse(const std::string& text) {
    try {
        return JsonParser(text).run();
    } catch (const std::exception&) {
        JsonParseResult result;
        if (!result.error) {
            result.error = ProcessError(ErrorCode::E_PARSE_JSON, "malformed JSON");
        }
        return result;
    }
}

JsonParseResult json_parse_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        JsonParseResult result;
        result.error = ProcessError(ErrorCode::E_IO, "cannot open file: " + path);
        return result;
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return json_parse(oss.str());
}

std::string json_dump(const JsonValue& value, bool pretty) {
    std::string out;
    dump_value(out, value, pretty, 0);
    if (pretty) out.push_back('\n');
    return out;
}

bool write_text_file(const std::string& path, const std::string& content,
                     std::string* error_message) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        if (error_message) *error_message = "cannot write file: " + path;
        return false;
    }
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!out) {
        if (error_message) *error_message = "write failed: " + path;
        return false;
    }
    return true;
}

} // namespace lipsync
