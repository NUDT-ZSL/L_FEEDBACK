#include "json.h"

#include "core.h"

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>

const JsonValue* JsonValue::find(const std::string& key) const {
    if (!isObject()) return nullptr;
    const auto it = object_.find(key);
    return it == object_.end() ? nullptr : &it->second;
}

bool JsonValue::contains(const std::string& key) const {
    return find(key) != nullptr;
}

void JsonValue::set(const std::string& key, JsonValue value) {
    type_ = Type::Object;
    object_[key] = std::move(value);
}

void JsonValue::push(JsonValue value) {
    type_ = Type::Array;
    array_.push_back(std::move(value));
}

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text) {}

    JsonValue parse() {
        skipWhitespace();
        JsonValue value = parseValue();
        skipWhitespace();
        if (pos_ != text_.size()) {
            fail("trailing characters after JSON value");
        }
        return value;
    }

private:
    const std::string& text_;
    size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& message) const {
        throw RenderError("E_SCENE_INVALID",
                          "JSON parse error at byte " + std::to_string(pos_) + ": " + message);
    }

    void skipWhitespace() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    char peek() const {
        return pos_ < text_.size() ? text_[pos_] : '\0';
    }

    char take() {
        if (pos_ >= text_.size()) fail("unexpected end of input");
        return text_[pos_++];
    }

    void expect(char c) {
        if (peek() != c) {
            fail(std::string("expected '") + c + "'");
        }
        ++pos_;
    }

    JsonValue parseValue() {
        skipWhitespace();
        const char c = peek();
        switch (c) {
            case '{': return parseObject();
            case '[': return parseArray();
            case '"': return JsonValue(parseString());
            case 't': parseLiteral("true"); return JsonValue(true);
            case 'f': parseLiteral("false"); return JsonValue(false);
            case 'n': parseLiteral("null"); return JsonValue();
            default:
                if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
                fail("unexpected token");
        }
    }

    void parseLiteral(const char* literal) {
        for (const char* p = literal; *p; ++p) {
            if (take() != *p) fail("invalid literal");
        }
    }

    JsonValue parseObject() {
        expect('{');
        JsonValue::Object object;
        skipWhitespace();
        if (peek() == '}') {
            ++pos_;
            return JsonValue(std::move(object));
        }
        while (true) {
            skipWhitespace();
            if (peek() != '"') fail("object key must be a string");
            std::string key = parseString();
            skipWhitespace();
            expect(':');
            JsonValue value = parseValue();
            object[std::move(key)] = std::move(value);
            skipWhitespace();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect('}');
            break;
        }
        return JsonValue(std::move(object));
    }

    JsonValue parseArray() {
        expect('[');
        JsonValue::Array array;
        skipWhitespace();
        if (peek() == ']') {
            ++pos_;
            return JsonValue(std::move(array));
        }
        while (true) {
            array.push_back(parseValue());
            skipWhitespace();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect(']');
            break;
        }
        return JsonValue(std::move(array));
    }

    std::string parseString() {
        expect('"');
        std::string result;
        while (true) {
            const char c = take();
            if (c == '"') break;
            if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
            if (c != '\\') {
                result.push_back(c);
                continue;
            }
            const char esc = take();
            switch (esc) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': {
                    const unsigned code = parseHex4();
                    // The sample scenes are ASCII. Preserve basic BMP/UTF-8 for messages/names.
                    if (code <= 0x7f) {
                        result.push_back(static_cast<char>(code));
                    } else if (code <= 0x7ff) {
                        result.push_back(static_cast<char>(0xc0 | (code >> 6)));
                        result.push_back(static_cast<char>(0x80 | (code & 0x3f)));
                    } else {
                        result.push_back(static_cast<char>(0xe0 | (code >> 12)));
                        result.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
                        result.push_back(static_cast<char>(0x80 | (code & 0x3f)));
                    }
                    break;
                }
                default: fail("invalid string escape");
            }
        }
        return result;
    }

    unsigned parseHex4() {
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = take();
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<unsigned>(c - 'A' + 10);
            else fail("invalid unicode escape");
        }
        return value;
    }

    JsonValue parseNumber() {
        const size_t start = pos_;
        if (peek() == '-') ++pos_;
        if (peek() == '0') {
            ++pos_;
        } else if (peek() >= '1' && peek() <= '9') {
            while (peek() >= '0' && peek() <= '9') ++pos_;
        } else {
            fail("invalid number");
        }
        if (peek() == '.') {
            ++pos_;
            if (!(peek() >= '0' && peek() <= '9')) fail("invalid fraction");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') ++pos_;
            if (!(peek() >= '0' && peek() <= '9')) fail("invalid exponent");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }
        const std::string token = text_.substr(start, pos_ - start);
        errno = 0;
        char* end = nullptr;
        const double value = std::strtod(token.c_str(), &end);
        if (end == nullptr || *end != '\0') fail("invalid numeric token");
        // Overflow to infinity is intentional here: transform validation reports
        // E_TRANSFORM_INVALID rather than conflating it with malformed JSON.
        return JsonValue(value);
    }
};

JsonValue JsonValue::parse(const std::string& text) {
    JsonParser parser(text);
    return parser.parse();
}

JsonValue JsonValue::parseFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw RenderError("E_SCENE_INVALID", "cannot open scene file: " + path);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    if (!in.good() && !in.eof()) {
        throw RenderError("E_SCENE_INVALID", "cannot read scene file: " + path);
    }
    return parse(buffer.str());
}

std::string jsonEscape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
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
                    static const char* digits = "0123456789abcdef";
                    out += "\\u00";
                    out.push_back(digits[(static_cast<unsigned char>(c) >> 4) & 0xf]);
                    out.push_back(digits[static_cast<unsigned char>(c) & 0xf]);
                } else {
                    out.push_back(c);
                }
        }
    }
    return out;
}
