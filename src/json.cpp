#include "json.h"

#include <charconv>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace cloud::json {
namespace {

[[noreturn]] void typeError(const char* expected) {
    throw ParseError(std::string("JSON value is not ") + expected);
}

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    Value parseDocument() {
        skipWhitespace();
        Value value = parseValue();
        skipWhitespace();
        if (pos_ != text_.size()) {
            fail("trailing characters");
        }
        return value;
    }

private:
    const std::string& text_;
    std::size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& message) const {
        std::ostringstream out;
        out << "JSON parse error at byte " << pos_ << ": " << message;
        throw ParseError(out.str());
    }

    char peek() const {
        return pos_ < text_.size() ? text_[pos_] : '\0';
    }

    char take() {
        if (pos_ >= text_.size()) {
            fail("unexpected end of input");
        }
        return text_[pos_++];
    }

    void skipWhitespace() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
                break;
            }
            ++pos_;
        }
    }

    void expect(char expected) {
        if (take() != expected) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    Value parseValue() {
        switch (peek()) {
            case '{':
                return parseObject();
            case '[':
                return parseArray();
            case '"':
                return Value(parseString());
            case 't':
                parseLiteral("true");
                return Value(true);
            case 'f':
                parseLiteral("false");
                return Value(false);
            case 'n':
                parseLiteral("null");
                return Value(nullptr);
            default:
                if (peek() == '-' || (peek() >= '0' && peek() <= '9')) {
                    return Value(parseNumber());
                }
                fail("unexpected token");
        }
    }

    void parseLiteral(const char* literal) {
        for (const char* p = literal; *p != '\0'; ++p) {
            if (take() != *p) {
                fail(std::string("invalid literal '") + literal + "'");
            }
        }
    }

    Value parseObject() {
        expect('{');
        Value::Object object;
        skipWhitespace();
        if (peek() == '}') {
            ++pos_;
            return Value(std::move(object));
        }
        for (;;) {
            skipWhitespace();
            if (peek() != '"') {
                fail("object key must be a string");
            }
            std::string key = parseString();
            skipWhitespace();
            expect(':');
            skipWhitespace();
            auto [it, inserted] = object.emplace(key, parseValue());
            if (!inserted) {
                fail("duplicate object key '" + key + "'");
            }
            skipWhitespace();
            const char separator = take();
            if (separator == '}') {
                break;
            }
            if (separator != ',') {
                fail("expected ',' or '}'");
            }
        }
        return Value(std::move(object));
    }

    Value parseArray() {
        expect('[');
        Value::Array array;
        skipWhitespace();
        if (peek() == ']') {
            ++pos_;
            return Value(std::move(array));
        }
        for (;;) {
            skipWhitespace();
            array.push_back(parseValue());
            skipWhitespace();
            const char separator = take();
            if (separator == ']') {
                break;
            }
            if (separator != ',') {
                fail("expected ',' or ']'");
            }
        }
        return Value(std::move(array));
    }

    std::string parseString() {
        expect('"');
        std::string result;
        while (pos_ < text_.size()) {
            const unsigned char c = static_cast<unsigned char>(take());
            if (c == '"') {
                return result;
            }
            if (c < 0x20) {
                fail("control character in string");
            }
            if (c != '\\') {
                result.push_back(static_cast<char>(c));
                continue;
            }
            const char escaped = take();
            switch (escaped) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': {
                    const unsigned codepoint = parseHex4();
                    appendUtf8(result, codepoint);
                    break;
                }
                default:
                    fail("invalid string escape");
            }
        }
        fail("unterminated string");
    }

    unsigned parseHex4() {
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = take();
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<unsigned>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<unsigned>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<unsigned>(c - 'A' + 10);
            } else {
                fail("invalid Unicode escape");
            }
        }
        return value;
    }

    static void appendUtf8(std::string& out, unsigned codepoint) {
        if (codepoint <= 0x7f) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    double parseNumber() {
        const std::size_t start = pos_;
        if (peek() == '-') {
            ++pos_;
        }
        if (peek() == '0') {
            ++pos_;
        } else {
            if (peek() < '1' || peek() > '9') {
                fail("invalid number");
            }
            while (peek() >= '0' && peek() <= '9') {
                ++pos_;
            }
        }
        if (peek() == '.') {
            ++pos_;
            if (peek() < '0' || peek() > '9') {
                fail("invalid fraction");
            }
            while (peek() >= '0' && peek() <= '9') {
                ++pos_;
            }
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') {
                ++pos_;
            }
            if (peek() < '0' || peek() > '9') {
                fail("invalid exponent");
            }
            while (peek() >= '0' && peek() <= '9') {
                ++pos_;
            }
        }
        double value = 0.0;
        const char* begin = text_.data() + start;
        const char* end = text_.data() + pos_;
        const auto result = std::from_chars(begin, end, value, std::chars_format::general);
        if (result.ec != std::errc() || result.ptr != end || !std::isfinite(value)) {
            fail("number out of range");
        }
        return value;
    }
};

void dumpString(std::ostream& out, const std::string& value) {
    out << '"';
    for (const unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned>(c) << std::dec;
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    out << '"';
}

void dumpValue(std::ostream& out, const Value& value, bool pretty, int depth) {
    const auto indent = [&](int level) {
        if (pretty) {
            out << '\n' << std::string(static_cast<std::size_t>(level) * 2, ' ');
        }
    };
    if (value.isNull()) {
        out << "null";
    } else if (value.isBool()) {
        out << (value.asBool() ? "true" : "false");
    } else if (value.isNumber()) {
        const double number = value.asNumber();
        if (std::floor(number) == number && std::abs(number) < 9.0e15) {
            out << std::fixed << std::setprecision(0) << number << std::defaultfloat;
        } else {
            std::ostringstream numberOut;
            numberOut << std::setprecision(17) << number;
            out << numberOut.str();
        }
    } else if (value.isString()) {
        dumpString(out, value.asString());
    } else if (value.isArray()) {
        const auto& array = value.asArray();
        out << '[';
        for (std::size_t i = 0; i < array.size(); ++i) {
            if (i != 0) {
                out << ',';
            }
            if (pretty) {
                indent(depth + 1);
            }
            dumpValue(out, array[i], pretty, depth + 1);
        }
        if (pretty && !array.empty()) {
            indent(depth);
        }
        out << ']';
    } else {
        const auto& object = value.asObject();
        out << '{';
        std::size_t index = 0;
        for (const auto& [key, child] : object) {
            if (index++ != 0) {
                out << ',';
            }
            if (pretty) {
                indent(depth + 1);
            }
            dumpString(out, key);
            out << (pretty ? ": " : ":");
            dumpValue(out, child, pretty, depth + 1);
        }
        if (pretty && !object.empty()) {
            indent(depth);
        }
        out << '}';
    }
}

}  // namespace

bool Value::asBool() const {
    const auto* value = std::get_if<bool>(&value_);
    if (value == nullptr) {
        typeError("a boolean");
    }
    return *value;
}

double Value::asNumber() const {
    const auto* value = std::get_if<double>(&value_);
    if (value == nullptr) {
        typeError("a number");
    }
    return *value;
}

std::int64_t Value::asInt64() const {
    return static_cast<std::int64_t>(asNumber());
}

std::uint64_t Value::asUInt64() const {
    const double value = asNumber();
    if (value < 0.0) {
        throw ParseError("negative value cannot be read as unsigned");
    }
    return static_cast<std::uint64_t>(value);
}

const std::string& Value::asString() const {
    const auto* value = std::get_if<std::string>(&value_);
    if (value == nullptr) {
        typeError("a string");
    }
    return *value;
}

const Value::Array& Value::asArray() const {
    const auto* value = std::get_if<Array>(&value_);
    if (value == nullptr) {
        typeError("an array");
    }
    return *value;
}

const Value::Object& Value::asObject() const {
    const auto* value = std::get_if<Object>(&value_);
    if (value == nullptr) {
        typeError("an object");
    }
    return *value;
}

Value::Array& Value::asArray() {
    auto* value = std::get_if<Array>(&value_);
    if (value == nullptr) {
        typeError("an array");
    }
    return *value;
}

Value::Object& Value::asObject() {
    auto* value = std::get_if<Object>(&value_);
    if (value == nullptr) {
        typeError("an object");
    }
    return *value;
}

const Value* Value::find(const std::string& key) const {
    const auto* object = std::get_if<Object>(&value_);
    if (object == nullptr) {
        return nullptr;
    }
    const auto it = object->find(key);
    return it == object->end() ? nullptr : &it->second;
}

Value& Value::operator[](const std::string& key) {
    auto* object = std::get_if<Object>(&value_);
    if (object == nullptr) {
        value_ = Object{};
        object = std::get_if<Object>(&value_);
    }
    return (*object)[key];
}

Value parse(const std::string& text) {
    return Parser(text).parseDocument();
}

std::string stringify(const Value& value, bool pretty) {
    std::ostringstream out;
    dumpValue(out, value, pretty, 0);
    return out.str();
}

}  // namespace cloud::json

