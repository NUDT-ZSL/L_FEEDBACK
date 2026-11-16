#include "json.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>

namespace preview::json {
namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    Value parseDocument() {
        if (text_.size() >= 3 &&
            static_cast<unsigned char>(text_[0]) == 0xEF &&
            static_cast<unsigned char>(text_[1]) == 0xBB &&
            static_cast<unsigned char>(text_[2]) == 0xBF) {
            position_ = 3;
        }
        skipWhitespace();
        Value value = parseValue(0);
        skipWhitespace();
        if (!atEnd()) {
            fail("unexpected trailing data");
        }
        return value;
    }

private:
    static constexpr int kMaxDepth = 128;

    const std::string& text_;
    std::size_t position_ = 0;

    bool atEnd() const noexcept { return position_ >= text_.size(); }

    char peek() const { return atEnd() ? '\0' : text_[position_]; }

    char take() {
        if (atEnd()) {
            fail("unexpected end of input");
        }
        return text_[position_++];
    }

    [[noreturn]] void fail(const std::string& message) const {
        std::size_t line = 1;
        std::size_t column = 1;
        for (std::size_t i = 0; i < position_ && i < text_.size(); ++i) {
            if (text_[i] == '\n') {
                ++line;
                column = 1;
            } else {
                ++column;
            }
        }
        std::ostringstream out;
        out << "JSON parse error at line " << line << ", column " << column << ": " << message;
        throw ParseError(out.str());
    }

    void skipWhitespace() {
        while (!atEnd()) {
            const char c = peek();
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++position_;
            } else {
                break;
            }
        }
    }

    Value parseValue(int depth) {
        if (depth > kMaxDepth) {
            fail("maximum nesting depth exceeded");
        }
        if (atEnd()) {
            fail("expected a value");
        }

        switch (peek()) {
        case '{':
            return parseObject(depth + 1);
        case '[':
            return parseArray(depth + 1);
        case '"': {
            Value value;
            value.type = Value::Type::String;
            value.string = parseString();
            return value;
        }
        case 't':
            consumeLiteral("true");
            return makeBoolean(true);
        case 'f':
            consumeLiteral("false");
            return makeBoolean(false);
        case 'n':
            consumeLiteral("null");
            return Value{};
        default:
            if (peek() == '-' || (peek() >= '0' && peek() <= '9')) {
                return parseNumber();
            }
            fail("unexpected character");
        }
    }

    static Value makeBoolean(bool value) {
        Value result;
        result.type = Value::Type::Boolean;
        result.boolean = value;
        return result;
    }

    void consumeLiteral(const char* literal) {
        for (const char* cursor = literal; *cursor != '\0'; ++cursor) {
            if (atEnd() || take() != *cursor) {
                fail(std::string("expected literal '") + literal + "'");
            }
        }
    }

    Value parseObject(int depth) {
        Value value;
        value.type = Value::Type::Object;
        take();
        skipWhitespace();
        if (peek() == '}') {
            take();
            return value;
        }

        while (true) {
            if (peek() != '"') {
                fail("expected an object key");
            }
            const std::string key = parseString();
            skipWhitespace();
            if (take() != ':') {
                fail("expected ':' after object key");
            }
            skipWhitespace();
            Value member = parseValue(depth);
            if (value.object.find(key) != value.object.end()) {
                fail("duplicate object key '" + key + "'");
            }
            value.object.emplace(key, std::move(member));
            skipWhitespace();
            const char separator = take();
            if (separator == '}') {
                return value;
            }
            if (separator != ',') {
                fail("expected ',' or '}' in object");
            }
            skipWhitespace();
        }
    }

    Value parseArray(int depth) {
        Value value;
        value.type = Value::Type::Array;
        take();
        skipWhitespace();
        if (peek() == ']') {
            take();
            return value;
        }

        while (true) {
            value.array.push_back(parseValue(depth));
            skipWhitespace();
            const char separator = take();
            if (separator == ']') {
                return value;
            }
            if (separator != ',') {
                fail("expected ',' or ']' in array");
            }
            skipWhitespace();
        }
    }

    static void appendUtf8(std::string& output, std::uint32_t codePoint) {
        if (codePoint <= 0x7F) {
            output.push_back(static_cast<char>(codePoint));
        } else if (codePoint <= 0x7FF) {
            output.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
            output.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
        } else if (codePoint <= 0xFFFF) {
            output.push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
            output.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
        } else {
            output.push_back(static_cast<char>(0xF0 | (codePoint >> 18)));
            output.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
        }
    }

    std::uint32_t parseHex4() {
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = take();
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<std::uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<std::uint32_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<std::uint32_t>(c - 'A' + 10);
            } else {
                fail("invalid unicode escape");
            }
        }
        return value;
    }

    std::string parseString() {
        if (take() != '"') {
            fail("expected a string");
        }

        std::string result;
        while (!atEnd()) {
            const char c = take();
            if (c == '"') {
                return result;
            }
            if (static_cast<unsigned char>(c) < 0x20) {
                fail("unescaped control character in string");
            }
            if (c != '\\') {
                result.push_back(c);
                continue;
            }

            const char escaped = take();
            switch (escaped) {
            case '"':
                result.push_back('"');
                break;
            case '\\':
                result.push_back('\\');
                break;
            case '/':
                result.push_back('/');
                break;
            case 'b':
                result.push_back('\b');
                break;
            case 'f':
                result.push_back('\f');
                break;
            case 'n':
                result.push_back('\n');
                break;
            case 'r':
                result.push_back('\r');
                break;
            case 't':
                result.push_back('\t');
                break;
            case 'u': {
                std::uint32_t codePoint = parseHex4();
                if (codePoint >= 0xD800 && codePoint <= 0xDBFF) {
                    if (take() != '\\' || take() != 'u') {
                        fail("missing low surrogate");
                    }
                    const std::uint32_t low = parseHex4();
                    if (low < 0xDC00 || low > 0xDFFF) {
                        fail("invalid low surrogate");
                    }
                    codePoint = 0x10000 + ((codePoint - 0xD800) << 10) + (low - 0xDC00);
                } else if (codePoint >= 0xDC00 && codePoint <= 0xDFFF) {
                    fail("unexpected low surrogate");
                }
                appendUtf8(result, codePoint);
                break;
            }
            default:
                fail("invalid string escape");
            }
        }
        fail("unterminated string");
    }

    Value parseNumber() {
        const std::size_t start = position_;
        if (peek() == '-') {
            ++position_;
        }

        if (atEnd()) {
            fail("incomplete number");
        }
        if (peek() == '0') {
            ++position_;
            if (!atEnd() && peek() >= '0' && peek() <= '9') {
                fail("leading zero in number");
            }
        } else if (peek() >= '1' && peek() <= '9') {
            while (!atEnd() && peek() >= '0' && peek() <= '9') {
                ++position_;
            }
        } else {
            fail("invalid number");
        }

        if (!atEnd() && peek() == '.') {
            ++position_;
            if (atEnd() || peek() < '0' || peek() > '9') {
                fail("missing fractional digits");
            }
            while (!atEnd() && peek() >= '0' && peek() <= '9') {
                ++position_;
            }
        }

        if (!atEnd() && (peek() == 'e' || peek() == 'E')) {
            ++position_;
            if (!atEnd() && (peek() == '+' || peek() == '-')) {
                ++position_;
            }
            if (atEnd() || peek() < '0' || peek() > '9') {
                fail("missing exponent digits");
            }
            while (!atEnd() && peek() >= '0' && peek() <= '9') {
                ++position_;
            }
        }

        double number = 0.0;
        const char* first = text_.data() + start;
        const char* last = text_.data() + position_;
        const auto conversion = std::from_chars(first, last, number, std::chars_format::general);
        if (conversion.ec != std::errc() || conversion.ptr != last || !std::isfinite(number)) {
            fail("number is not finite");
        }

        Value value;
        value.type = Value::Type::Number;
        value.number = number;
        return value;
    }
};

} // namespace

Value parse(const std::string& text) {
    return Parser(text).parseDocument();
}

} // namespace preview::json
