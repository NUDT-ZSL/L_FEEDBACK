#include "raster/json.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <system_error>

namespace raster {
namespace {

class Parser {
public:
    explicit Parser(std::string_view source) : source_(source) {}

    JsonValue parse() {
        skipWhitespace();
        JsonValue root = parseValue(0);
        skipWhitespace();
        if (position_ != source_.size()) {
            fail("unexpected trailing characters");
        }
        return root;
    }

private:
    static constexpr int kMaxDepth = 256;

    std::string_view source_;
    std::size_t position_ = 0;

    [[noreturn]] void fail(const std::string& message) const {
        std::size_t line = 1;
        std::size_t column = 1;
        for (std::size_t i = 0; i < position_ && i < source_.size(); ++i) {
            if (source_[i] == '\n') {
                ++line;
                column = 1;
            } else {
                ++column;
            }
        }
        throw JsonError("JSON error at line " + std::to_string(line) +
                        ", column " + std::to_string(column) + ": " + message);
    }

    void skipWhitespace() {
        while (position_ < source_.size()) {
            const char c = source_[position_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++position_;
            } else {
                break;
            }
        }
    }

    char peek() const {
        if (position_ >= source_.size()) {
            return '\0';
        }
        return source_[position_];
    }

    bool consume(char expected) {
        if (peek() == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    JsonValue parseValue(int depth) {
        if (depth > kMaxDepth) {
            fail("maximum JSON nesting depth exceeded");
        }
        if (position_ >= source_.size()) {
            fail("unexpected end of input");
        }

        switch (peek()) {
            case '{':
                return parseObject(depth + 1);
            case '[':
                return parseArray(depth + 1);
            case '"': {
                JsonValue value;
                value.type = JsonValue::Type::String;
                value.string = parseString();
                return value;
            }
            case 't':
                parseLiteral("true");
                {
                    JsonValue value;
                    value.type = JsonValue::Type::Boolean;
                    value.boolean = true;
                    return value;
                }
            case 'f':
                parseLiteral("false");
                {
                    JsonValue value;
                    value.type = JsonValue::Type::Boolean;
                    value.boolean = false;
                    return value;
                }
            case 'n':
                parseLiteral("null");
                return JsonValue{};
            default:
                if (peek() == '-' || (peek() >= '0' && peek() <= '9')) {
                    return parseNumber();
                }
                fail("unexpected token");
        }
    }

    void parseLiteral(std::string_view literal) {
        if (source_.substr(position_, literal.size()) != literal) {
            fail("invalid literal");
        }
        position_ += literal.size();
    }

    JsonValue parseNumber() {
        const std::size_t start = position_;
        if (consume('-')) {
            if (position_ == source_.size()) {
                fail("incomplete number");
            }
        }

        if (consume('0')) {
            if (position_ < source_.size() && peek() >= '0' && peek() <= '9') {
                fail("leading zeros are not allowed");
            }
        } else {
            if (peek() < '1' || peek() > '9') {
                fail("expected a digit");
            }
            while (position_ < source_.size() && peek() >= '0' &&
                   peek() <= '9') {
                ++position_;
            }
        }

        if (consume('.')) {
            if (position_ == source_.size() || peek() < '0' || peek() > '9') {
                fail("expected a digit after decimal point");
            }
            while (position_ < source_.size() && peek() >= '0' &&
                   peek() <= '9') {
                ++position_;
            }
        }

        if (peek() == 'e' || peek() == 'E') {
            ++position_;
            if (peek() == '+' || peek() == '-') {
                ++position_;
            }
            if (position_ == source_.size() || peek() < '0' || peek() > '9') {
                fail("expected a digit in exponent");
            }
            while (position_ < source_.size() && peek() >= '0' &&
                   peek() <= '9') {
                ++position_;
            }
        }

        const std::string_view text = source_.substr(start, position_ - start);
        double value = 0.0;
        const auto result =
            std::from_chars(text.data(), text.data() + text.size(), value);
        if (result.ec == std::errc::result_out_of_range ||
            !std::isfinite(value)) {
            fail("number is not finite");
        }
        if (result.ec != std::errc{} ||
            result.ptr != text.data() + text.size()) {
            fail("invalid number");
        }

        JsonValue parsed;
        parsed.type = JsonValue::Type::Number;
        parsed.number = value;
        return parsed;
    }

    static void appendUtf8(std::string& output, std::uint32_t codepoint) {
        if (codepoint <= 0x7fU) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ffU) {
            output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0xffffU) {
            output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
            output.push_back(
                static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else {
            output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
            output.push_back(
                static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
            output.push_back(
                static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        }
    }

    std::uint32_t parseHex4() {
        if (position_ + 4 > source_.size()) {
            fail("incomplete unicode escape");
        }
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = source_[position_++];
            value <<= 4U;
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
        expect('"');
        std::string result;
        while (position_ < source_.size()) {
            const unsigned char c =
                static_cast<unsigned char>(source_[position_++]);
            if (c == '"') {
                return result;
            }
            if (c == '\\') {
                if (position_ == source_.size()) {
                    fail("incomplete escape sequence");
                }
                const char escaped = source_[position_++];
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
                        std::uint32_t codepoint = parseHex4();
                        if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
                            if (source_.substr(position_, 2) != "\\u") {
                                fail("missing low surrogate");
                            }
                            position_ += 2;
                            const std::uint32_t low = parseHex4();
                            if (low < 0xdc00U || low > 0xdfffU) {
                                fail("invalid low surrogate");
                            }
                            codepoint = 0x10000U +
                                        ((codepoint - 0xd800U) << 10U) +
                                        (low - 0xdc00U);
                        } else if (codepoint >= 0xdc00U &&
                                   codepoint <= 0xdfffU) {
                            fail("unexpected low surrogate");
                        }
                        appendUtf8(result, codepoint);
                        break;
                    }
                    default:
                        fail("invalid escape sequence");
                }
            } else {
                if (c < 0x20U) {
                    fail("unescaped control character in string");
                }
                result.push_back(static_cast<char>(c));
            }
        }
        fail("unterminated string");
    }

    JsonValue parseArray(int depth) {
        expect('[');
        JsonValue value;
        value.type = JsonValue::Type::Array;
        skipWhitespace();
        if (consume(']')) {
            return value;
        }

        while (true) {
            skipWhitespace();
            value.array.push_back(parseValue(depth));
            skipWhitespace();
            if (consume(']')) {
                return value;
            }
            expect(',');
            skipWhitespace();
        }
    }

    JsonValue parseObject(int depth) {
        expect('{');
        JsonValue value;
        value.type = JsonValue::Type::Object;
        skipWhitespace();
        if (consume('}')) {
            return value;
        }

        while (true) {
            skipWhitespace();
            if (peek() != '"') {
                fail("expected string key");
            }
            std::string key = parseString();
            for (const auto& entry : value.object) {
                if (entry.first == key) {
                    fail("duplicate object key '" + key + "'");
                }
            }
            skipWhitespace();
            expect(':');
            skipWhitespace();
            value.object.emplace_back(std::move(key), parseValue(depth));
            skipWhitespace();
            if (consume('}')) {
                return value;
            }
            expect(',');
            skipWhitespace();
        }
    }
};

}  // namespace

JsonValue parseJson(std::string_view source) {
    Parser parser(source);
    return parser.parse();
}

}  // namespace raster
