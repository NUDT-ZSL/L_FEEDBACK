#include "json.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ct::json {
namespace {

[[noreturn]] void fail(std::string& error, std::size_t offset, const std::string& message) {
    std::ostringstream out;
    out << "JSON parse error at byte " << offset << ": " << message;
    error = out.str();
    throw std::runtime_error(error);
}

}  // namespace

class Parser {
public:
    Parser(std::string_view input, std::string& error) : input_(input), error_(error) {}

    Value parseDocument() {
        skipWhitespace();
        Value result = parseValue();
        skipWhitespace();
        if (pos_ != input_.size()) {
            fail(error_, pos_, "unexpected trailing data");
        }
        return result;
    }

private:
    void skipWhitespace() {
        while (pos_ < input_.size()) {
            const char c = input_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    char peek() const {
        return pos_ < input_.size() ? input_[pos_] : '\0';
    }

    char take() {
        if (pos_ >= input_.size()) {
            fail(error_, pos_, "unexpected end of input");
        }
        return input_[pos_++];
    }

    void expect(char expected) {
        if (pos_ >= input_.size()) {
            fail(error_, pos_, "unexpected end of input");
        }
        if (input_[pos_] != expected) {
            fail(error_, pos_, std::string("expected '") + expected + "'");
        }
        ++pos_;
    }

    bool consumeLiteral(std::string_view literal) {
        if (input_.substr(pos_, literal.size()) != literal) {
            return false;
        }
        pos_ += literal.size();
        return true;
    }

    Value parseValue() {
        skipWhitespace();
        const char c = peek();
        switch (c) {
            case '{':
                return parseObject();
            case '[':
                return parseArray();
            case '"': {
                Value value;
                value.type_ = Value::Type::String;
                value.string_value_ = parseString();
                return value;
            }
            case 't': {
                if (!consumeLiteral("true")) {
                    fail(error_, pos_, "invalid literal");
                }
                Value value;
                value.type_ = Value::Type::Bool;
                value.bool_value_ = true;
                return value;
            }
            case 'f': {
                if (!consumeLiteral("false")) {
                    fail(error_, pos_, "invalid literal");
                }
                Value value;
                value.type_ = Value::Type::Bool;
                value.bool_value_ = false;
                return value;
            }
            case 'n': {
                if (!consumeLiteral("null")) {
                    fail(error_, pos_, "invalid literal");
                }
                return Value{};
            }
            default:
                if (c == '-' || (c >= '0' && c <= '9')) {
                    return parseNumber();
                }
                fail(error_, pos_, "expected a JSON value");
        }
    }

    Value parseObject() {
        Value value;
        value.type_ = Value::Type::Object;
        expect('{');
        skipWhitespace();
        if (peek() == '}') {
            ++pos_;
            return value;
        }
        while (true) {
            skipWhitespace();
            if (peek() != '"') {
                fail(error_, pos_, "object keys must be strings");
            }
            const std::string key = parseString();
            skipWhitespace();
            expect(':');
            Value member = parseValue();
            if (!value.object_value_.emplace(key, std::move(member)).second) {
                fail(error_, pos_, "duplicate object key '" + key + "'");
            }
            skipWhitespace();
            if (peek() == '}') {
                ++pos_;
                break;
            }
            expect(',');
        }
        return value;
    }

    Value parseArray() {
        Value value;
        value.type_ = Value::Type::Array;
        expect('[');
        skipWhitespace();
        if (peek() == ']') {
            ++pos_;
            return value;
        }
        while (true) {
            value.array_value_.push_back(parseValue());
            skipWhitespace();
            if (peek() == ']') {
                ++pos_;
                break;
            }
            expect(',');
        }
        return value;
    }

    std::string parseString() {
        expect('"');
        std::string result;
        while (true) {
            if (pos_ >= input_.size()) {
                fail(error_, pos_, "unterminated string");
            }
            const unsigned char c = static_cast<unsigned char>(input_[pos_++]);
            if (c == '"') {
                break;
            }
            if (c < 0x20) {
                fail(error_, pos_ - 1, "control character in string");
            }
            if (c != '\\') {
                result.push_back(static_cast<char>(c));
                continue;
            }
            if (pos_ >= input_.size()) {
                fail(error_, pos_, "unterminated escape");
            }
            const char esc = input_[pos_++];
            switch (esc) {
                case '"':
                case '\\':
                case '/':
                    result.push_back(esc);
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
                    if (pos_ + 4 > input_.size()) {
                        fail(error_, pos_, "truncated unicode escape");
                    }
                    unsigned codepoint = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char digit = input_[pos_++];
                        codepoint <<= 4;
                        if (digit >= '0' && digit <= '9') {
                            codepoint |= static_cast<unsigned>(digit - '0');
                        } else if (digit >= 'a' && digit <= 'f') {
                            codepoint |= static_cast<unsigned>(digit - 'a' + 10);
                        } else if (digit >= 'A' && digit <= 'F') {
                            codepoint |= static_cast<unsigned>(digit - 'A' + 10);
                        } else {
                            fail(error_, pos_ - 1, "invalid unicode escape");
                        }
                    }
                    if (codepoint <= 0x7f) {
                        result.push_back(static_cast<char>(codepoint));
                    } else if (codepoint <= 0x7ff) {
                        result.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
                        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
                    } else {
                        result.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
                        result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
                        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
                    }
                    break;
                }
                default:
                    fail(error_, pos_ - 1, "invalid escape");
            }
        }
        return result;
    }

    Value parseNumber() {
        const std::size_t start = pos_;
        if (peek() == '-') {
            ++pos_;
        }
        if (peek() == '0') {
            ++pos_;
        } else {
            if (peek() < '1' || peek() > '9') {
                fail(error_, pos_, "invalid number");
            }
            while (peek() >= '0' && peek() <= '9') {
                ++pos_;
            }
        }
        if (peek() == '.') {
            ++pos_;
            if (peek() < '0' || peek() > '9') {
                fail(error_, pos_, "expected digit after decimal point");
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
                fail(error_, pos_, "expected exponent digit");
            }
            while (peek() >= '0' && peek() <= '9') {
                ++pos_;
            }
        }

        const std::string token(input_.substr(start, pos_ - start));
        errno = 0;
        char* end = nullptr;
        const double parsed = std::strtod(token.c_str(), &end);
        if (errno == ERANGE || end == token.c_str() || *end != '\0' ||
            !std::isfinite(parsed)) {
            fail(error_, start, "number is not finite or cannot be represented");
        }
        Value value;
        value.type_ = Value::Type::Number;
        value.number_value_ = parsed;
        return value;
    }

    std::string_view input_;
    std::string& error_;
    std::size_t pos_ = 0;
};

Value Value::parse(std::string_view text, std::string& error) {
    error.clear();
    try {
        Parser parser(text, error);
        return parser.parseDocument();
    } catch (const std::runtime_error&) {
        return Value{};
    }
}

}  // namespace ct::json
