#include "json.hpp"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace lfb::json {

namespace {

constexpr int kMaxDepth = 128;

class Parser {
public:
    explicit Parser(std::string_view input) : input_(input) {}

    Value parseDocument() {
        skipWhitespace();
        Value value = parseValue(0);
        skipWhitespace();
        if (pos_ != input_.size()) {
            fail("unexpected trailing data");
        }
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& message) const {
        std::ostringstream out;
        out << "JSON parse error at byte " << pos_ << ": " << message;
        throw ParseError(out.str());
    }

    void skipWhitespace() {
        while (pos_ < input_.size()) {
            const char c = input_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    bool consume(char c) {
        if (pos_ < input_.size() && input_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    void expect(char c) {
        if (!consume(c)) {
            fail(std::string("expected '") + c + "'");
        }
    }

    Value parseValue(int depth) {
        if (depth > kMaxDepth) {
            fail("nesting depth exceeds 128");
        }
        if (pos_ >= input_.size()) {
            fail("unexpected end of input");
        }
        const char c = input_[pos_];
        switch (c) {
            case '{': return parseObject(depth + 1);
            case '[': return parseArray(depth + 1);
            case '"': return Value(parseString());
            case 't': parseLiteral("true"); return Value(true);
            case 'f': parseLiteral("false"); return Value(false);
            case 'n': parseLiteral("null"); return Value();
            default:
                if (c == '-' || (c >= '0' && c <= '9')) {
                    return Value(parseNumber());
                }
                fail("unexpected token");
        }
    }

    void parseLiteral(std::string_view literal) {
        if (input_.substr(pos_, literal.size()) != literal) {
            fail("invalid literal");
        }
        pos_ += literal.size();
    }

    Value parseObject(int depth) {
        expect('{');
        skipWhitespace();
        std::vector<std::pair<std::string, Value>> members;
        if (consume('}')) {
            return Value(std::move(members));
        }
        while (true) {
            skipWhitespace();
            if (pos_ >= input_.size() || input_[pos_] != '"') {
                fail("object key must be a string");
            }
            std::string key = parseString();
            for (const auto& member : members) {
                if (member.first == key) {
                    fail("duplicate object key '" + key + "'");
                }
            }
            skipWhitespace();
            expect(':');
            skipWhitespace();
            Value value = parseValue(depth);
            members.emplace_back(std::move(key), std::move(value));
            skipWhitespace();
            if (consume('}')) {
                break;
            }
            expect(',');
        }
        return Value(std::move(members));
    }

    Value parseArray(int depth) {
        expect('[');
        skipWhitespace();
        std::vector<Value> elements;
        if (consume(']')) {
            return Value(std::move(elements));
        }
        while (true) {
            skipWhitespace();
            elements.push_back(parseValue(depth));
            skipWhitespace();
            if (consume(']')) {
                break;
            }
            expect(',');
        }
        return Value(std::move(elements));
    }

    double parseNumber() {
        const std::size_t start = pos_;
        consume('-');
        if (pos_ >= input_.size()) {
            fail("incomplete number");
        }
        if (input_[pos_] == '0') {
            ++pos_;
            if (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') {
                fail("leading zero in number");
            }
        } else if (input_[pos_] >= '1' && input_[pos_] <= '9') {
            ++pos_;
            while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') {
                ++pos_;
            }
        } else {
            fail("invalid number");
        }

        if (consume('.')) {
            if (pos_ >= input_.size() || input_[pos_] < '0' || input_[pos_] > '9') {
                fail("fraction requires at least one digit");
            }
            while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') {
                ++pos_;
            }
        }

        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= input_.size() || input_[pos_] < '0' || input_[pos_] > '9') {
                fail("exponent requires at least one digit");
            }
            while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') {
                ++pos_;
            }
        }

        const std::string token(input_.substr(start, pos_ - start));
        char* end = nullptr;
        errno = 0;
        const double value = std::strtod(token.c_str(), &end);
        if (end != token.c_str() + token.size() || errno == ERANGE || !std::isfinite(value)) {
            fail("number is out of range");
        }
        return value;
    }

    static void appendUtf8(std::string& out, std::uint32_t cp) {
        if (cp <= 0x7F) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    std::uint32_t parseHex4() {
        if (pos_ + 4 > input_.size()) {
            fail("incomplete \\u escape");
        }
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = input_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<std::uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<std::uint32_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<std::uint32_t>(c - 'A' + 10);
            } else {
                fail("invalid \\u escape");
            }
        }
        return value;
    }

    std::string parseString() {
        expect('"');
        std::string out;
        while (pos_ < input_.size()) {
            const unsigned char c = static_cast<unsigned char>(input_[pos_++]);
            if (c == '"') {
                return out;
            }
            if (c < 0x20) {
                fail("unescaped control character in string");
            }
            if (c != '\\') {
                out.push_back(static_cast<char>(c));
                continue;
            }
            if (pos_ >= input_.size()) {
                fail("incomplete escape sequence");
            }
            const char esc = input_[pos_++];
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
                    std::uint32_t cp = parseHex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (pos_ + 2 > input_.size() || input_[pos_] != '\\' || input_[pos_ + 1] != 'u') {
                            fail("high surrogate without low surrogate");
                        }
                        pos_ += 2;
                        const std::uint32_t low = parseHex4();
                        if (low < 0xDC00 || low > 0xDFFF) {
                            fail("invalid low surrogate");
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail("unexpected low surrogate");
                    }
                    appendUtf8(out, cp);
                    break;
                }
                default: fail("invalid escape sequence");
            }
        }
        fail("unterminated string");
    }

    std::string_view input_;
    std::size_t pos_ = 0;
};

}  // namespace

const Value* Value::find(std::string_view key) const {
    if (type_ != Type::Object) {
        return nullptr;
    }
    for (const auto& member : object_) {
        if (member.first == key) {
            return &member.second;
        }
    }
    return nullptr;
}

void Value::set(std::string key, Value value) {
    if (type_ != Type::Object) {
        type_ = Type::Object;
    }
    for (auto& member : object_) {
        if (member.first == key) {
            member.second = std::move(value);
            return;
        }
    }
    object_.emplace_back(std::move(key), std::move(value));
}

Value parse(std::string_view input) {
    return Parser(input).parseDocument();
}

std::string typeName(Value::Type type) {
    switch (type) {
        case Value::Type::Null: return "null";
        case Value::Type::Bool: return "boolean";
        case Value::Type::Number: return "number";
        case Value::Type::String: return "string";
        case Value::Type::Array: return "array";
        case Value::Type::Object: return "object";
    }
    return "unknown";
}

}  // namespace lfb::json
