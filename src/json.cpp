#include "json.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <sstream>

namespace subdiv {
namespace {

class Parser {
public:
    explicit Parser(const std::string& input) : input_(input) {}

    bool parse(JsonValue& value, std::string& error) {
        skipWhitespace();
        if (!parseValue(value, 0)) {
            error = error_;
            return false;
        }
        skipWhitespace();
        if (position_ != input_.size()) {
            fail("unexpected trailing content");
            error = error_;
            return false;
        }
        return true;
    }

private:
    bool parseValue(JsonValue& value, int depth) {
        if (depth > 128) {
            return fail("JSON nesting is too deep");
        }
        skipWhitespace();
        if (position_ >= input_.size()) {
            return fail("expected a value");
        }

        const char token = input_[position_];
        if (token == '{') {
            return parseObject(value, depth + 1);
        }
        if (token == '[') {
            return parseArray(value, depth + 1);
        }
        if (token == '"') {
            value.type = JsonValue::Type::String;
            return parseString(value.string);
        }
        if (token == 't') {
            return parseLiteral("true", value, JsonValue::Type::Boolean, true);
        }
        if (token == 'f') {
            return parseLiteral("false", value, JsonValue::Type::Boolean, false);
        }
        if (token == 'n') {
            return parseNull(value);
        }
        if (token == '-' || std::isdigit(static_cast<unsigned char>(token)) != 0) {
            return parseNumber(value);
        }
        return fail("expected a JSON value");
    }

    bool parseObject(JsonValue& value, int depth) {
        value.type = JsonValue::Type::Object;
        ++position_;
        skipWhitespace();
        if (consume('}')) {
            return true;
        }

        while (true) {
            skipWhitespace();
            if (position_ >= input_.size() || input_[position_] != '"') {
                return fail("expected an object key string");
            }
            std::string key;
            if (!parseString(key)) {
                return false;
            }
            skipWhitespace();
            if (!consume(':')) {
                return fail("expected ':' after object key");
            }
            JsonValue child;
            if (!parseValue(child, depth)) {
                return false;
            }
            value.object.emplace_back(std::move(key), std::move(child));
            skipWhitespace();
            if (consume('}')) {
                return true;
            }
            if (!consume(',')) {
                return fail("expected ',' or '}' in object");
            }
        }
    }

    bool parseArray(JsonValue& value, int depth) {
        value.type = JsonValue::Type::Array;
        ++position_;
        skipWhitespace();
        if (consume(']')) {
            return true;
        }
        while (true) {
            JsonValue child;
            if (!parseValue(child, depth)) {
                return false;
            }
            value.array.push_back(std::move(child));
            skipWhitespace();
            if (consume(']')) {
                return true;
            }
            if (!consume(',')) {
                return fail("expected ',' or ']' in array");
            }
        }
    }

    bool parseString(std::string& output) {
        if (!consume('"')) {
            return fail("expected a string");
        }
        output.clear();
        while (position_ < input_.size()) {
            const char c = input_[position_++];
            if (c == '"') {
                return true;
            }
            if (c == '\\') {
                if (position_ >= input_.size()) {
                    return fail("unterminated string escape");
                }
                const char escaped = input_[position_++];
                switch (escaped) {
                    case '"': output.push_back('"'); break;
                    case '\\': output.push_back('\\'); break;
                    case '/': output.push_back('/'); break;
                    case 'b': output.push_back('\b'); break;
                    case 'f': output.push_back('\f'); break;
                    case 'n': output.push_back('\n'); break;
                    case 'r': output.push_back('\r'); break;
                    case 't': output.push_back('\t'); break;
                    case 'u': {
                        std::uint32_t code = 0;
                        if (!parseUnicodeEscape(code)) {
                            return false;
                        }
                        if (code <= 0x7fU) {
                            output.push_back(static_cast<char>(code));
                        } else if (code <= 0x7ffU) {
                            output.push_back(static_cast<char>(0xc0U | (code >> 6U)));
                            output.push_back(static_cast<char>(0x80U | (code & 0x3fU)));
                        } else {
                            output.push_back(static_cast<char>(0xe0U | (code >> 12U)));
                            output.push_back(static_cast<char>(0x80U | ((code >> 6U) & 0x3fU)));
                            output.push_back(static_cast<char>(0x80U | (code & 0x3fU)));
                        }
                        break;
                    }
                    default:
                        return fail("invalid string escape");
                }
            } else {
                if (static_cast<unsigned char>(c) < 0x20U) {
                    return fail("unescaped control character in string");
                }
                output.push_back(c);
            }
        }
        return fail("unterminated string");
    }

    bool parseUnicodeEscape(std::uint32_t& code) {
        if (position_ + 4 > input_.size()) {
            return fail("truncated unicode escape");
        }
        code = 0;
        for (int i = 0; i < 4; ++i) {
            const char hex = input_[position_++];
            code <<= 4U;
            if (hex >= '0' && hex <= '9') {
                code |= static_cast<std::uint32_t>(hex - '0');
            } else if (hex >= 'a' && hex <= 'f') {
                code |= static_cast<std::uint32_t>(hex - 'a' + 10);
            } else if (hex >= 'A' && hex <= 'F') {
                code |= static_cast<std::uint32_t>(hex - 'A' + 10);
            } else {
                return fail("invalid unicode escape digit");
            }
        }
        return true;
    }

    bool parseNumber(JsonValue& value) {
        const std::size_t start = position_;
        if (input_[position_] == '-') {
            ++position_;
        }
        if (position_ >= input_.size()) {
            return fail("truncated number");
        }
        if (input_[position_] == '0') {
            ++position_;
        } else if (std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
                ++position_;
            }
        } else {
            return fail("invalid number");
        }
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            if (position_ >= input_.size() ||
                std::isdigit(static_cast<unsigned char>(input_[position_])) == 0) {
                return fail("expected digits after decimal point");
            }
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
                ++position_;
            }
        }
        if (position_ < input_.size() &&
            (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() &&
                (input_[position_] == '+' || input_[position_] == '-')) {
                ++position_;
            }
            if (position_ >= input_.size() ||
                std::isdigit(static_cast<unsigned char>(input_[position_])) == 0) {
                return fail("expected exponent digits");
            }
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
                ++position_;
            }
        }

        const std::string token = input_.substr(start, position_ - start);
        char* end = nullptr;
        const double parsed = std::strtod(token.c_str(), &end);
        if (end == nullptr || *end != '\0') {
            return fail("invalid number");
        }
        value.type = JsonValue::Type::Number;
        value.number = parsed;
        return true;
    }

    bool parseLiteral(const char* literal, JsonValue& value,
                      JsonValue::Type type, bool booleanValue) {
        const std::size_t length = std::char_traits<char>::length(literal);
        if (input_.compare(position_, length, literal) != 0) {
            return fail("invalid literal");
        }
        position_ += length;
        value.type = type;
        value.boolean = booleanValue;
        return true;
    }

    bool parseNull(JsonValue& value) {
        if (input_.compare(position_, 4, "null") != 0) {
            return fail("invalid literal");
        }
        position_ += 4;
        value.type = JsonValue::Type::Null;
        return true;
    }

    bool consume(char expected) {
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void skipWhitespace() {
        while (position_ < input_.size()) {
            const char c = input_[position_];
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
                break;
            }
            ++position_;
        }
    }

    bool fail(const std::string& message) {
        if (error_.empty()) {
            std::size_t line = 1;
            std::size_t column = 1;
            for (std::size_t i = 0; i < position_ && i < input_.size(); ++i) {
                if (input_[i] == '\n') {
                    ++line;
                    column = 1;
                } else {
                    ++column;
                }
            }
            std::ostringstream stream;
            stream << "JSON line " << line << ", column " << column << ": " << message;
            error_ = stream.str();
        }
        return false;
    }

    const std::string& input_;
    std::size_t position_ = 0;
    std::string error_;
};

}  // namespace

const JsonValue* JsonValue::find(const std::string& key) const {
    if (type != Type::Object) {
        return nullptr;
    }
    for (const auto& member : object) {
        if (member.first == key) {
            return &member.second;
        }
    }
    return nullptr;
}

bool parseJson(const std::string& text, JsonValue& value, std::string& error) {
    Parser parser(text);
    return parser.parse(value, error);
}

}  // namespace subdiv

