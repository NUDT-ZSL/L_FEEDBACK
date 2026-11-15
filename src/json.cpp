#include "automix/json.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

namespace automix {

JsonValue JsonValue::array(std::vector<JsonValue> values) {
    JsonValue result;
    result.type_ = Type::Array;
    result.array_ = std::move(values);
    return result;
}

JsonValue JsonValue::object(std::map<std::string, JsonValue> values) {
    JsonValue result;
    result.type_ = Type::Object;
    result.object_ = std::move(values);
    return result;
}

const JsonValue* JsonValue::find(const std::string& key) const {
    if (type_ != Type::Object) {
        return nullptr;
    }
    const auto it = object_.find(key);
    return it == object_.end() ? nullptr : &it->second;
}

namespace {

class Parser {
public:
    struct ParseFailure {};

    Parser(const std::string& text, JsonParseError& error)
        : text_(text), error_(error) {}

    JsonValue parse() {
        skipWhitespace();
        JsonValue value = parseValue();
        skipWhitespace();
        if (index_ != text_.size()) {
            fail("unexpected trailing JSON content");
        }
        return value;
    }

private:
    JsonValue parseValue() {
        if (index_ >= text_.size()) {
            fail("unexpected end of JSON");
        }
        const char c = text_[index_];
        switch (c) {
            case 'n': expectLiteral("null"); return JsonValue();
            case 't': expectLiteral("true"); return JsonValue(true);
            case 'f': expectLiteral("false"); return JsonValue(false);
            case '"': return JsonValue(parseString());
            case '[': return parseArray();
            case '{': return parseObject();
            default:
                if (c == '-' || (c >= '0' && c <= '9')) {
                    return JsonValue(parseNumber());
                }
                fail("unexpected character while parsing a JSON value");
        }
        return JsonValue();
    }

    JsonValue parseArray() {
        ++index_;
        std::vector<JsonValue> values;
        skipWhitespace();
        if (consume(']')) {
            return JsonValue::array(std::move(values));
        }
        while (true) {
            skipWhitespace();
            values.push_back(parseValue());
            skipWhitespace();
            if (consume(']')) {
                break;
            }
            if (!consume(',')) {
                fail("expected ',' or ']' in array");
            }
        }
        return JsonValue::array(std::move(values));
    }

    JsonValue parseObject() {
        ++index_;
        std::map<std::string, JsonValue> values;
        skipWhitespace();
        if (consume('}')) {
            return JsonValue::object(std::move(values));
        }
        while (true) {
            skipWhitespace();
            if (index_ >= text_.size() || text_[index_] != '"') {
                fail("expected a string key in object");
            }
            std::string key = parseString();
            skipWhitespace();
            if (!consume(':')) {
                fail("expected ':' after object key");
            }
            skipWhitespace();
            JsonValue value = parseValue();
            if (!values.emplace(std::move(key), std::move(value)).second) {
                fail("duplicate object key");
            }
            skipWhitespace();
            if (consume('}')) {
                break;
            }
            if (!consume(',')) {
                fail("expected ',' or '}' in object");
            }
        }
        return JsonValue::object(std::move(values));
    }

    std::string parseString() {
        ++index_;
        std::string result;
        while (index_ < text_.size()) {
            const unsigned char c = static_cast<unsigned char>(text_[index_++]);
            if (c == '"') {
                return result;
            }
            if (c < 0x20) {
                fail("unescaped control character in string");
            }
            if (c != '\\') {
                result.push_back(static_cast<char>(c));
                continue;
            }
            if (index_ >= text_.size()) {
                fail("unterminated escape sequence");
            }
            const char esc = text_[index_++];
            switch (esc) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': appendUnicodeEscape(result); break;
                default: fail("invalid string escape");
            }
        }
        fail("unterminated string");
        return {};
    }

    void appendUnicodeEscape(std::string& output) {
        const unsigned int first = parseHex4();
        unsigned int codePoint = first;
        if (first >= 0xD800 && first <= 0xDBFF) {
            if (index_ + 2 > text_.size() || text_[index_] != '\\' ||
                text_[index_ + 1] != 'u') {
                fail("high surrogate without a low surrogate");
            }
            index_ += 2;
            const unsigned int second = parseHex4();
            if (second < 0xDC00 || second > 0xDFFF) {
                fail("invalid low surrogate");
            }
            codePoint = 0x10000u + ((first - 0xD800u) << 10u) + (second - 0xDC00u);
        } else if (first >= 0xDC00 && first <= 0xDFFF) {
            fail("unexpected low surrogate");
        }
        appendUtf8(output, codePoint);
    }

    unsigned int parseHex4() {
        if (index_ + 4 > text_.size()) {
            fail("incomplete Unicode escape");
        }
        unsigned int value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[index_++];
            value <<= 4u;
            if (c >= '0' && c <= '9') {
                value |= static_cast<unsigned int>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<unsigned int>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<unsigned int>(c - 'A' + 10);
            } else {
                fail("invalid hexadecimal digit in Unicode escape");
            }
        }
        return value;
    }

    static void appendUtf8(std::string& output, unsigned int codePoint) {
        if (codePoint <= 0x7Fu) {
            output.push_back(static_cast<char>(codePoint));
        } else if (codePoint <= 0x7FFu) {
            output.push_back(static_cast<char>(0xC0u | (codePoint >> 6u)));
            output.push_back(static_cast<char>(0x80u | (codePoint & 0x3Fu)));
        } else if (codePoint <= 0xFFFFu) {
            output.push_back(static_cast<char>(0xE0u | (codePoint >> 12u)));
            output.push_back(static_cast<char>(0x80u | ((codePoint >> 6u) & 0x3Fu)));
            output.push_back(static_cast<char>(0x80u | (codePoint & 0x3Fu)));
        } else {
            output.push_back(static_cast<char>(0xF0u | (codePoint >> 18u)));
            output.push_back(static_cast<char>(0x80u | ((codePoint >> 12u) & 0x3Fu)));
            output.push_back(static_cast<char>(0x80u | ((codePoint >> 6u) & 0x3Fu)));
            output.push_back(static_cast<char>(0x80u | (codePoint & 0x3Fu)));
        }
    }

    double parseNumber() {
        const std::size_t start = index_;
        if (consume('-')) {
            if (index_ >= text_.size()) {
                fail("incomplete number");
            }
        }
        if (consume('0')) {
            if (index_ < text_.size() && text_[index_] >= '0' && text_[index_] <= '9') {
                fail("leading zero in number");
            }
        } else {
            if (index_ >= text_.size() || text_[index_] < '1' || text_[index_] > '9') {
                fail("invalid number");
            }
            while (index_ < text_.size() && text_[index_] >= '0' && text_[index_] <= '9') {
                ++index_;
            }
        }
        if (consume('.')) {
            const std::size_t fractionStart = index_;
            while (index_ < text_.size() && text_[index_] >= '0' && text_[index_] <= '9') {
                ++index_;
            }
            if (fractionStart == index_) {
                fail("fraction has no digits");
            }
        }
        if (index_ < text_.size() && (text_[index_] == 'e' || text_[index_] == 'E')) {
            ++index_;
            if (index_ < text_.size() && (text_[index_] == '+' || text_[index_] == '-')) {
                ++index_;
            }
            const std::size_t exponentStart = index_;
            while (index_ < text_.size() && text_[index_] >= '0' && text_[index_] <= '9') {
                ++index_;
            }
            if (exponentStart == index_) {
                fail("exponent has no digits");
            }
        }
        const std::string token = text_.substr(start, index_ - start);
        char* end = nullptr;
        errno = 0;
        const double value = std::strtod(token.c_str(), &end);
        if (end != token.c_str() + token.size() || errno == ERANGE || !std::isfinite(value)) {
            fail("number is outside the supported finite range");
        }
        return value;
    }

    void expectLiteral(std::string_view literal) {
        if (text_.compare(index_, literal.size(), literal) != 0) {
            fail("invalid literal");
        }
        index_ += literal.size();
    }

    bool consume(char expected) {
        if (index_ < text_.size() && text_[index_] == expected) {
            ++index_;
            return true;
        }
        return false;
    }

    void skipWhitespace() {
        while (index_ < text_.size()) {
            const char c = text_[index_];
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
                break;
            }
            ++index_;
        }
    }

    [[noreturn]] void fail(const std::string& message) {
        error_.message = message;
        error_.line = 1;
        error_.column = 1;
        for (std::size_t i = 0; i < index_ && i < text_.size(); ++i) {
            if (text_[i] == '\n') {
                ++error_.line;
                error_.column = 1;
            } else {
                ++error_.column;
            }
        }
        throw ParseFailure{};
    }

    const std::string& text_;
    JsonParseError& error_;
    std::size_t index_ = 0;
};

}  // namespace

JsonValue parseJson(const std::string& text, JsonParseError& error) {
    error = {};
    try {
        Parser parser(text, error);
        return parser.parse();
    } catch (const Parser::ParseFailure&) {
        return JsonValue();
    }
}

std::string jsonQuote(const std::string& text) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : text) {
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
                    static const char hex[] = "0123456789abcdef";
                    out << "\\u00" << hex[(c >> 4u) & 0x0Fu] << hex[c & 0x0Fu];
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    out << '"';
    return out.str();
}

}  // namespace automix
