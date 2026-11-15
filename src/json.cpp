#include "json.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace vlr::json {

const Value* Value::find(std::string_view key) const {
    if (!isObject()) {
        return nullptr;
    }
    const auto& object = asObject();
    const auto it = object.find(std::string(key));
    return it == object.end() ? nullptr : &it->second;
}

namespace {

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Value parseDocument() {
        skipWhitespace();
        Value result = parseValue();
        skipWhitespace();
        if (position_ != text_.size()) {
            error("trailing characters after JSON value");
        }
        return result;
    }

private:
    [[noreturn]] void error(const std::string& message) const {
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
        throw std::runtime_error(
            "JSON parse error at line " + std::to_string(line) +
            ", column " + std::to_string(column) + ": " + message);
    }

    void skipWhitespace() {
        while (position_ < text_.size()) {
            const char c = text_[position_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++position_;
            } else {
                break;
            }
        }
    }

    bool consume(char expected) {
        if (position_ < text_.size() && text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    Value parseValue() {
        if (position_ >= text_.size()) {
            error("unexpected end of input");
        }
        switch (text_[position_]) {
        case 'n': parseLiteral("null"); return Value();
        case 't': parseLiteral("true"); return Value(true);
        case 'f': parseLiteral("false"); return Value(false);
        case '"': return Value(parseString());
        case '[': return Value(parseArray());
        case '{': return Value(parseObject());
        default:
            if (text_[position_] == '-' ||
                (text_[position_] >= '0' && text_[position_] <= '9')) {
                return Value(parseNumber());
            }
            error("unexpected token");
        }
    }

    void parseLiteral(std::string_view literal) {
        if (text_.substr(position_, literal.size()) != literal) {
            error("invalid literal");
        }
        position_ += literal.size();
    }

    double parseNumber() {
        const std::size_t start = position_;
        if (text_[position_] == '-') {
            ++position_;
        }
        if (position_ >= text_.size()) {
            error("incomplete number");
        }
        if (text_[position_] == '0') {
            ++position_;
        } else if (text_[position_] >= '1' && text_[position_] <= '9') {
            while (position_ < text_.size() &&
                   text_[position_] >= '0' && text_[position_] <= '9') {
                ++position_;
            }
        } else {
            error("invalid number");
        }
        if (position_ < text_.size() && text_[position_] == '.') {
            ++position_;
            const std::size_t fractionStart = position_;
            while (position_ < text_.size() &&
                   text_[position_] >= '0' && text_[position_] <= '9') {
                ++position_;
            }
            if (position_ == fractionStart) {
                error("missing fractional digits");
            }
        }
        if (position_ < text_.size() &&
            (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() &&
                (text_[position_] == '+' || text_[position_] == '-')) {
                ++position_;
            }
            const std::size_t exponentStart = position_;
            while (position_ < text_.size() &&
                   text_[position_] >= '0' && text_[position_] <= '9') {
                ++position_;
            }
            if (position_ == exponentStart) {
                error("missing exponent digits");
            }
        }
        const std::string number(text_.substr(start, position_ - start));
        char* end = nullptr;
        const double value = std::strtod(number.c_str(), &end);
        if (end != number.c_str() + number.size() || !std::isfinite(value)) {
            error("number is not finite");
        }
        return value;
    }

    static void appendUtf8(std::string& output, std::uint32_t cp) {
        if (cp <= 0x7f) {
            output.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7ff) {
            output.push_back(static_cast<char>(0xc0 | (cp >> 6)));
            output.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else if (cp <= 0xffff) {
            output.push_back(static_cast<char>(0xe0 | (cp >> 12)));
            output.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else {
            output.push_back(static_cast<char>(0xf0 | (cp >> 18)));
            output.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        }
    }

    std::uint32_t parseHex4() {
        if (position_ + 4 > text_.size()) {
            error("incomplete unicode escape");
        }
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[position_++];
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<std::uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<std::uint32_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<std::uint32_t>(c - 'A' + 10);
            } else {
                error("invalid unicode escape");
            }
        }
        return value;
    }

    std::string parseString() {
        if (!consume('"')) {
            error("expected string");
        }
        std::string result;
        while (position_ < text_.size()) {
            const unsigned char c = static_cast<unsigned char>(text_[position_++]);
            if (c == '"') {
                return result;
            }
            if (c < 0x20) {
                error("control character in string");
            }
            if (c != '\\') {
                result.push_back(static_cast<char>(c));
                continue;
            }
            if (position_ >= text_.size()) {
                error("incomplete escape");
            }
            const char escaped = text_[position_++];
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
                std::uint32_t cp = parseHex4();
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    if (position_ + 2 > text_.size() ||
                        text_[position_] != '\\' || text_[position_ + 1] != 'u') {
                        error("missing low surrogate");
                    }
                    position_ += 2;
                    const std::uint32_t low = parseHex4();
                    if (low < 0xdc00 || low > 0xdfff) {
                        error("invalid low surrogate");
                    }
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                } else if (cp >= 0xdc00 && cp <= 0xdfff) {
                    error("unexpected low surrogate");
                }
                appendUtf8(result, cp);
                break;
            }
            default:
                error("invalid escape");
            }
        }
        error("unterminated string");
    }

    Value::Array parseArray() {
        consume('[');
        Value::Array result;
        skipWhitespace();
        if (consume(']')) {
            return result;
        }
        while (true) {
            skipWhitespace();
            result.push_back(parseValue());
            skipWhitespace();
            if (consume(']')) {
                return result;
            }
            if (!consume(',')) {
                error("expected ',' or ']'");
            }
        }
    }

    Value::Object parseObject() {
        consume('{');
        Value::Object result;
        skipWhitespace();
        if (consume('}')) {
            return result;
        }
        while (true) {
            skipWhitespace();
            if (position_ >= text_.size() || text_[position_] != '"') {
                error("expected object key");
            }
            std::string key = parseString();
            skipWhitespace();
            if (!consume(':')) {
                error("expected ':' after object key");
            }
            skipWhitespace();
            auto [it, inserted] = result.emplace(std::move(key), parseValue());
            if (!inserted) {
                error("duplicate object key");
            }
            skipWhitespace();
            if (consume('}')) {
                return result;
            }
            if (!consume(',')) {
                error("expected ',' or '}'");
            }
        }
    }

    std::string_view text_;
    std::size_t position_ = 0;
};

void indentLine(std::ostringstream& out, int indent) {
    for (int i = 0; i < indent; ++i) {
        out.put(' ');
    }
}

void stringifyImpl(std::ostringstream& out, const Value& value, int indent, int depth) {
    switch (value.type()) {
    case Value::Type::Null:
        out << "null";
        break;
    case Value::Type::Bool:
        out << (value.asBool() ? "true" : "false");
        break;
    case Value::Type::Number: {
        const double number = value.asNumber();
        std::ostringstream numberText;
        numberText << std::setprecision(17) << number;
        out << numberText.str();
        break;
    }
    case Value::Type::String:
        out << '"' << escape(value.asString()) << '"';
        break;
    case Value::Type::Array: {
        const auto& array = value.asArray();
        out << '[';
        if (!array.empty()) {
            for (std::size_t i = 0; i < array.size(); ++i) {
                if (i != 0) {
                    out << ',';
                }
                if (indent >= 0) {
                    out << '\n';
                    indentLine(out, (depth + 1) * indent);
                }
                stringifyImpl(out, array[i], indent, depth + 1);
            }
            if (indent >= 0) {
                out << '\n';
                indentLine(out, depth * indent);
            }
        }
        out << ']';
        break;
    }
    case Value::Type::Object: {
        const auto& object = value.asObject();
        out << '{';
        if (!object.empty()) {
            std::size_t index = 0;
            for (const auto& [key, item] : object) {
                if (index++ != 0) {
                    out << ',';
                }
                if (indent >= 0) {
                    out << '\n';
                    indentLine(out, (depth + 1) * indent);
                }
                out << '"' << escape(key) << "\":";
                if (indent >= 0) {
                    out << ' ';
                }
                stringifyImpl(out, item, indent, depth + 1);
            }
            if (indent >= 0) {
                out << '\n';
                indentLine(out, depth * indent);
            }
        }
        out << '}';
        break;
    }
    }
}

} // namespace

Value parse(std::string_view text) {
    return Parser(text).parseDocument();
}

std::string stringify(const Value& value, int indent) {
    std::ostringstream out;
    stringifyImpl(out, value, indent, 0);
    return out.str();
}

std::string escape(std::string_view text) {
    std::ostringstream out;
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
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(c) << std::dec;
            } else {
                out << static_cast<char>(c);
            }
        }
    }
    return out.str();
}

} // namespace vlr::json
