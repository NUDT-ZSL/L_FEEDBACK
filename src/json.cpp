#include "json.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>

namespace tilegen {

namespace {

std::string location(std::size_t line, std::size_t column) {
    std::ostringstream out;
    out << "JSON line " << line << ", column " << column;
    return out.str();
}

}  // namespace

JsonError::JsonError(const std::string& message, std::size_t line, std::size_t column)
    : std::runtime_error(location(line, column) + ": " + message),
      line_(line),
      column_(column) {}

JsonValue::JsonValue(bool value) : type_(Type::Boolean), bool_(value) {}
JsonValue::JsonValue(double value) : type_(Type::Number), number_(value) {}
JsonValue::JsonValue(std::string value) : type_(Type::String), string_(std::move(value)) {}
JsonValue::JsonValue(std::vector<JsonValue> value)
    : type_(Type::Array), array_(std::move(value)) {}
JsonValue::JsonValue(std::map<std::string, JsonValue> value)
    : type_(Type::Object), object_(std::move(value)) {}

bool JsonValue::asBool() const {
    if (!isBool()) {
        throw std::logic_error("JSON value is not a boolean");
    }
    return bool_;
}

double JsonValue::asNumber() const {
    if (!isNumber()) {
        throw std::logic_error("JSON value is not a number");
    }
    return number_;
}

const std::string& JsonValue::asString() const {
    if (!isString()) {
        throw std::logic_error("JSON value is not a string");
    }
    return string_;
}

const std::vector<JsonValue>& JsonValue::asArray() const {
    if (!isArray()) {
        throw std::logic_error("JSON value is not an array");
    }
    return array_;
}

const std::map<std::string, JsonValue>& JsonValue::asObject() const {
    if (!isObject()) {
        throw std::logic_error("JSON value is not an object");
    }
    return object_;
}

const JsonValue* JsonValue::find(const std::string& key) const {
    if (!isObject()) {
        return nullptr;
    }
    auto it = object_.find(key);
    return it == object_.end() ? nullptr : &it->second;
}

bool JsonValue::contains(const std::string& key) const {
    return find(key) != nullptr;
}

JsonValue JsonParser::parse(const std::string& text) {
    text_ = text;
    position_ = 0;
    if (text_.size() >= 3 &&
        static_cast<unsigned char>(text_[0]) == 0xEF &&
        static_cast<unsigned char>(text_[1]) == 0xBB &&
        static_cast<unsigned char>(text_[2]) == 0xBF) {
        position_ = 3;
    }
    skipWhitespace();
    JsonValue value = parseValue(0);
    skipWhitespace();
    if (position_ != text_.size()) {
        error("unexpected trailing content");
    }
    return value;
}

JsonValue JsonParser::parseFile(const std::string& path) {
    return parse(readTextFile(path));
}

JsonValue JsonParser::parseValue(std::size_t depth) {
    if (depth > 128) {
        error("maximum JSON nesting depth exceeded");
    }
    skipWhitespace();
    if (position_ >= text_.size()) {
        error("unexpected end of input");
    }
    switch (peek()) {
        case '{':
            return parseObject(depth + 1);
        case '[':
            return parseArray(depth + 1);
        case '"':
            return parseString();
        case 't':
            parseLiteral("true", JsonValue(true));
            return JsonValue(true);
        case 'f':
            parseLiteral("false", JsonValue(false));
            return JsonValue(false);
        case 'n':
            parseLiteral("null", JsonValue());
            return JsonValue();
        default:
            if (peek() == '-' || (peek() >= '0' && peek() <= '9')) {
                return parseNumber();
            }
            error("expected a JSON value");
    }
}

JsonValue JsonParser::parseObject(std::size_t depth) {
    expect('{', "expected '{'");
    std::map<std::string, JsonValue> object;
    skipWhitespace();
    if (consume('}')) {
        return JsonValue(std::move(object));
    }
    while (true) {
        skipWhitespace();
        if (peek() != '"') {
            error("expected an object key");
        }
        JsonValue key = parseString();
        skipWhitespace();
        expect(':', "expected ':' after object key");
        if (object.find(key.asString()) != object.end()) {
            error("duplicate object key '" + key.asString() + "'");
        }
        object.emplace(key.asString(), parseValue(depth));
        skipWhitespace();
        if (consume('}')) {
            break;
        }
        expect(',', "expected ',' or '}' in object");
    }
    return JsonValue(std::move(object));
}

JsonValue JsonParser::parseArray(std::size_t depth) {
    expect('[', "expected '['");
    std::vector<JsonValue> array;
    skipWhitespace();
    if (consume(']')) {
        return JsonValue(std::move(array));
    }
    while (true) {
        array.push_back(parseValue(depth));
        skipWhitespace();
        if (consume(']')) {
            break;
        }
        expect(',', "expected ',' or ']' in array");
    }
    return JsonValue(std::move(array));
}

JsonValue JsonParser::parseString() {
    expect('"', "expected '\"'");
    std::string out;
    while (position_ < text_.size()) {
        unsigned char ch = static_cast<unsigned char>(take());
        if (ch == '"') {
            return JsonValue(std::move(out));
        }
        if (ch < 0x20) {
            error("unescaped control character in string");
        }
        if (ch != '\\') {
            out.push_back(static_cast<char>(ch));
            continue;
        }
        if (position_ >= text_.size()) {
            error("unterminated escape sequence");
        }
        char escape = take();
        switch (escape) {
            case '"':
                out.push_back('"');
                break;
            case '\\':
                out.push_back('\\');
                break;
            case '/':
                out.push_back('/');
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u': {
                auto readHex = [this]() -> unsigned int {
                    if (position_ + 4 > text_.size()) {
                        error("truncated \\u escape");
                    }
                    unsigned int value = 0;
                    for (int i = 0; i < 4; ++i) {
                        char digit = take();
                        value <<= 4;
                        if (digit >= '0' && digit <= '9') {
                            value += static_cast<unsigned int>(digit - '0');
                        } else if (digit >= 'a' && digit <= 'f') {
                            value += static_cast<unsigned int>(digit - 'a' + 10);
                        } else if (digit >= 'A' && digit <= 'F') {
                            value += static_cast<unsigned int>(digit - 'A' + 10);
                        } else {
                            error("invalid hex digit in \\u escape");
                        }
                    }
                    return value;
                };
                unsigned int codepoint = readHex();
                if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                    if (position_ + 2 > text_.size() || peek() != '\\') {
                        error("high surrogate without low surrogate");
                    }
                    take();
                    if (take() != 'u') {
                        error("high surrogate without low surrogate");
                    }
                    unsigned int low = readHex();
                    if (low < 0xDC00 || low > 0xDFFF) {
                        error("invalid low surrogate");
                    }
                    codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
                } else if (codepoint >= 0xDC00 && codepoint <= 0xDFFF) {
                    error("unexpected low surrogate");
                }
                appendUtf8(out, codepoint);
                break;
            }
            default:
                error("invalid escape sequence");
        }
    }
    error("unterminated string");
}

JsonValue JsonParser::parseNumber() {
    const std::size_t start = position_;
    if (peek() == '-') {
        ++position_;
    }
    if (position_ >= text_.size()) {
        error("invalid number");
    }
    if (peek() == '0') {
        ++position_;
        if (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
            error("leading zero in number");
        }
    } else if (peek() >= '1' && peek() <= '9') {
        while (position_ < text_.size() && text_[position_] >= '0' &&
               text_[position_] <= '9') {
            ++position_;
        }
    } else {
        error("invalid number");
    }
    if (position_ < text_.size() && text_[position_] == '.') {
        ++position_;
        if (position_ >= text_.size() || text_[position_] < '0' || text_[position_] > '9') {
            error("invalid fractional part");
        }
        while (position_ < text_.size() && text_[position_] >= '0' &&
               text_[position_] <= '9') {
            ++position_;
        }
    }
    if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
        ++position_;
        if (position_ < text_.size() &&
            (text_[position_] == '+' || text_[position_] == '-')) {
            ++position_;
        }
        if (position_ >= text_.size() || text_[position_] < '0' || text_[position_] > '9') {
            error("invalid exponent");
        }
        while (position_ < text_.size() && text_[position_] >= '0' &&
               text_[position_] <= '9') {
            ++position_;
        }
    }

    double value = 0.0;
    const char* begin = text_.data() + start;
    const char* end = text_.data() + position_;
    auto result = std::from_chars(begin, end, value, std::chars_format::general);
    if (result.ec == std::errc::result_out_of_range) {
        // Preserve overflow as a non-finite value so schema validation can
        // report the exact JSON field path instead of only a source position.
        value = text_[start] == '-' ? -std::numeric_limits<double>::infinity()
                                    : std::numeric_limits<double>::infinity();
    } else if (result.ec != std::errc() || result.ptr != end) {
        error("invalid number");
    }
    return JsonValue(value);
}

void JsonParser::parseLiteral(const char* literal, const JsonValue&) {
    const std::size_t start = position_;
    for (const char* p = literal; *p != '\0'; ++p) {
        if (position_ >= text_.size() || text_[position_] != *p) {
            position_ = start;
            error("invalid literal");
        }
        ++position_;
    }
}

void JsonParser::skipWhitespace() {
    while (position_ < text_.size()) {
        char ch = text_[position_];
        if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
            break;
        }
        ++position_;
    }
}

char JsonParser::peek() const {
    if (position_ >= text_.size()) {
        return '\0';
    }
    return text_[position_];
}

char JsonParser::take() {
    if (position_ >= text_.size()) {
        error("unexpected end of input");
    }
    return text_[position_++];
}

bool JsonParser::consume(char expected) {
    if (position_ < text_.size() && text_[position_] == expected) {
        ++position_;
        return true;
    }
    return false;
}

void JsonParser::expect(char expected, const std::string& message) {
    if (!consume(expected)) {
        error(message);
    }
}

[[noreturn]] void JsonParser::error(const std::string& message) const {
    throw JsonError(message, lineAt(position_), columnAt(position_));
}

void JsonParser::appendUtf8(std::string& out, unsigned int codepoint) {
    if (codepoint <= 0x7F) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

std::size_t JsonParser::lineAt(std::size_t position) const {
    std::size_t line = 1;
    for (std::size_t i = 0; i < position && i < text_.size(); ++i) {
        if (text_[i] == '\n') {
            ++line;
        }
    }
    return line;
}

std::size_t JsonParser::columnAt(std::size_t position) const {
    std::size_t lineStart = 0;
    for (std::size_t i = 0; i < position && i < text_.size(); ++i) {
        if (text_[i] == '\n') {
            lineStart = i + 1;
        }
    }
    return position - lineStart + 1;
}

std::string readTextFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open file: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("cannot read file: " + path);
    }
    return buffer.str();
}

std::string jsonPath(const std::string& base, const std::string& key) {
    return base + "." + key;
}

std::string jsonPath(const std::string& base, std::size_t index) {
    return base + "[" + std::to_string(index) + "]";
}

}  // namespace tilegen
