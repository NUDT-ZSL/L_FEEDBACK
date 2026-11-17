#include "json.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <system_error>
#include <utility>

namespace softraster {

JsonValue::JsonValue() : value_(nullptr) {}
JsonValue::JsonValue(std::nullptr_t) : value_(nullptr) {}
JsonValue::JsonValue(bool value) : value_(value) {}
JsonValue::JsonValue(double value) : value_(value) {}
JsonValue::JsonValue(std::string value) : value_(std::move(value)) {}
JsonValue::JsonValue(Array value) : value_(std::move(value)) {}
JsonValue::JsonValue(Object value) : value_(std::move(value)) {}

bool JsonValue::isNull() const { return std::holds_alternative<std::nullptr_t>(value_); }
bool JsonValue::isBool() const { return std::holds_alternative<bool>(value_); }
bool JsonValue::isNumber() const { return std::holds_alternative<double>(value_); }
bool JsonValue::isString() const { return std::holds_alternative<std::string>(value_); }
bool JsonValue::isArray() const { return std::holds_alternative<Array>(value_); }
bool JsonValue::isObject() const { return std::holds_alternative<Object>(value_); }

bool JsonValue::asBool() const { return std::get<bool>(value_); }
double JsonValue::asNumber() const { return std::get<double>(value_); }
const std::string& JsonValue::asString() const { return std::get<std::string>(value_); }
const JsonValue::Array& JsonValue::asArray() const { return std::get<Array>(value_); }
JsonValue::Array& JsonValue::asArray() { return std::get<Array>(value_); }
const JsonValue::Object& JsonValue::asObject() const { return std::get<Object>(value_); }
JsonValue::Object& JsonValue::asObject() { return std::get<Object>(value_); }

const JsonValue* JsonValue::find(std::string_view key) const {
    if (!isObject()) {
        return nullptr;
    }
    const auto& object = asObject();
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

const JsonValue::Storage& JsonValue::storage() const { return value_; }

namespace {

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    JsonParseResult parse() {
        skipWhitespace();
        if (atEnd()) {
            fail("empty JSON document");
            return failResult();
        }
        JsonValue root;
        if (!parseValue(root)) {
            return failResult();
        }
        skipWhitespace();
        if (!atEnd()) {
            fail("unexpected trailing characters");
            return failResult();
        }
        return JsonParseResult{std::optional<JsonValue>(std::move(root)), {}, 0};
    }

private:
    std::string_view text_;
    std::size_t offset_ = 0;
    std::string error_;
    std::size_t errorOffset_ = 0;

    [[nodiscard]] bool atEnd() const { return offset_ >= text_.size(); }
    [[nodiscard]] char peek() const { return atEnd() ? '\0' : text_[offset_]; }

    void skipWhitespace() {
        while (!atEnd()) {
            const char ch = text_[offset_];
            if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
                ++offset_;
            } else {
                break;
            }
        }
    }

    bool fail(std::string message) {
        if (error_.empty()) {
            error_ = std::move(message);
            errorOffset_ = offset_;
        }
        return false;
    }

    JsonParseResult failResult() const {
        return JsonParseResult{std::nullopt, error_, errorOffset_};
    }

    bool consume(char expected) {
        if (peek() != expected) {
            return fail(std::string("expected '") + expected + "'");
        }
        ++offset_;
        return true;
    }

    bool parseValue(JsonValue& out) {
        skipWhitespace();
        if (atEnd()) {
            return fail("unexpected end of input");
        }
        switch (peek()) {
            case 'n': return parseLiteral("null", JsonValue(nullptr), out);
            case 't': return parseLiteral("true", JsonValue(true), out);
            case 'f': return parseLiteral("false", JsonValue(false), out);
            case '"': {
                std::string string;
                if (!parseString(string)) {
                    return false;
                }
                out = JsonValue(std::move(string));
                return true;
            }
            case '[': return parseArray(out);
            case '{': return parseObject(out);
            default:
                if (peek() == '-' || (peek() >= '0' && peek() <= '9')) {
                    double number = 0.0;
                    if (!parseNumber(number)) {
                        return false;
                    }
                    out = JsonValue(number);
                    return true;
                }
                return fail("unexpected token");
        }
    }

    bool parseLiteral(std::string_view literal, JsonValue value, JsonValue& out) {
        if (text_.substr(offset_, literal.size()) != literal) {
            return fail("invalid literal");
        }
        offset_ += literal.size();
        out = std::move(value);
        return true;
    }

    static int hexValue(char ch) {
        if (ch >= '0' && ch <= '9') return ch - '0';
        if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
        if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
        return -1;
    }

    bool parseHex4(std::uint32_t& value) {
        if (offset_ + 4 > text_.size()) {
            return fail("truncated unicode escape");
        }
        value = 0;
        for (int i = 0; i < 4; ++i) {
            const int digit = hexValue(text_[offset_++]);
            if (digit < 0) {
                return fail("invalid unicode escape");
            }
            value = (value << 4U) | static_cast<std::uint32_t>(digit);
        }
        return true;
    }

    static void appendUtf8(std::string& out, std::uint32_t codepoint) {
        if (codepoint <= 0x7fU) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ffU) {
            out.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
            out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0xffffU) {
            out.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
            out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else {
            out.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
            out.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
            out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        }
    }

    bool parseString(std::string& out) {
        if (!consume('"')) {
            return false;
        }
        while (!atEnd()) {
            const unsigned char ch = static_cast<unsigned char>(text_[offset_++]);
            if (ch == '"') {
                return true;
            }
            if (ch < 0x20U) {
                return fail("unescaped control character in string");
            }
            if (ch != '\\') {
                out.push_back(static_cast<char>(ch));
                continue;
            }
            if (atEnd()) {
                return fail("truncated string escape");
            }
            const char escaped = text_[offset_++];
            switch (escaped) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    std::uint32_t codepoint = 0;
                    if (!parseHex4(codepoint)) {
                        return false;
                    }
                    if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
                        if (offset_ + 2 > text_.size() || text_[offset_] != '\\' ||
                            text_[offset_ + 1] != 'u') {
                            return fail("missing low surrogate");
                        }
                        offset_ += 2;
                        std::uint32_t low = 0;
                        if (!parseHex4(low) || low < 0xdc00U || low > 0xdfffU) {
                            return fail("invalid low surrogate");
                        }
                        codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) +
                                    (low - 0xdc00U);
                    } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
                        return fail("unexpected low surrogate");
                    }
                    appendUtf8(out, codepoint);
                    break;
                }
                default: return fail("invalid string escape");
            }
        }
        return fail("unterminated string");
    }

    bool parseNumber(double& out) {
        const std::size_t start = offset_;
        if (peek() == '-') {
            ++offset_;
        }
        if (atEnd()) {
            return fail("truncated number");
        }
        if (peek() == '0') {
            ++offset_;
            if (!atEnd() && peek() >= '0' && peek() <= '9') {
                return fail("leading zero in number");
            }
        } else if (peek() >= '1' && peek() <= '9') {
            while (!atEnd() && peek() >= '0' && peek() <= '9') {
                ++offset_;
            }
        } else {
            return fail("invalid number");
        }
        if (!atEnd() && peek() == '.') {
            ++offset_;
            if (atEnd() || peek() < '0' || peek() > '9') {
                return fail("missing fractional digits");
            }
            while (!atEnd() && peek() >= '0' && peek() <= '9') {
                ++offset_;
            }
        }
        if (!atEnd() && (peek() == 'e' || peek() == 'E')) {
            ++offset_;
            if (!atEnd() && (peek() == '+' || peek() == '-')) {
                ++offset_;
            }
            if (atEnd() || peek() < '0' || peek() > '9') {
                return fail("missing exponent digits");
            }
            while (!atEnd() && peek() >= '0' && peek() <= '9') {
                ++offset_;
            }
        }

        const char* begin = text_.data() + start;
        const char* end = text_.data() + offset_;
        double parsed = 0.0;
        const auto conversion = std::from_chars(begin, end, parsed, std::chars_format::general);
        if (conversion.ec == std::errc::result_out_of_range) {
            out = (text_[start] == '-') ? -std::numeric_limits<double>::infinity()
                                        : std::numeric_limits<double>::infinity();
            return true;
        }
        if (conversion.ec != std::errc() || conversion.ptr != end) {
            return fail("invalid number");
        }
        out = parsed;
        return true;
    }

    bool parseArray(JsonValue& out) {
        if (!consume('[')) {
            return false;
        }
        JsonValue::Array values;
        skipWhitespace();
        if (!atEnd() && peek() == ']') {
            ++offset_;
            out = JsonValue(std::move(values));
            return true;
        }
        while (true) {
            JsonValue value;
            if (!parseValue(value)) {
                return false;
            }
            values.push_back(std::move(value));
            skipWhitespace();
            if (atEnd()) {
                return fail("unterminated array");
            }
            if (peek() == ']') {
                ++offset_;
                out = JsonValue(std::move(values));
                return true;
            }
            if (peek() != ',') {
                return fail("expected ',' or ']'");
            }
            ++offset_;
            skipWhitespace();
            if (!atEnd() && peek() == ']') {
                return fail("trailing comma in array");
            }
        }
    }

    bool parseObject(JsonValue& out) {
        if (!consume('{')) {
            return false;
        }
        JsonValue::Object object;
        skipWhitespace();
        if (!atEnd() && peek() == '}') {
            ++offset_;
            out = JsonValue(std::move(object));
            return true;
        }
        while (true) {
            skipWhitespace();
            if (atEnd() || peek() != '"') {
                return fail("expected object key");
            }
            std::string key;
            if (!parseString(key)) {
                return false;
            }
            skipWhitespace();
            if (!consume(':')) {
                return false;
            }
            JsonValue value;
            if (!parseValue(value)) {
                return false;
            }
            const auto inserted = object.emplace(key, std::move(value));
            if (!inserted.second) {
                return fail("duplicate object key");
            }
            skipWhitespace();
            if (atEnd()) {
                return fail("unterminated object");
            }
            if (peek() == '}') {
                ++offset_;
                out = JsonValue(std::move(object));
                return true;
            }
            if (peek() != ',') {
                return fail("expected ',' or '}'");
            }
            ++offset_;
            skipWhitespace();
            if (!atEnd() && peek() == '}') {
                return fail("trailing comma in object");
            }
        }
    }
};

void writeIndent(std::ostringstream& out, int indent) {
    for (int i = 0; i < indent; ++i) {
        out.put(' ');
    }
}

void writeEscapedString(std::ostringstream& out, std::string_view text) {
    out.put('"');
    for (const unsigned char ch : text) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20U) {
                    const char* hex = "0123456789abcdef";
                    out << "\\u00" << hex[(ch >> 4U) & 0x0fU] << hex[ch & 0x0fU];
                } else {
                    out.put(static_cast<char>(ch));
                }
        }
    }
    out.put('"');
}

void writeValue(std::ostringstream& out, const JsonValue& value, bool pretty, int indent) {
    if (value.isNull()) {
        out << "null";
    } else if (value.isBool()) {
        out << (value.asBool() ? "true" : "false");
    } else if (value.isNumber()) {
        const double number = value.asNumber();
        if (!std::isfinite(number)) {
            out << "null";
        } else if (number == 0.0) {
            out << '0';
        } else {
            out << std::setprecision(17) << number;
        }
    } else if (value.isString()) {
        writeEscapedString(out, value.asString());
    } else if (value.isArray()) {
        const auto& array = value.asArray();
        if (array.empty()) {
            out << "[]";
            return;
        }
        out << '[';
        for (std::size_t i = 0; i < array.size(); ++i) {
            if (i != 0) {
                out << ',';
            }
            if (pretty) {
                out << '\n';
                writeIndent(out, indent + 2);
            }
            writeValue(out, array[i], pretty, indent + 2);
        }
        if (pretty) {
            out << '\n';
            writeIndent(out, indent);
        }
        out << ']';
    } else {
        const auto& object = value.asObject();
        if (object.empty()) {
            out << "{}";
            return;
        }
        out << '{';
        bool first = true;
        for (const auto& [key, child] : object) {
            if (!first) {
                out << ',';
            }
            first = false;
            if (pretty) {
                out << '\n';
                writeIndent(out, indent + 2);
            }
            writeEscapedString(out, key);
            out << (pretty ? ": " : ":");
            writeValue(out, child, pretty, indent + 2);
        }
        if (pretty) {
            out << '\n';
            writeIndent(out, indent);
        }
        out << '}';
    }
}

}  // namespace

JsonParseResult parseJson(std::string_view text) {
    Parser parser(text);
    return parser.parse();
}

std::string writeJson(const JsonValue& value, bool pretty) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    writeValue(out, value, pretty, 0);
    out << '\n';
    return out.str();
}

}  // namespace softraster
