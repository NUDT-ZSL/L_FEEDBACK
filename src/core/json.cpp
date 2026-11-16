#include "core/json.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <system_error>

namespace uid166 {

namespace {

void appendIndent(std::string& out, size_t depth) {
    out.append(depth * 2, ' ');
}

void appendUtf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7f) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0x10ffff) {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
}

class JsonParser {
public:
    JsonParser(std::string_view text, ErrorCode parseCode, ErrorCode schemaCode)
        : text_(text), parseCode_(parseCode), schemaCode_(schemaCode) {}

    JsonValue parse() {
        skipWhitespace();
        JsonValue value = parseValue(0);
        skipWhitespace();
        if (pos_ != text_.size()) fail("trailing characters");
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& message) const {
        throw AppError(parseCode_, "JSON at byte " + std::to_string(pos_) + ": " + message);
    }

    void skipWhitespace() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
            ++pos_;
        }
    }

    bool consume(char c) {
        if (pos_ < text_.size() && text_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    void expect(char c) {
        if (!consume(c)) fail(std::string("expected '") + c + "'");
    }

    JsonValue parseValue(int depth) {
        if (depth > 128) fail("nesting limit exceeded");
        if (++nodes_ > 6'000'000) fail("node limit exceeded");
        skipWhitespace();
        if (pos_ >= text_.size()) fail("unexpected end of input");
        const char c = text_[pos_];
        if (c == '{') return parseObject(depth + 1);
        if (c == '[') return parseArray(depth + 1);
        if (c == '"') {
            JsonValue v;
            v.type = JsonType::stringValue;
            v.string = parseString();
            return v;
        }
        if (c == 't') {
            literal("true");
            JsonValue v;
            v.type = JsonType::booleanValue;
            v.boolean = true;
            return v;
        }
        if (c == 'f') {
            literal("false");
            JsonValue v;
            v.type = JsonType::booleanValue;
            v.boolean = false;
            return v;
        }
        if (c == 'n') {
            literal("null");
            return {};
        }
        if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
        fail("unexpected token");
    }

    void literal(std::string_view expected) {
        if (text_.substr(pos_, expected.size()) != expected) fail("invalid literal");
        pos_ += expected.size();
    }

    JsonValue parseObject(int depth) {
        expect('{');
        JsonValue v;
        v.type = JsonType::objectValue;
        skipWhitespace();
        if (consume('}')) return v;
        while (true) {
            skipWhitespace();
            if (pos_ >= text_.size() || text_[pos_] != '"') fail("object key must be a string");
            std::string name = parseString();
            skipWhitespace();
            expect(':');
            JsonValue member = parseValue(depth);
            v.object.emplace_back(std::move(name), std::move(member));
            skipWhitespace();
            if (consume('}')) break;
            expect(',');
        }
        return v;
    }

    JsonValue parseArray(int depth) {
        expect('[');
        JsonValue v;
        v.type = JsonType::arrayValue;
        skipWhitespace();
        if (consume(']')) return v;
        while (true) {
            v.array.push_back(parseValue(depth));
            skipWhitespace();
            if (consume(']')) break;
            expect(',');
        }
        return v;
    }

    static int hexValue(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    uint32_t parseHex4() {
        if (pos_ + 4 > text_.size()) fail("short unicode escape");
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const int h = hexValue(text_[pos_++]);
            if (h < 0) fail("invalid unicode escape");
            value = (value << 4) | static_cast<uint32_t>(h);
        }
        return value;
    }

    std::string parseString() {
        expect('"');
        std::string out;
        while (pos_ < text_.size()) {
            const unsigned char c = static_cast<unsigned char>(text_[pos_++]);
            if (c == '"') return out;
            if (c < 0x20) fail("control character in string");
            if (c != '\\') {
                out.push_back(static_cast<char>(c));
                continue;
            }
            if (pos_ >= text_.size()) fail("short escape");
            const char e = text_[pos_++];
            switch (e) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                uint32_t cp = parseHex4();
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    if (pos_ + 2 > text_.size() || text_[pos_] != '\\' || text_[pos_ + 1] != 'u') {
                        fail("missing low surrogate");
                    }
                    pos_ += 2;
                    const uint32_t low = parseHex4();
                    if (low < 0xdc00 || low > 0xdfff) fail("invalid low surrogate");
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                } else if (cp >= 0xdc00 && cp <= 0xdfff) {
                    fail("unexpected low surrogate");
                }
                appendUtf8(out, cp);
                break;
            }
            default: fail("invalid escape");
            }
        }
        fail("unterminated string");
    }

    JsonValue parseNumber() {
        const size_t start = pos_;
        consume('-');
        if (pos_ >= text_.size()) fail("short number");
        if (text_[pos_] == '0') {
            ++pos_;
        } else {
            if (text_[pos_] < '1' || text_[pos_] > '9') fail("invalid number");
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        if (consume('.')) {
            if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
                fail("fraction requires digits");
            }
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
                fail("exponent requires digits");
            }
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        double value = 0.0;
        const auto result = std::from_chars(text_.data() + start, text_.data() + pos_, value);
        if (result.ec != std::errc{} || result.ptr != text_.data() + pos_ || !std::isfinite(value)) {
            fail("number out of range");
        }
        JsonValue v;
        v.type = JsonType::numberValue;
        v.number = value;
        return v;
    }

    std::string_view text_;
    ErrorCode parseCode_;
    ErrorCode schemaCode_;
    size_t pos_ = 0;
    size_t nodes_ = 0;
};

} // namespace

void JsonWriter::beforeValue() {
    if (afterKey_) {
        afterKey_ = false;
        return;
    }
    if (stack_.empty()) return;
    if (!counts_.back()) {
        if (pretty_) out_.push_back('\n');
    } else {
        out_.push_back(',');
        if (pretty_) out_.push_back('\n');
    }
    if (pretty_) appendIndent(out_, stack_.size());
    ++counts_.back();
}

void JsonWriter::newline() {
    if (pretty_) {
        out_.push_back('\n');
        appendIndent(out_, stack_.empty() ? 0 : stack_.size() - 1);
    }
}

void JsonWriter::beginObject() {
    beforeValue();
    out_.push_back('{');
    stack_.push_back(Container::object);
    counts_.push_back(0);
}

void JsonWriter::endObject() {
    const bool had = counts_.back() != 0;
    stack_.pop_back();
    counts_.pop_back();
    if (had) newline();
    out_.push_back('}');
}

void JsonWriter::beginArray() {
    beforeValue();
    out_.push_back('[');
    stack_.push_back(Container::array);
    counts_.push_back(0);
}

void JsonWriter::endArray() {
    const bool had = counts_.back() != 0;
    stack_.pop_back();
    counts_.pop_back();
    if (had) newline();
    out_.push_back(']');
}

void JsonWriter::key(std::string_view value) {
    beforeValue();
    appendEscaped(value);
    out_.push_back(':');
    if (pretty_) out_.push_back(' ');
    afterKey_ = true;
}

void JsonWriter::string(std::string_view value) {
    beforeValue();
    appendEscaped(value);
}

void JsonWriter::number(double value) {
    beforeValue();
    if (!std::isfinite(value)) {
        out_ += "null";
        return;
    }
    char buf[64];
    const auto result = std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::general, 9);
    if (result.ec == std::errc{}) out_.append(buf, result.ptr);
    else out_.push_back('0');
}

void JsonWriter::integer(int64_t value) {
    beforeValue();
    char buf[32];
    const auto result = std::to_chars(buf, buf + sizeof(buf), value);
    out_.append(buf, result.ptr);
}

void JsonWriter::boolean(bool value) {
    beforeValue();
    out_ += value ? "true" : "false";
}

void JsonWriter::null() {
    beforeValue();
    out_ += "null";
}

void JsonWriter::appendEscaped(std::string_view value) {
    out_.push_back('"');
    for (const char ch : value) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
        case '"': out_ += "\\\""; break;
        case '\\': out_ += "\\\\"; break;
        case '\b': out_ += "\\b"; break;
        case '\f': out_ += "\\f"; break;
        case '\n': out_ += "\\n"; break;
        case '\r': out_ += "\\r"; break;
        case '\t': out_ += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                out_ += buf;
            } else {
                out_.push_back(ch);
            }
        }
    }
    out_.push_back('"');
}

void writeTextFile(const std::filesystem::path& path, std::string_view text) {
    std::error_code ec;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) throw AppError(ErrorCode::outputWrite, "cannot create output directory: " + ec.message());
    }
    const std::filesystem::path temp = path.string() + ".tmp";
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) throw AppError(ErrorCode::outputWrite, "cannot open output file: " + path.string());
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!out) throw AppError(ErrorCode::outputWrite, "cannot write output file: " + path.string());
    }
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(temp, path, ec);
    if (ec) {
        std::filesystem::remove(temp, ec);
        throw AppError(ErrorCode::outputWrite, "cannot replace output file: " + path.string());
    }
}

std::string readTextFile(const std::filesystem::path& path, ErrorCode missingCode,
                         ErrorCode readCode, size_t maxBytes) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        throw AppError(missingCode, "file not found: " + path.string());
    }
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) throw AppError(readCode, "cannot stat file: " + path.string());
    if (size > maxBytes) throw AppError(readCode, "file exceeds size limit: " + path.string());
    std::ifstream in(path, std::ios::binary);
    if (!in) throw AppError(readCode, "cannot open file: " + path.string());
    std::string text(static_cast<size_t>(size), '\0');
    if (!text.empty()) in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in || static_cast<size_t>(in.gcount()) != text.size()) {
        throw AppError(readCode, "short read: " + path.string());
    }
    return text;
}

const JsonValue* JsonValue::find(std::string_view name) const {
    for (const auto& [key, value] : object) {
        if (key == name) return &value;
    }
    return nullptr;
}

JsonValue parseJson(std::string_view text, ErrorCode parseCode, ErrorCode schemaCode) {
    return JsonParser(text, parseCode, schemaCode).parse();
}

JsonValue parseJsonFile(const std::filesystem::path& path, ErrorCode missingCode,
                        ErrorCode readCode, ErrorCode parseCode, ErrorCode schemaCode) {
    return parseJson(readTextFile(path, missingCode, readCode), parseCode, schemaCode);
}

const JsonValue& requireObject(const JsonValue& value, std::string_view where,
                               ErrorCode schemaCode) {
    if (value.type != JsonType::objectValue) {
        throw AppError(schemaCode, std::string(where) + " must be an object");
    }
    return value;
}

const JsonValue& requireArray(const JsonValue& value, std::string_view where,
                              ErrorCode schemaCode) {
    if (value.type != JsonType::arrayValue) {
        throw AppError(schemaCode, std::string(where) + " must be an array");
    }
    return value;
}

const JsonValue& requireMember(const JsonValue& object, std::string_view name,
                               ErrorCode schemaCode) {
    (void)requireObject(object, "value", schemaCode);
    const JsonValue* member = object.find(name);
    if (!member) throw AppError(schemaCode, "missing member: " + std::string(name));
    return *member;
}

double requireNumber(const JsonValue& value, std::string_view where, ErrorCode schemaCode) {
    if (value.type != JsonType::numberValue || !std::isfinite(value.number)) {
        throw AppError(schemaCode, std::string(where) + " must be a finite number");
    }
    return value.number;
}

int64_t requireInteger(const JsonValue& value, std::string_view where, ErrorCode schemaCode) {
    const double n = requireNumber(value, where, schemaCode);
    if (n < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
        n > static_cast<double>(std::numeric_limits<int64_t>::max()) ||
        std::floor(n) != n) {
        throw AppError(schemaCode, std::string(where) + " must be an integer");
    }
    return static_cast<int64_t>(n);
}

std::string requireString(const JsonValue& value, std::string_view where, ErrorCode schemaCode) {
    if (value.type != JsonType::stringValue) {
        throw AppError(schemaCode, std::string(where) + " must be a string");
    }
    return value.string;
}

} // namespace uid166

