#include "json.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

namespace orv {
namespace {

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text) {}

  bool parse(JsonValue& value, JsonParseError& error) {
    skipWhitespace();
    if (!parseValue(value)) {
      error = error_;
      return false;
    }
    skipWhitespace();
    if (pos_ != text_.size()) {
      setError("unexpected trailing characters");
      error = error_;
      return false;
    }
    return true;
  }

 private:
  bool parseValue(JsonValue& value) {
    if (pos_ >= text_.size()) {
      return setError("unexpected end of input");
    }
    switch (text_[pos_]) {
      case 'n':
        return parseLiteral("null", value);
      case 't':
        value = JsonValue(true);
        return parseLiteral("true");
      case 'f':
        value = JsonValue(false);
        return parseLiteral("false");
      case '"': {
        std::string string;
        if (!parseString(string)) {
          return false;
        }
        value = JsonValue(std::move(string));
        return true;
      }
      case '[':
        return parseArray(value);
      case '{':
        return parseObject(value);
      default:
        if (text_[pos_] == '-' || (text_[pos_] >= '0' && text_[pos_] <= '9')) {
          return parseNumber(value);
        }
        return setError("unexpected character while parsing value");
    }
  }

  bool parseLiteral(const char* literal) {
    const std::size_t length = std::char_traits<char>::length(literal);
    if (text_.compare(pos_, length, literal) != 0) {
      return setError("invalid literal");
    }
    pos_ += length;
    return true;
  }

  bool parseLiteral(const char* literal, JsonValue& value) {
    value = JsonValue();
    return parseLiteral(literal);
  }

  bool parseArray(JsonValue& value) {
    ++pos_;
    skipWhitespace();
    JsonValue::Array array;
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
      value = JsonValue(std::move(array));
      return true;
    }
    while (true) {
      JsonValue element;
      if (!parseValue(element)) {
        return false;
      }
      array.push_back(std::move(element));
      skipWhitespace();
      if (pos_ >= text_.size()) {
        return setError("unterminated array");
      }
      if (text_[pos_] == ',') {
        ++pos_;
        skipWhitespace();
        continue;
      }
      if (text_[pos_] == ']') {
        ++pos_;
        value = JsonValue(std::move(array));
        return true;
      }
      return setError("expected ',' or ']'");
    }
  }

  bool parseObject(JsonValue& value) {
    ++pos_;
    skipWhitespace();
    JsonValue::Object object;
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      value = JsonValue(std::move(object));
      return true;
    }
    while (true) {
      if (pos_ >= text_.size() || text_[pos_] != '"') {
        return setError("expected object key string");
      }
      std::string key;
      if (!parseString(key)) {
        return false;
      }
      skipWhitespace();
      if (pos_ >= text_.size() || text_[pos_] != ':') {
        return setError("expected ':' after object key");
      }
      ++pos_;
      skipWhitespace();
      JsonValue element;
      if (!parseValue(element)) {
        return false;
      }
      if (!object.emplace(key, std::move(element)).second) {
        return setError("duplicate object key: " + key);
      }
      skipWhitespace();
      if (pos_ >= text_.size()) {
        return setError("unterminated object");
      }
      if (text_[pos_] == ',') {
        ++pos_;
        skipWhitespace();
        continue;
      }
      if (text_[pos_] == '}') {
        ++pos_;
        value = JsonValue(std::move(object));
        return true;
      }
      return setError("expected ',' or '}'");
    }
  }

  bool parseString(std::string& output) {
    ++pos_;
    while (pos_ < text_.size()) {
      const unsigned char ch = static_cast<unsigned char>(text_[pos_++]);
      if (ch == '"') {
        return true;
      }
      if (ch < 0x20) {
        return setError("unescaped control character in string");
      }
      if (ch != '\\') {
        output.push_back(static_cast<char>(ch));
        continue;
      }
      if (pos_ >= text_.size()) {
        return setError("unterminated escape sequence");
      }
      const char escaped = text_[pos_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          output.push_back(escaped);
          break;
        case 'b':
          output.push_back('\b');
          break;
        case 'f':
          output.push_back('\f');
          break;
        case 'n':
          output.push_back('\n');
          break;
        case 'r':
          output.push_back('\r');
          break;
        case 't':
          output.push_back('\t');
          break;
        case 'u':
          if (!parseUnicodeEscape(output)) {
            return false;
          }
          break;
        default:
          return setError("invalid string escape");
      }
    }
    return setError("unterminated string");
  }

  bool parseUnicodeEscape(std::string& output) {
    if (pos_ + 4 > text_.size()) {
      return setError("truncated unicode escape");
    }
    unsigned code = 0;
    for (int i = 0; i < 4; ++i) {
      const char ch = text_[pos_++];
      code <<= 4;
      if (ch >= '0' && ch <= '9') {
        code |= static_cast<unsigned>(ch - '0');
      } else if (ch >= 'a' && ch <= 'f') {
        code |= static_cast<unsigned>(ch - 'a' + 10);
      } else if (ch >= 'A' && ch <= 'F') {
        code |= static_cast<unsigned>(ch - 'A' + 10);
      } else {
        return setError("invalid unicode escape");
      }
    }
    if (code >= 0xD800 && code <= 0xDBFF) {
      if (pos_ + 6 > text_.size() || text_[pos_] != '\\' || text_[pos_ + 1] != 'u') {
        return setError("unpaired high surrogate");
      }
      pos_ += 2;
      unsigned low = 0;
      for (int i = 0; i < 4; ++i) {
        const char ch = text_[pos_++];
        low <<= 4;
        if (ch >= '0' && ch <= '9') {
          low |= static_cast<unsigned>(ch - '0');
        } else if (ch >= 'a' && ch <= 'f') {
          low |= static_cast<unsigned>(ch - 'a' + 10);
        } else if (ch >= 'A' && ch <= 'F') {
          low |= static_cast<unsigned>(ch - 'A' + 10);
        } else {
          return setError("invalid low surrogate");
        }
      }
      if (low < 0xDC00 || low > 0xDFFF) {
        return setError("invalid low surrogate");
      }
      code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
    } else if (code >= 0xDC00 && code <= 0xDFFF) {
      return setError("unpaired low surrogate");
    }
    appendUtf8(output, code);
    return true;
  }

  static void appendUtf8(std::string& output, unsigned code) {
    if (code <= 0x7F) {
      output.push_back(static_cast<char>(code));
    } else if (code <= 0x7FF) {
      output.push_back(static_cast<char>(0xC0 | (code >> 6)));
      output.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else if (code <= 0xFFFF) {
      output.push_back(static_cast<char>(0xE0 | (code >> 12)));
      output.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      output.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
      output.push_back(static_cast<char>(0xF0 | (code >> 18)));
      output.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
      output.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      output.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
  }

  bool parseNumber(JsonValue& value) {
    const std::size_t start = pos_;
    if (text_[pos_] == '-') {
      ++pos_;
    }
    if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
      return setError("invalid number");
    }
    if (text_[pos_] == '0') {
      ++pos_;
      if (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        return setError("leading zero in number");
      }
    } else {
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
        return setError("missing fractional digits");
      }
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
        return setError("missing exponent digits");
      }
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        ++pos_;
      }
    }

    const std::string token = text_.substr(start, pos_ - start);
    char* end = nullptr;
    const double number = std::strtod(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0' || !std::isfinite(number)) {
      return setError("number is not finite");
    }
    value = JsonValue(number);
    return true;
  }

  void skipWhitespace() {
    while (pos_ < text_.size()) {
      const char ch = text_[pos_];
      if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
        break;
      }
      ++pos_;
      if (ch == '\n') {
        ++line_;
        column_ = 1;
      } else {
        ++column_;
      }
    }
  }

  bool setError(std::string message) {
    error_.message = std::move(message);
    error_.line = line_;
    error_.column = column_;
    return false;
  }

  const std::string& text_;
  std::size_t pos_ = 0;
  std::size_t line_ = 1;
  std::size_t column_ = 1;
  JsonParseError error_;
};

}  // namespace

bool parseJson(const std::string& text, JsonValue& value, JsonParseError& error) {
  Parser parser(text);
  if (!parser.parse(value, error)) {
    return false;
  }
  return true;
}

std::string jsonEscape(const std::string& input) {
  std::ostringstream stream;
  for (const unsigned char ch : input) {
    switch (ch) {
      case '"':
        stream << "\\\"";
        break;
      case '\\':
        stream << "\\\\";
        break;
      case '\b':
        stream << "\\b";
        break;
      case '\f':
        stream << "\\f";
        break;
      case '\n':
        stream << "\\n";
        break;
      case '\r':
        stream << "\\r";
        break;
      case '\t':
        stream << "\\t";
        break;
      default:
        if (ch < 0x20) {
          stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned>(ch) << std::dec;
        } else {
          stream << static_cast<char>(ch);
        }
    }
  }
  return stream.str();
}

std::string jsonNumber(double value, int precision) {
  if (!std::isfinite(value)) {
    return "null";
  }
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(precision) << value;
  std::string result = stream.str();
  if (result == "-0") {
    result = "0";
  }
  return result;
}

}  // namespace orv
