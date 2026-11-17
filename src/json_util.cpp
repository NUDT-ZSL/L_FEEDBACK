#include "json_util.hpp"

#include "errors.hpp"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <system_error>

namespace terrain {
namespace {

std::optional<std::string> extractValueText(const std::string& document,
                                            const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const size_t keyPosition = document.find(needle);
  if (keyPosition == std::string::npos) {
    return std::nullopt;
  }
  const size_t colon = document.find(':', keyPosition + needle.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  size_t begin = colon + 1;
  while (begin < document.size() &&
         (document[begin] == ' ' || document[begin] == '\t' ||
          document[begin] == '\r' || document[begin] == '\n')) {
    ++begin;
  }
  if (begin >= document.size()) {
    return std::nullopt;
  }
  if (document[begin] == '"') {
    size_t end = begin + 1;
    bool escaped = false;
    while (end < document.size()) {
      if (escaped) {
        escaped = false;
      } else if (document[end] == '\\') {
        escaped = true;
      } else if (document[end] == '"') {
        return document.substr(begin + 1, end - begin - 1);
      }
      ++end;
    }
    return std::nullopt;
  }
  if (document[begin] == '[') {
    const size_t end = document.find(']', begin + 1);
    if (end == std::string::npos) {
      return std::nullopt;
    }
    return document.substr(begin, end - begin + 1);
  }
  size_t end = begin;
  while (end < document.size() &&
         document[end] != ',' && document[end] != '}' &&
         document[end] != ']' && document[end] != '\r' &&
         document[end] != '\n') {
    ++end;
  }
  if (end == begin) {
    return std::nullopt;
  }
  return document.substr(begin, end - begin);
}

bool parseDouble(const std::string& text, double& value) {
  try {
    size_t consumed = 0;
    value = std::stod(text, &consumed);
    while (consumed < text.size() &&
           (text[consumed] == ' ' || text[consumed] == '\t' ||
            text[consumed] == '\r' || text[consumed] == '\n')) {
      ++consumed;
    }
    return consumed == text.size() && std::isfinite(value);
  } catch (...) {
    return false;
  }
}

}  // namespace

std::string jsonEscape(const std::string& input) {
  std::string result;
  result.reserve(input.size() + 8);
  for (const char character : input) {
    const unsigned char value = static_cast<unsigned char>(character);
    switch (character) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\b':
        result += "\\b";
        break;
      case '\f':
        result += "\\f";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (value < 0x20) {
          std::ostringstream escape;
          escape << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(value);
          result += escape.str();
        } else {
          result.push_back(character);
        }
        break;
    }
  }
  return result;
}

std::string jsonNumber(float value, int precision) {
  if (!std::isfinite(value) || std::abs(value) < std::pow(10.0f, -precision)) {
    value = 0.0f;
  }
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string jsonNumber(double value, int precision) {
  if (!std::isfinite(value) || std::abs(value) < std::pow(10.0, -precision)) {
    value = 0.0;
  }
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string jsonVec3(const Vec3& value, int precision) {
  return "[" + jsonNumber(value.x, precision) + "," +
         jsonNumber(value.y, precision) + "," +
         jsonNumber(value.z, precision) + "]";
}

void writeTextFileAtomic(const std::filesystem::path& path, const std::string& text) {
  if (!path.parent_path().empty()) {
    std::error_code directoryError;
    std::filesystem::create_directories(path.parent_path(), directoryError);
    if (directoryError) {
      throw businessError("IO_ERROR",
                          "Cannot create output directory: " +
                              path.parent_path().string());
    }
  }
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      throw businessError("IO_ERROR", "Cannot create output file: " + temporary.string());
    }
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    stream.flush();
    if (!stream) {
      std::filesystem::remove(temporary);
      throw businessError("IO_ERROR", "Failed to write output file: " + path.string());
    }
  }

  std::error_code error;
  std::filesystem::remove(path, error);
  error.clear();
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary);
    throw businessError("IO_ERROR", "Cannot publish output file: " + path.string());
  }
}

std::optional<std::string> readTextFile(const std::filesystem::path& path,
                                        size_t maximumBytes) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    return std::nullopt;
  }
  const std::streampos end = stream.tellg();
  if (end < 0 || static_cast<uint64_t>(end) > maximumBytes) {
    return std::nullopt;
  }
  std::string document(static_cast<size_t>(end), '\0');
  stream.seekg(0, std::ios::beg);
  if (!document.empty()) {
    stream.read(document.data(), static_cast<std::streamsize>(document.size()));
    if (!stream) {
      return std::nullopt;
    }
  }
  return document;
}

std::optional<double> jsonReadNumber(const std::string& document, const std::string& key) {
  const std::optional<std::string> text = extractValueText(document, key);
  if (!text.has_value()) {
    return std::nullopt;
  }
  double value = 0.0;
  if (!parseDouble(*text, value)) {
    return std::nullopt;
  }
  return value;
}

std::optional<long long> jsonReadInteger(const std::string& document, const std::string& key) {
  const std::optional<double> number = jsonReadNumber(document, key);
  if (!number.has_value() || std::floor(*number) != *number) {
    return std::nullopt;
  }
  return static_cast<long long>(*number);
}

std::optional<std::string> jsonReadString(const std::string& document, const std::string& key) {
  const std::optional<std::string> text = extractValueText(document, key);
  if (!text.has_value()) {
    return std::nullopt;
  }
  return text;
}

std::optional<Vec3> jsonReadVec3(const std::string& document, const std::string& key) {
  const std::optional<std::string> text = extractValueText(document, key);
  if (!text.has_value() || text->size() < 2 || text->front() != '[' ||
      text->back() != ']') {
    return std::nullopt;
  }
  const std::string body = text->substr(1, text->size() - 2);
  double values[3] = {};
  size_t start = 0;
  for (int i = 0; i < 3; ++i) {
    const size_t comma = body.find(',', start);
    const size_t end = (i == 2) ? body.size() : comma;
    if (end == std::string::npos) {
      return std::nullopt;
    }
    if (!parseDouble(body.substr(start, end - start), values[i])) {
      return std::nullopt;
    }
    start = end + 1;
  }
  return Vec3(static_cast<float>(values[0]), static_cast<float>(values[1]),
              static_cast<float>(values[2]));
}

std::string errorJson(const std::string& code, const std::string& message, int exitCode) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << "{\n"
         << "  \"ok\": false,\n"
         << "  \"error\": {\n"
         << "    \"code\": \"" << jsonEscape(code) << "\",\n"
         << "    \"message\": \"" << jsonEscape(message) << "\"\n"
         << "  },\n"
         << "  \"exit_code\": " << exitCode << "\n"
         << "}\n";
  return stream.str();
}

}  // namespace terrain

