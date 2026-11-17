#pragma once

#include "math.hpp"

#include <filesystem>
#include <optional>
#include <string>

namespace terrain {

std::string jsonEscape(const std::string& input);
std::string jsonNumber(float value, int precision = 6);
std::string jsonNumber(double value, int precision = 6);
std::string jsonVec3(const Vec3& value, int precision = 6);
void writeTextFileAtomic(const std::filesystem::path& path, const std::string& text);
std::optional<std::string> readTextFile(const std::filesystem::path& path,
                                        size_t maximumBytes = 2u * 1024u * 1024u);

std::optional<double> jsonReadNumber(const std::string& document, const std::string& key);
std::optional<long long> jsonReadInteger(const std::string& document, const std::string& key);
std::optional<std::string> jsonReadString(const std::string& document, const std::string& key);
std::optional<Vec3> jsonReadVec3(const std::string& document, const std::string& key);

std::string errorJson(const std::string& code, const std::string& message, int exitCode);

}  // namespace terrain

