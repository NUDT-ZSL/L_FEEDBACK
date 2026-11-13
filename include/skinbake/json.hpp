#pragma once

#include <string>
#include <utility>
#include <vector>

namespace skinbake {

std::string json_escape(const std::string& value);
std::string json_string(const std::string& value);
std::string json_u64(std::uint64_t value);
std::string json_i64(std::int64_t value);
std::string json_double(double value);
std::string json_bool(bool value);

using JsonField = std::pair<std::string, std::string>;
std::string json_object(const std::vector<JsonField>& fields);

}

