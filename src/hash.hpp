#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vlr {

std::string sha256Hex(const std::uint8_t* data, std::size_t size);
std::string sha256Hex(const std::vector<std::uint8_t>& data);
std::string sha256Hex(const std::string& data);

} // namespace vlr
