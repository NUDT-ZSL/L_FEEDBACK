#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace skinbake {

std::vector<std::uint8_t> read_file_bytes(const std::string& path);
void atomic_write_bytes(const std::string& path,
                        const std::vector<std::uint8_t>& bytes);
void remove_file_if_exists(const std::string& path);

}

