#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace cloud {

std::uint32_t crc32(std::span<const std::uint8_t> bytes);
std::uint32_t crc32(const void* data, std::size_t size);

}  // namespace cloud

