#include "crc32.h"

#include <array>

namespace cloud {
namespace {

constexpr std::array<std::uint32_t, 256> makeTable() {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t i = 0; i < table.size(); ++i) {
        std::uint32_t value = i;
        for (int bit = 0; bit < 8; ++bit) {
            value = (value & 1u) != 0u ? (value >> 1u) ^ 0xedb88320u : value >> 1u;
        }
        table[i] = value;
    }
    return table;
}

constexpr auto kTable = makeTable();

}  // namespace

std::uint32_t crc32(std::span<const std::uint8_t> bytes) {
    std::uint32_t state = 0xffffffffu;
    for (const std::uint8_t byte : bytes) {
        state = kTable[(state ^ byte) & 0xffu] ^ (state >> 8u);
    }
    return state ^ 0xffffffffu;
}

std::uint32_t crc32(const void* data, std::size_t size) {
    return crc32(std::span<const std::uint8_t>(static_cast<const std::uint8_t*>(data), size));
}

}  // namespace cloud

