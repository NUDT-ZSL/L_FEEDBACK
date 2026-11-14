#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "baker/error.h"
#include "baker/model.h"

namespace baker {

constexpr std::uint32_t byte_swap32(std::uint32_t value) noexcept {
    return ((value & 0x000000ffu) << 24) | ((value & 0x0000ff00u) << 8) |
           ((value & 0x00ff0000u) >> 8) | ((value & 0xff000000u) >> 24);
}

constexpr std::uint64_t byte_swap64(std::uint64_t value) noexcept {
    return (static_cast<std::uint64_t>(byte_swap32(static_cast<std::uint32_t>(value)))
            << 32) |
           static_cast<std::uint64_t>(byte_swap32(static_cast<std::uint32_t>(value >> 32)));
}

class ByteReader {
public:
    explicit ByteReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

    [[nodiscard]] std::size_t offset() const noexcept { return offset_; }
    [[nodiscard]] std::size_t remaining() const noexcept { return bytes_.size() - offset_; }
    [[nodiscard]] bool at_end() const noexcept { return offset_ == bytes_.size(); }

    void require(std::size_t count, const char* what) const {
        if (count > remaining()) {
            throw BakerError(ErrorCode::kLength,
                             std::string("truncated ") + what + " at byte " +
                                 std::to_string(offset_));
        }
    }

    [[nodiscard]] std::uint32_t u32(const char* what) {
        require(4, what);
        std::uint32_t value = 0;
        std::memcpy(&value, bytes_.data() + offset_, sizeof(value));
        offset_ += 4;
        if constexpr (std::endian::native == std::endian::big) {
            value = byte_swap32(value);
        }
        return value;
    }

    [[nodiscard]] std::uint64_t u64(const char* what) {
        require(8, what);
        std::uint64_t value = 0;
        std::memcpy(&value, bytes_.data() + offset_, sizeof(value));
        offset_ += 8;
        if constexpr (std::endian::native == std::endian::big) {
            value = byte_swap64(value);
        }
        return value;
    }

    [[nodiscard]] float f32(const char* what) {
        const std::uint32_t bits = u32(what);
        const float value = std::bit_cast<float>(bits);
        return value;
    }

    [[nodiscard]] std::span<const std::uint8_t> bytes(std::size_t count, const char* what) {
        require(count, what);
        const auto result = bytes_.subspan(offset_, count);
        offset_ += count;
        return result;
    }

private:
    std::span<const std::uint8_t> bytes_;
    std::size_t offset_ = 0;
};

inline void append_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    if constexpr (std::endian::native == std::endian::big) {
        value = byte_swap32(value);
    }
    const auto* first = reinterpret_cast<const std::uint8_t*>(&value);
    out.insert(out.end(), first, first + sizeof(value));
}

inline void append_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
    if constexpr (std::endian::native == std::endian::big) {
        value = byte_swap64(value);
    }
    const auto* first = reinterpret_cast<const std::uint8_t*>(&value);
    out.insert(out.end(), first, first + sizeof(value));
}

inline void append_f32(std::vector<std::uint8_t>& out, float value) {
    append_u32(out, std::bit_cast<std::uint32_t>(value));
}

}  // namespace baker
