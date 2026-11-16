#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace lcr {

class Fnv1a64 {
 public:
  void writeBytes(const void* data, const std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) {
      value_ ^= bytes[i];
      value_ *= 1099511628211ULL;
    }
  }

  void writeU32(const std::uint32_t value) {
    const std::uint8_t bytes[4] = {
        static_cast<std::uint8_t>(value & 0xFFU),
        static_cast<std::uint8_t>((value >> 8U) & 0xFFU),
        static_cast<std::uint8_t>((value >> 16U) & 0xFFU),
        static_cast<std::uint8_t>((value >> 24U) & 0xFFU),
    };
    writeBytes(bytes, 4);
  }

  std::uint64_t value() const { return value_; }

  static std::uint64_t hashBytes(const void* data, const std::size_t size) {
    Fnv1a64 hasher;
    hasher.writeBytes(data, size);
    return hasher.value();
  }

 private:
  std::uint64_t value_ = 14695981039346656037ULL;
};

}  // namespace lcr

