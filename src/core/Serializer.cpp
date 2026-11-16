#include "core/Serializer.hpp"

#include <cstring>

#include "core/Hash.hpp"

namespace lcr {

void CanonicalWriter::u8(const std::uint8_t value) { bytes_.push_back(value); }

void CanonicalWriter::u32(const std::uint32_t value) {
  bytes_.push_back(static_cast<std::uint8_t>(value & 0xFFU));
  bytes_.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
  bytes_.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
  bytes_.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
}

void CanonicalWriter::u64(const std::uint64_t value) {
  u32(static_cast<std::uint32_t>(value & 0xFFFFFFFFULL));
  u32(static_cast<std::uint32_t>(value >> 32U));
}

void CanonicalWriter::i32(const std::int32_t value) {
  u32(static_cast<std::uint32_t>(value));
}

void CanonicalWriter::f32(const float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit");
  std::memcpy(&bits, &value, sizeof(bits));
  u32(bits);
}

void CanonicalWriter::string(const std::string& value) {
  u32(static_cast<std::uint32_t>(value.size()));
  bytes_.insert(bytes_.end(), value.begin(), value.end());
}

std::string hexU64(const std::uint64_t value) {
  constexpr char kHex[] = "0123456789abcdef";
  std::uint64_t remaining = value;
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i) {
    out[static_cast<std::size_t>(i)] = kHex[remaining & 0xFU];
    remaining >>= 4U;
  }
  return out;
}

std::uint64_t hashBytes(const std::vector<std::uint8_t>& bytes) {
  return Fnv1a64::hashBytes(bytes.data(), bytes.size());
}

std::uint64_t hashString(const std::string& value) {
  return Fnv1a64::hashBytes(value.data(), value.size());
}

}  // namespace lcr
