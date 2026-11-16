#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lcr {

// Canonical encoding: little endian integers and IEEE-754 float bit patterns.
// This makes hashes independent of host endianness and string layout.
class CanonicalWriter {
 public:
  void u8(std::uint8_t value);
  void u32(std::uint32_t value);
  void u64(std::uint64_t value);
  void i32(std::int32_t value);
  void f32(float value);
  void string(const std::string& value);

  const std::vector<std::uint8_t>& bytes() const { return bytes_; }

 private:
  std::vector<std::uint8_t> bytes_;
};

std::string hexU64(std::uint64_t value);
std::uint64_t hashBytes(const std::vector<std::uint8_t>& bytes);
std::uint64_t hashString(const std::string& value);

}  // namespace lcr

