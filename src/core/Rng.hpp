#pragma once

#include <cstdint>

namespace lcr {

// SplitMix64 is explicitly specified here instead of using <random> so a seed
// produces the same stream across MSVC, libstdc++ and libc++.
class DeterministicRng {
 public:
  explicit DeterministicRng(const std::uint64_t seed) : state_(seed + 0x9E3779B97F4A7C15ULL) {}

  std::uint64_t nextU64() {
    std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
  }

  std::uint32_t nextU32() { return static_cast<std::uint32_t>(nextU64() >> 32U); }

  std::uint32_t bounded(const std::uint32_t bound) {
    if (bound == 0U) {
      return 0U;
    }
    // Rejection sampling avoids modulo bias and remains deterministic.
    const std::uint32_t limit = UINT32_MAX - (UINT32_MAX % bound);
    std::uint32_t value = nextU32();
    while (value >= limit) {
      value = nextU32();
    }
    return value % bound;
  }

  float unit() {
    // 24 significant bits are exactly representable in float.
    return static_cast<float>(nextU32() >> 8U) * (1.0F / 16777216.0F);
  }

  float symmetric() { return unit() * 2.0F - 1.0F; }

  float range(const float minValue, const float maxValue) {
    return minValue + (maxValue - minValue) * unit();
  }

 private:
  std::uint64_t state_;
};

}  // namespace lcr

