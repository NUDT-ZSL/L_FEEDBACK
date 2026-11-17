#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace softraster {

class Sha256 {
public:
    Sha256();

    void update(std::span<const std::uint8_t> data);
    void update(std::string_view text);
    [[nodiscard]] std::array<std::uint8_t, 32> finish();

private:
    void transform(const std::uint8_t* block);

    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> buffer_{};
    std::uint64_t totalBytes_ = 0;
    std::size_t bufferSize_ = 0;
};

[[nodiscard]] std::string sha256Hex(std::span<const std::uint8_t> data);
[[nodiscard]] std::string sha256Hex(std::string_view text);

}  // namespace softraster
