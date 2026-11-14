#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace vr {

class Sha256 {
public:
    Sha256();

    void update(const void* data, std::size_t size);
    void update(std::string_view text);
    std::string digest_hex() const;
    std::uint64_t byte_count() const noexcept { return total_; }

private:
    void transform(const std::uint8_t* block);

    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> buffer_;
    std::size_t buffered_ = 0;
    std::uint64_t total_ = 0;
};

std::string sha256_bytes(std::string_view bytes);
std::string sha256_file(const std::filesystem::path& path);

}  // namespace vr

