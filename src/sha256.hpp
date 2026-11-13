#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rigskin {

class Sha256 {
public:
    Sha256();

    void update(const void* data, std::size_t size);
    void update(const std::string& data);
    void update(const std::vector<std::uint8_t>& data);
    std::array<std::uint8_t, 32> finish();

    static std::array<std::uint8_t, 32> digest(
        const void* data, std::size_t size);
    static std::string hex(const std::array<std::uint8_t, 32>& bytes);
    static std::string digestHex(const std::vector<std::uint8_t>& bytes);

private:
    void transform(const std::uint8_t block[64]);
    void padAndFinish();

    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t bufferSize_ = 0;
    std::uint64_t totalBytes_ = 0;
    bool finished_ = false;
};

}  // namespace rigskin

