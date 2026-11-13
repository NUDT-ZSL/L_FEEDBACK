#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace lfb {

class Sha256 {
public:
    Sha256();

    void update(const void* data, std::size_t size);
    void update(std::string_view data);
    std::string hexDigest();

    static std::string hash(std::string_view data);
    static std::string hash(const void* data, std::size_t size);

private:
    void transform(const std::uint8_t block[64]);
    void finalize();

    std::uint32_t state_[8];
    std::uint8_t buffer_[64];
    std::size_t bufferSize_ = 0;
    std::uint64_t totalBytes_ = 0;
    bool finalized_ = false;
    std::uint8_t digest_[32]{};
};

}  // namespace lfb
