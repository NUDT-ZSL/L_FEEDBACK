#include "sha256.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace rigskin {
namespace {

constexpr std::array<std::uint32_t, 64> kConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

inline std::uint32_t rotateRight(std::uint32_t value, std::uint32_t count) {
    return (value >> count) | (value << (32U - count));
}

inline std::uint32_t readBigEndian(const std::uint8_t* data) {
    return (static_cast<std::uint32_t>(data[0]) << 24U) |
           (static_cast<std::uint32_t>(data[1]) << 16U) |
           (static_cast<std::uint32_t>(data[2]) << 8U) |
           static_cast<std::uint32_t>(data[3]);
}

}  // namespace

Sha256::Sha256()
    : state_{
          0x6a09e667U,
          0xbb67ae85U,
          0x3c6ef372U,
          0xa54ff53aU,
          0x510e527fU,
          0x9b05688cU,
          0x1f83d9abU,
          0x5be0cd19U,
      } {}

void Sha256::transform(const std::uint8_t block[64]) {
    std::array<std::uint32_t, 64> words{};
    for (int index = 0; index < 16; ++index) {
        words[static_cast<std::size_t>(index)] = readBigEndian(block + index * 4);
    }
    for (int index = 16; index < 64; ++index) {
        const std::uint32_t s0 =
            rotateRight(words[static_cast<std::size_t>(index - 15)], 7U) ^
            rotateRight(words[static_cast<std::size_t>(index - 15)], 18U) ^
            (words[static_cast<std::size_t>(index - 15)] >> 3U);
        const std::uint32_t s1 =
            rotateRight(words[static_cast<std::size_t>(index - 2)], 17U) ^
            rotateRight(words[static_cast<std::size_t>(index - 2)], 19U) ^
            (words[static_cast<std::size_t>(index - 2)] >> 10U);
        words[static_cast<std::size_t>(index)] =
            words[static_cast<std::size_t>(index - 16)] + s0 +
            words[static_cast<std::size_t>(index - 7)] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];

    for (int index = 0; index < 64; ++index) {
        const std::uint32_t sum1 =
            rotateRight(e, 6U) ^ rotateRight(e, 11U) ^ rotateRight(e, 25U);
        const std::uint32_t choice = (e & f) ^ ((~e) & g);
        const std::uint32_t temporary1 =
            h + sum1 + choice + kConstants[static_cast<std::size_t>(index)] +
            words[static_cast<std::size_t>(index)];
        const std::uint32_t sum0 =
            rotateRight(a, 2U) ^ rotateRight(a, 13U) ^ rotateRight(a, 22U);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temporary2 = sum0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::update(const void* data, std::size_t size) {
    if (finished_) {
        throw std::logic_error("SHA-256 context is already finished");
    }
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    totalBytes_ += size;

    while (size > 0) {
        const std::size_t available = 64 - bufferSize_;
        const std::size_t take = std::min(size, available);
        std::memcpy(buffer_.data() + bufferSize_, bytes, take);
        bufferSize_ += take;
        bytes += take;
        size -= take;

        if (bufferSize_ == 64) {
            transform(buffer_.data());
            bufferSize_ = 0;
        }
    }
}

void Sha256::update(const std::string& data) {
    update(data.data(), data.size());
}

void Sha256::update(const std::vector<std::uint8_t>& data) {
    update(data.data(), data.size());
}

void Sha256::padAndFinish() {
    const std::uint64_t bitCount = totalBytes_ * 8U;
    buffer_[bufferSize_++] = 0x80U;

    if (bufferSize_ > 56) {
        while (bufferSize_ < 64) {
            buffer_[bufferSize_++] = 0U;
        }
        transform(buffer_.data());
        bufferSize_ = 0;
    }

    while (bufferSize_ < 56) {
        buffer_[bufferSize_++] = 0U;
    }
    for (int index = 7; index >= 0; --index) {
        buffer_[bufferSize_++] =
            static_cast<std::uint8_t>((bitCount >> (index * 8)) & 0xffU);
    }
    transform(buffer_.data());
}

std::array<std::uint8_t, 32> Sha256::finish() {
    if (!finished_) {
        padAndFinish();
        finished_ = true;
    }

    std::array<std::uint8_t, 32> result{};
    for (std::size_t index = 0; index < state_.size(); ++index) {
        const std::uint32_t value = state_[index];
        result[index * 4] = static_cast<std::uint8_t>((value >> 24U) & 0xffU);
        result[index * 4 + 1] = static_cast<std::uint8_t>((value >> 16U) & 0xffU);
        result[index * 4 + 2] = static_cast<std::uint8_t>((value >> 8U) & 0xffU);
        result[index * 4 + 3] = static_cast<std::uint8_t>(value & 0xffU);
    }
    return result;
}

std::array<std::uint8_t, 32> Sha256::digest(const void* data, std::size_t size) {
    Sha256 context;
    context.update(data, size);
    return context.finish();
}

std::string Sha256::hex(const std::array<std::uint8_t, 32>& bytes) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (const std::uint8_t byte : bytes) {
        result.push_back(kHex[(byte >> 4U) & 0x0fU]);
        result.push_back(kHex[byte & 0x0fU]);
    }
    return result;
}

std::string Sha256::digestHex(const std::vector<std::uint8_t>& bytes) {
    return hex(digest(bytes.data(), bytes.size()));
}

}  // namespace rigskin
