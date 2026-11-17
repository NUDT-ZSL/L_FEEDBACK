#include "sha256.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>

namespace softraster {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
    0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
    0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
    0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

std::uint32_t rotateRight(std::uint32_t value, unsigned bits) {
    return (value >> bits) | (value << (32U - bits));
}

}  // namespace

Sha256::Sha256()
    : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU, 0x510e527fU,
             0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

void Sha256::transform(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i) {
        const std::size_t offset = i * 4;
        words[i] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                   (static_cast<std::uint32_t>(block[offset + 1]) << 16U) |
                   (static_cast<std::uint32_t>(block[offset + 2]) << 8U) |
                   static_cast<std::uint32_t>(block[offset + 3]);
    }
    for (std::size_t i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotateRight(words[i - 15], 7U) ^
                                 rotateRight(words[i - 15], 18U) ^ (words[i - 15] >> 3U);
        const std::uint32_t s1 = rotateRight(words[i - 2], 17U) ^
                                 rotateRight(words[i - 2], 19U) ^ (words[i - 2] >> 10U);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];

    for (std::size_t i = 0; i < 64; ++i) {
        const std::uint32_t sum1 =
            rotateRight(e, 6U) ^ rotateRight(e, 11U) ^ rotateRight(e, 25U);
        const std::uint32_t choose = (e & f) ^ (~e & g);
        const std::uint32_t temp1 = h + sum1 + choose + kRoundConstants[i] + words[i];
        const std::uint32_t sum0 =
            rotateRight(a, 2U) ^ rotateRight(a, 13U) ^ rotateRight(a, 22U);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
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

void Sha256::update(std::span<const std::uint8_t> data) {
    totalBytes_ += static_cast<std::uint64_t>(data.size());
    std::size_t offset = 0;
    while (offset < data.size()) {
        const std::size_t available = 64 - bufferSize_;
        const std::size_t count = std::min(available, data.size() - offset);
        std::copy_n(data.data() + offset, count, buffer_.data() + bufferSize_);
        bufferSize_ += count;
        offset += count;
        if (bufferSize_ == 64) {
            transform(buffer_.data());
            bufferSize_ = 0;
        }
    }
}

void Sha256::update(std::string_view text) {
    update(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
}

std::array<std::uint8_t, 32> Sha256::finish() {
    const std::uint64_t bitLength = totalBytes_ * 8ULL;
    buffer_[bufferSize_++] = 0x80U;
    if (bufferSize_ > 56) {
        std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(bufferSize_), buffer_.end(),
                  0U);
        transform(buffer_.data());
        bufferSize_ = 0;
    }
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(bufferSize_),
              buffer_.begin() + 56, 0U);
    for (std::size_t i = 0; i < 8; ++i) {
        buffer_[56 + i] =
            static_cast<std::uint8_t>((bitLength >> ((7U - static_cast<unsigned>(i)) * 8U)) &
                                      0xffU);
    }
    transform(buffer_.data());

    std::array<std::uint8_t, 32> digest{};
    for (std::size_t i = 0; i < 8; ++i) {
        digest[i * 4] = static_cast<std::uint8_t>((state_[i] >> 24U) & 0xffU);
        digest[i * 4 + 1] = static_cast<std::uint8_t>((state_[i] >> 16U) & 0xffU);
        digest[i * 4 + 2] = static_cast<std::uint8_t>((state_[i] >> 8U) & 0xffU);
        digest[i * 4 + 3] = static_cast<std::uint8_t>(state_[i] & 0xffU);
    }
    return digest;
}

std::string sha256Hex(std::span<const std::uint8_t> data) {
    Sha256 hash;
    hash.update(data);
    const auto digest = hash.finish();
    std::ostringstream out;
    out << std::hex << std::nouppercase << std::setfill('0');
    for (const std::uint8_t byte : digest) {
        out << std::setw(2) << static_cast<unsigned>(byte);
    }
    return out.str();
}

std::string sha256Hex(std::string_view text) { return sha256Hex(std::span<const std::uint8_t>(
    reinterpret_cast<const std::uint8_t*>(text.data()), text.size())); }

}  // namespace softraster
