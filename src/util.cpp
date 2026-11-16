#include "util.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace mv {
namespace {

constexpr std::array<std::uint32_t, 64> kRound = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

std::uint32_t rotr(std::uint32_t value, int count) {
    return (value >> count) | (value << (32 - count));
}

}  // namespace

Sha256::Sha256()
    : state_{{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu,
              0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u}} {}

void Sha256::transform(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i) {
        w[static_cast<std::size_t>(i)] =
            (static_cast<std::uint32_t>(block[i * 4]) << 24) |
            (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
            (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
            static_cast<std::uint32_t>(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const auto s0 = rotr(w[static_cast<std::size_t>(i - 15)], 7) ^
                        rotr(w[static_cast<std::size_t>(i - 15)], 18) ^
                        (w[static_cast<std::size_t>(i - 15)] >> 3);
        const auto s1 = rotr(w[static_cast<std::size_t>(i - 2)], 17) ^
                        rotr(w[static_cast<std::size_t>(i - 2)], 19) ^
                        (w[static_cast<std::size_t>(i - 2)] >> 10);
        w[static_cast<std::size_t>(i)] = w[static_cast<std::size_t>(i - 16)] + s0 +
                                         w[static_cast<std::size_t>(i - 7)] + s1;
    }

    auto a = state_[0];
    auto b = state_[1];
    auto c = state_[2];
    auto d = state_[3];
    auto e = state_[4];
    auto f = state_[5];
    auto g = state_[6];
    auto h = state_[7];

    for (int i = 0; i < 64; ++i) {
        const auto s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const auto ch = (e & f) ^ ((~e) & g);
        const auto temp1 = h + s1 + ch + kRound[static_cast<std::size_t>(i)] +
                           w[static_cast<std::size_t>(i)];
        const auto s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const auto maj = (a & b) ^ (a & c) ^ (b & c);
        const auto temp2 = s0 + maj;
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

void Sha256::update(const void* data, std::size_t size) {
    if (finalized_) {
        throw std::logic_error("SHA-256 context already finalized");
    }
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    bitCount_ += static_cast<std::uint64_t>(size) * 8u;
    while (size > 0) {
        const auto take = std::min(size, buffer_.size() - bufferSize_);
        std::memcpy(buffer_.data() + bufferSize_, bytes, take);
        bufferSize_ += take;
        bytes += take;
        size -= take;
        if (bufferSize_ == buffer_.size()) {
            transform(buffer_.data());
            bufferSize_ = 0;
        }
    }
}

void Sha256::update(std::string_view text) {
    update(text.data(), text.size());
}

std::string Sha256::hexDigest() {
    if (!finalized_) {
        const auto originalBits = bitCount_;
        const std::uint8_t marker = 0x80;
        update(&marker, 1);
        const std::uint8_t zero = 0;
        while (bufferSize_ != 56) {
            update(&zero, 1);
        }
        std::array<std::uint8_t, 8> length{};
        for (int i = 0; i < 8; ++i) {
            length[static_cast<std::size_t>(7 - i)] =
                static_cast<std::uint8_t>(originalBits >> (i * 8));
        }
        update(length.data(), length.size());
        finalized_ = true;
    }

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const auto word : state_) {
        out << std::setw(8) << word;
    }
    return out.str();
}

std::string Sha256::hex(const std::string& data) {
    Sha256 digest;
    digest.update(data);
    return digest.hexDigest();
}

std::string Sha256::file(const std::filesystem::path& path) {
    return hex(readTextFile(path));
}

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open file for reading: " + path.string());
    }
    std::ostringstream result;
    result << input.rdbuf();
    return result.str();
}

std::vector<std::uint8_t> readBinaryFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open file for reading: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("cannot determine file size: " + path.string());
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
    if (!result.empty()) {
        input.read(reinterpret_cast<char*>(result.data()), size);
    }
    return result;
}

void writeBinaryFile(const std::filesystem::path& path, const void* data, std::size_t size) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open file for writing: " + path.string());
    }
    if (size > 0) {
        output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
    if (!output) {
        throw std::runtime_error("failed while writing file: " + path.string());
    }
}

void writeTextFile(const std::filesystem::path& path, const std::string& text) {
    writeBinaryFile(path, text.data(), text.size());
}

void writeTextFileAtomic(const std::filesystem::path& path, const std::string& text) {
    writeBinaryFileAtomic(path, text.data(), text.size());
}

void writeBinaryFileAtomic(const std::filesystem::path& path, const void* data, std::size_t size) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    auto temporary = path;
    temporary += ".tmp";
    writeBinaryFile(temporary, data, size);
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(temporary, path, ec);
    }
    if (ec) {
        throw std::runtime_error("cannot finalize file: " + path.string() + ": " + ec.message());
    }
}

std::string trim(std::string value) {
    const auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(),
                std::find_if(value.begin(), value.end(), notSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
    return value;
}

std::vector<std::string> split(std::string_view value, char delimiter) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    while (begin <= value.size()) {
        const auto end = value.find(delimiter, begin);
        const auto part = value.substr(begin, end == std::string_view::npos
                                                  ? value.size() - begin
                                                  : end - begin);
        result.emplace_back(part);
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    return result;
}

std::string join(const std::vector<std::string>& values, std::string_view delimiter) {
    std::string result;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            result += delimiter;
        }
        result += values[i];
    }
    return result;
}

std::string sanitizeName(std::string value) {
    for (auto& c : value) {
        const auto ch = static_cast<unsigned char>(c);
        if (!std::isalnum(ch) && c != '-' && c != '_' && c != '.') {
            c = '_';
        }
    }
    if (value.empty() || value == "." || value == "..") {
        value = "unnamed";
    }
    return value;
}

std::string formatFloat(float value) {
    std::ostringstream out;
    out << std::setprecision(9) << value;
    return out.str();
}

std::string sha256OfStrings(const std::vector<std::string>& values) {
    Sha256 digest;
    for (const auto& value : values) {
        const auto length = std::to_string(value.size());
        digest.update(length);
        digest.update(":");
        digest.update(value);
        digest.update(";");
    }
    return digest.hexDigest();
}

std::uint32_t splitmix32(std::uint32_t& state) {
    state += 0x9e3779b9u;
    auto z = state;
    z = (z ^ (z >> 16)) * 0x21f0aaadu;
    z = (z ^ (z >> 15)) * 0x735a2d97u;
    return z ^ (z >> 15);
}

std::array<std::uint8_t, 4> proceduralTexel(const TextureSpec& spec, int x, int y) {
    auto state = spec.seed ^ (static_cast<std::uint32_t>(x) * 0x85ebca6bu) ^
                 (static_cast<std::uint32_t>(y) * 0xc2b2ae35u) ^
                 (static_cast<std::uint32_t>(spec.width) << 16) ^
                 static_cast<std::uint32_t>(spec.height);
    const auto a = splitmix32(state);
    const auto b = splitmix32(state);
    const auto c = splitmix32(state);
    const auto d = splitmix32(state);
    return {{static_cast<std::uint8_t>(a & 0xffu),
             static_cast<std::uint8_t>((b >> 8) & 0xffu),
             static_cast<std::uint8_t>((c >> 16) & 0xffu),
             static_cast<std::uint8_t>((d >> 24) & 0xffu)}};
}

std::vector<std::uint8_t> generateProceduralTexture(const TextureSpec& spec) {
    std::vector<std::uint8_t> result(static_cast<std::size_t>(spec.width) *
                                     static_cast<std::size_t>(spec.height) * 4u);
    for (int y = 0; y < spec.height; ++y) {
        for (int x = 0; x < spec.width; ++x) {
            const auto texel = proceduralTexel(spec, x, y);
            const auto offset =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(spec.width) +
                 static_cast<std::size_t>(x)) *
                4u;
            std::copy(texel.begin(), texel.end(), result.begin() +
                                                   static_cast<std::ptrdiff_t>(offset));
        }
    }
    return result;
}

MaterialError makeError(ExitCode code,
                        std::string errorCode,
                        std::string stage,
                        std::string message) {
    MaterialError error;
    error.exitCode = code;
    error.code = std::move(errorCode);
    error.stage = std::move(stage);
    error.message = std::move(message);
    return error;
}

}  // namespace mv
