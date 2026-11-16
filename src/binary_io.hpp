#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace skel {

class ByteReader {
public:
    ByteReader(const std::uint8_t* data, std::size_t size)
        : data_(data), size_(size) {}

    std::size_t offset() const { return offset_; }
    std::size_t remaining() const { return size_ - offset_; }
    bool exhausted() const { return offset_ == size_; }

    bool readBytes(void* destination, std::size_t count) {
        if (count > remaining()) {
            return false;
        }
        std::memcpy(destination, data_ + offset_, count);
        offset_ += count;
        return true;
    }

    bool readU32(std::uint32_t& value) {
        std::uint8_t bytes[4]{};
        if (!readBytes(bytes, sizeof(bytes))) {
            return false;
        }
        value = static_cast<std::uint32_t>(bytes[0]) |
                (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                (static_cast<std::uint32_t>(bytes[3]) << 24U);
        return true;
    }

    bool readI32(std::int32_t& value) {
        std::uint32_t raw = 0;
        if (!readU32(raw)) {
            return false;
        }
        value = static_cast<std::int32_t>(raw);
        return true;
    }

    bool readF32(float& value) {
        std::uint32_t raw = 0;
        if (!readU32(raw)) {
            return false;
        }
        value = std::bit_cast<float>(raw);
        return true;
    }

    bool readString32(std::string& value) {
        char bytes[32]{};
        if (!readBytes(bytes, sizeof(bytes))) {
            return false;
        }
        std::size_t length = 0;
        while (length < sizeof(bytes) && bytes[length] != '\0') {
            ++length;
        }
        value.assign(bytes, length);
        return true;
    }

private:
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t offset_ = 0;
};

class ByteWriter {
public:
    void writeBytes(const void* source, std::size_t count) {
        const auto* bytes = static_cast<const std::uint8_t*>(source);
        data_.insert(data_.end(), bytes, bytes + count);
    }

    void writeU32(std::uint32_t value) {
        std::uint8_t bytes[4]{
            static_cast<std::uint8_t>(value & 0xFFU),
            static_cast<std::uint8_t>((value >> 8U) & 0xFFU),
            static_cast<std::uint8_t>((value >> 16U) & 0xFFU),
            static_cast<std::uint8_t>((value >> 24U) & 0xFFU),
        };
        writeBytes(bytes, sizeof(bytes));
    }

    void writeI32(std::int32_t value) {
        writeU32(static_cast<std::uint32_t>(value));
    }

    void writeF32(float value) {
        writeU32(std::bit_cast<std::uint32_t>(value));
    }

    void writeString32(const std::string& value) {
        char bytes[32]{};
        const std::size_t length = value.size() < sizeof(bytes) - 1U
                                       ? value.size()
                                       : sizeof(bytes) - 1U;
        std::memcpy(bytes, value.data(), length);
        writeBytes(bytes, sizeof(bytes));
    }

    std::vector<std::uint8_t>& data() { return data_; }
    const std::vector<std::uint8_t>& data() const { return data_; }

private:
    std::vector<std::uint8_t> data_;
};

} // namespace skel

