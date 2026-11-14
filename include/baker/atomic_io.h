#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

namespace baker {

struct AtomicWriteOptions {
    std::uint32_t delay_ms = 0;
};

std::uint32_t io_delay_ms_from_env();
void write_atomic_file(const std::filesystem::path& target,
                       std::span<const std::uint8_t> bytes,
                       const AtomicWriteOptions& options);

}  // namespace baker
