#pragma once

#include <filesystem>

namespace vr {

void run_export(const std::filesystem::path& input_dir,
                const std::filesystem::path& output_dir,
                bool resume,
                int abort_after_blocks);

}  // namespace vr

