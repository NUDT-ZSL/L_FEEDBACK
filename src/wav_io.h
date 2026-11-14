#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace spatial_audio::detail {

bool readMonoWav16(const std::string& path,
                   std::vector<float>& samples,
                   std::string& error);

bool writeMonoWav16(const std::string& path,
                    const std::vector<std::int16_t>& samples,
                    std::string& error);

bool writeStereoWav16(const std::string& path,
                      const std::vector<std::int16_t>& interleavedStereo,
                      std::string& error);

bool inspectStereoWav16(const std::string& path,
                        std::uint64_t expectedFrames,
                        std::string& error);

}  // namespace spatial_audio::detail

