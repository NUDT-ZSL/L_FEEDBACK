#include "ppm.hpp"

#include <atomic>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PPM
#include <stb/stb_image.h>

namespace skin {
namespace {

std::atomic<unsigned long long> gTempCounter{0};

std::string temporaryName(const std::filesystem::path& finalPath) {
    std::ostringstream name;
    name << finalPath.filename().string()
         << ".tmp-"
         << gTempCounter.fetch_add(1, std::memory_order_relaxed);
    return name.str();
}

}  // namespace

bool writePpmAtomic(
    const std::filesystem::path& path,
    const RgbImage& image,
    std::string& error
) {
    if (!image.valid()) {
        error = "invalid RGB8 image buffer";
        return false;
    }
    if (std::filesystem::exists(path)) {
        error = "refusing to overwrite existing output: " + path.string();
        return false;
    }

    const std::filesystem::path temporary =
        path.parent_path() / temporaryName(path);
    std::ofstream output(temporary, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        error = "cannot open temporary output: " + temporary.string();
        return false;
    }

    output << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    output.write(
        reinterpret_cast<const char*>(image.pixels.data()),
        static_cast<std::streamsize>(image.pixels.size())
    );
    output.flush();
    const bool wroteAll = output.good();
    output.close();

    if (!wroteAll) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        error = "failed while writing PPM data: " + path.string();
        return false;
    }

    std::error_code renameError;
    std::filesystem::rename(temporary, path, renameError);
    if (renameError) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        error = "cannot finalize PPM output: " + renameError.message();
        return false;
    }
    return true;
}

RgbImage loadImageForComparison(
    const std::filesystem::path& path,
    std::string& error
) {
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* data = stbi_load(
        path.string().c_str(),
        &width,
        &height,
        &channels,
        3
    );
    if (data == nullptr) {
        const char* reason = stbi_failure_reason();
        error = reason != nullptr ? reason : "unknown image decoder error";
        return {};
    }

    RgbImage image;
    image.width = width;
    image.height = height;
    if (width <= 0 || height <= 0) {
        stbi_image_free(data);
        error = "baseline has invalid dimensions";
        return {};
    }

    const std::size_t bytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
    image.pixels.assign(data, data + bytes);
    stbi_image_free(data);
    return image;
}

bool compareImages(
    const RgbImage& rendered,
    const RgbImage& baseline,
    PixelDifference& difference,
    std::string& error
) {
    if (!rendered.valid() || !baseline.valid()) {
        error = "one of the comparison images is invalid";
        return false;
    }
    if (rendered.width != baseline.width
        || rendered.height != baseline.height) {
        std::ostringstream message;
        message << "dimension mismatch: rendered="
                << rendered.width << 'x' << rendered.height
                << " baseline=" << baseline.width << 'x'
                << baseline.height;
        error = message.str();
        return false;
    }

    unsigned long long squaredSum = 0;
    unsigned long long absoluteSum = 0;
    int maximum = 0;
    for (std::size_t i = 0; i < rendered.pixels.size(); ++i) {
        const int a = rendered.pixels[i];
        const int b = baseline.pixels[i];
        const int delta = a - b;
        const int absolute = std::abs(delta);
        squaredSum += static_cast<unsigned long long>(delta * delta);
        absoluteSum += static_cast<unsigned long long>(absolute);
        maximum = std::max(maximum, absolute);
    }

    difference.pixelCount =
        static_cast<std::uint64_t>(rendered.width)
        * static_cast<std::uint64_t>(rendered.height);
    difference.maximumAbsoluteError = maximum;
    difference.meanSquaredError =
        static_cast<double>(squaredSum)
        / static_cast<double>(rendered.pixels.size());
    difference.meanAbsoluteError =
        static_cast<double>(absoluteSum)
        / static_cast<double>(rendered.pixels.size());
    return true;
}

OutputPathAllocator::OutputPathAllocator(std::filesystem::path directory)
    : directory_(std::move(directory)) {}

std::filesystem::path OutputPathAllocator::allocate(
    const std::string& groupName
) {
    const std::string stem = sanitizeFileStem(groupName);

    for (int sequence = 0; sequence < 100000; ++sequence) {
        std::ostringstream filename;
        filename << stem;
        if (sequence > 0) {
            filename << '_' << std::setw(3) << std::setfill('0') << sequence;
        }
        filename << ".ppm";

        const std::string key = filename.str();
        const std::filesystem::path candidate = directory_ / key;
        if (claimed_.contains(key)) {
            continue;
        }
        std::error_code existsError;
        const bool exists = std::filesystem::exists(candidate, existsError);
        if (existsError || exists) {
            continue;
        }
        claimed_.insert(key);
        return candidate;
    }

    throw GroupFailure(
        kErrorOutputIo,
        "unable to allocate a unique output filename for group: " + groupName
    );
}

void OutputPathAllocator::release(const std::filesystem::path& path) {
    claimed_.erase(path.filename().string());
}

}  // namespace skin

