#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image_io.hpp"

#include "errors.hpp"
#include "json_util.hpp"

#include <cstdio>
#include <fstream>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace terrain {
namespace {

struct PngMemory {
  std::vector<uint8_t> bytes;
};

void appendPngBytes(void* context, void* data, int size) {
  auto* memory = static_cast<PngMemory*>(context);
  const auto* begin = static_cast<const uint8_t*>(data);
  memory->bytes.insert(memory->bytes.end(), begin, begin + size);
}

long processId() {
#ifdef _WIN32
  return static_cast<long>(_getpid());
#else
  return static_cast<long>(getpid());
#endif
}

void publishFile(const std::filesystem::path& temporary,
                 const std::filesystem::path& target) {
#ifdef _WIN32
  if (MoveFileExW(temporary.c_str(), target.c_str(),
                  MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0) {
    return;
  }
  std::filesystem::remove(temporary);
  throw businessError("IO_ERROR", "Cannot publish PNG file: " + target.string());
#else
  std::error_code error;
  std::filesystem::rename(temporary, target, error);
  if (error) {
    std::filesystem::remove(temporary);
    throw businessError("IO_ERROR", "Cannot publish PNG file: " + target.string());
  }
#endif
}

}  // namespace

bool savePngAtomic(const std::filesystem::path& path, int width, int height,
                   const std::vector<uint8_t>& rgb) {
  if (width <= 0 || height <= 0 ||
      rgb.size() != static_cast<size_t>(width) * static_cast<size_t>(height) * 3u) {
    throw businessError("IMAGE_SIZE", "RGB buffer does not match image dimensions");
  }
  if (!path.parent_path().empty()) {
    std::error_code directoryError;
    std::filesystem::create_directories(path.parent_path(), directoryError);
    if (directoryError) {
      throw businessError("IO_ERROR",
                          "Cannot create PNG output directory: " +
                              path.parent_path().string());
    }
  }

  PngMemory memory;
  memory.bytes.reserve(rgb.size() / 2u);
  stbi_write_png_compression_level = 6;
  const int success =
      stbi_write_png_to_func(&appendPngBytes, &memory, width, height, 3,
                             rgb.data(), width * 3);
  if (success == 0 || memory.bytes.empty()) {
    throw businessError("IMAGE_ENCODE", "STB failed to encode the PNG image");
  }

  std::filesystem::path temporary = path;
  temporary += ".tmp-" + std::to_string(processId());
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      throw businessError("IO_ERROR",
                          "Cannot create temporary PNG: " + temporary.string());
    }
    stream.write(reinterpret_cast<const char*>(memory.bytes.data()),
                 static_cast<std::streamsize>(memory.bytes.size()));
    stream.flush();
    if (!stream) {
      std::filesystem::remove(temporary);
      throw businessError("IO_ERROR", "Cannot write temporary PNG: " + temporary.string());
    }
  }
  publishFile(temporary, path);
  return true;
}

bool loadPngRgb(const std::filesystem::path& path, LoadedImage& image) {
  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_uc* pixels = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
  if (pixels == nullptr || width <= 0 || height <= 0) {
    return false;
  }
  image.width = width;
  image.height = height;
  image.rgb.assign(pixels, pixels + static_cast<size_t>(width) * height * 3u);
  stbi_image_free(pixels);
  return true;
}

}  // namespace terrain
