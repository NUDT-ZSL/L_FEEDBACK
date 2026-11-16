#include <glad/glad.h>
#include <GL/osmesa.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "core/Cli.hpp"
#include "core/Error.hpp"
#include "core/Report.hpp"
#include "core/Serializer.hpp"
#include "core/Simulation.hpp"
#include "render/GLRenderer.hpp"

namespace {

using namespace lcr;

struct OffscreenContext {
  OSMesaContext context = nullptr;
  std::vector<std::uint8_t> color;
  int width = 0;
  int height = 0;

  OffscreenContext(const int widthIn, const int heightIn)
      : width(widthIn), height(heightIn) {
    color.assign(static_cast<std::size_t>(width) * height * 4U, 0U);
    const int attributes[] = {
        OSMESA_FORMAT, OSMESA_RGBA,
        OSMESA_DEPTH_BITS, 24,
        OSMESA_STENCIL_BITS, 8,
        OSMESA_PROFILE, OSMESA_CORE_PROFILE,
        OSMESA_CONTEXT_MAJOR_VERSION, 3,
        OSMESA_CONTEXT_MINOR_VERSION, 3,
        0,
    };
    context = OSMesaCreateContextAttribs(attributes, nullptr);
    if (context == nullptr) {
      fail(ErrorCode::GpuInit, "osmesa.context", "failed to create OSMesa 3.3 core context");
    }
    if (OSMesaMakeCurrent(
            context,
            color.data(),
            GL_UNSIGNED_BYTE,
            width,
            height) == GL_FALSE) {
      fail(ErrorCode::GpuInit, "osmesa.makeCurrent", "OSMesaMakeCurrent failed");
    }
    OSMesaPixelStore(OSMESA_Y_UP, 0);
    if (gladLoadGLLoader(reinterpret_cast<GLADloadproc>(OSMesaGetProcAddress)) == 0) {
      fail(ErrorCode::GpuInit, "glad.load", "failed to load OpenGL through OSMesa");
    }
  }

  ~OffscreenContext() {
    if (context != nullptr) {
      OSMesaDestroyContext(context);
    }
  }
};

std::vector<std::uint8_t> readPixels(const int width, const int height) {
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 4U);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
  return pixels;
}

void writePpm(
    const std::string& path,
    const std::vector<std::uint8_t>& rgba,
    const int width,
    const int height) {
  const std::filesystem::path output(path);
  if (output.has_parent_path()) {
    std::error_code error;
    std::filesystem::create_directories(output.parent_path(), error);
    if (error) {
      fail(ErrorCode::Io, "offscreen.image", "cannot create image directory");
    }
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    fail(ErrorCode::Io, "offscreen.image", "cannot open image: " + path);
  }
  out << "P6\n" << width << ' ' << height << "\n255\n";
  for (int y = height - 1; y >= 0; --y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t index =
          (static_cast<std::size_t>(y) * width + x) * 4U;
      out.put(static_cast<char>(rgba[index + 0U]));
      out.put(static_cast<char>(rgba[index + 1U]));
      out.put(static_cast<char>(rgba[index + 2U]));
    }
  }
  if (!out) {
    fail(ErrorCode::Io, "offscreen.image", "failed while writing image: " + path);
  }
}

std::uint64_t changedPixels(
    const std::vector<std::uint8_t>& a,
    const std::vector<std::uint8_t>& b) {
  std::uint64_t changed = 0;
  for (std::size_t i = 0; i + 3U < a.size(); i += 4U) {
    const int difference =
        std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])) +
        std::abs(static_cast<int>(a[i + 1U]) - static_cast<int>(b[i + 1U])) +
        std::abs(static_cast<int>(a[i + 2U]) - static_cast<int>(b[i + 2U]));
    if (difference > 8) {
      ++changed;
    }
  }
  return changed;
}

std::uint64_t nonBackgroundPixels(const std::vector<std::uint8_t>& pixels) {
  std::uint64_t count = 0;
  for (std::size_t i = 0; i + 3U < pixels.size(); i += 4U) {
    const int difference =
        std::abs(static_cast<int>(pixels[i]) - 7) +
        std::abs(static_cast<int>(pixels[i + 1U]) - 8) +
        std::abs(static_cast<int>(pixels[i + 2U]) - 12);
    if (difference > 20) {
      ++count;
    }
  }
  return count;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    CliOptions options = parseCommandLine(argc, argv);
    if (options.help) {
      printUsage(argv[0]);
      return 0;
    }
    SimulationConfig config;
    config.seed = options.seed;
    config.textureSize = options.textureSize;
    config.initialEquipment = options.equipment;
    config.scheduledSwitchTimes = options.switchTimes;
    config.forceTextureFailure = options.fault == FaultKind::Texture;
    Simulation simulation(config);
    if (options.fault == FaultKind::Bone) {
      simulation.assets().parts.front().mesh.vertices.front().joints[0] = kBoneCount;
      simulation.assets().validate();
    }

    OffscreenContext context(options.windowWidth, options.windowHeight);
    GLRenderer renderer;
    renderer.initialize(
        simulation.assets(),
        simulation.equipment(),
        options.fault == FaultKind::Shader,
        options.windowWidth,
        options.windowHeight);

    const double sampleTime = 2.0;
    simulation.advanceTo(sampleTime);
    OrbitCamera camera;
    const Pose poseBefore = simulation.interpolatedPose(sampleTime);
    const std::string equipmentBefore = simulation.equipment().active().id;
    renderer.render(
        poseBefore,
        simulation.equipment().active(),
        simulation.equipment().activeIndex(),
        camera);
    glFinish();
    const std::vector<std::uint8_t> imageBefore =
        readPixels(options.windowWidth, options.windowHeight);

    const SwitchEvent event = simulation.switchEquipmentNow(sampleTime);
    renderer.render(
        simulation.interpolatedPose(sampleTime),
        simulation.equipment().active(),
        simulation.equipment().activeIndex(),
        camera);
    glFinish();
    const std::vector<std::uint8_t> imageAfter =
        readPixels(options.windowWidth, options.windowHeight);
    const GLenum glError = glGetError();
    if (glError != GL_NO_ERROR) {
      fail(
          ErrorCode::GpuInit,
          "offscreen.render",
          "OpenGL error 0x" + std::to_string(static_cast<unsigned int>(glError)));
    }

    const std::filesystem::path reportPath(options.reportPath);
    const std::filesystem::path directory =
        reportPath.has_parent_path() ? reportPath.parent_path() : std::filesystem::path(".");
    const std::string imageBeforePath = (directory / "offscreen-before.ppm").string();
    const std::string imageAfterPath = (directory / "offscreen-after.ppm").string();
    writePpm(imageBeforePath, imageBefore, options.windowWidth, options.windowHeight);
    writePpm(imageAfterPath, imageAfter, options.windowWidth, options.windowHeight);

    const std::uint64_t changed = changedPixels(imageBefore, imageAfter);
    const std::uint64_t nonBackground = nonBackgroundPixels(imageAfter);
    const std::uint64_t totalPixels =
        static_cast<std::uint64_t>(options.windowWidth) * options.windowHeight;
    if (changed == 0U) {
      fail(
          ErrorCode::InternalInvariant,
          "offscreen.validation",
          "equipment switch produced no visible pixel change");
    }
    if (nonBackground == 0U) {
      fail(
          ErrorCode::InternalInvariant,
          "offscreen.validation",
          "offscreen image contains only clear color");
    }
    if (event.maxJointPositionDelta != 0.0F) {
      fail(
          ErrorCode::InternalInvariant,
          "offscreen.validation",
          "equipment switch changed the pose");
    }

    std::ofstream report(options.reportPath, std::ios::binary | std::ios::trunc);
    if (!report) {
      fail(ErrorCode::Io, "offscreen.report", "cannot write offscreen report");
    }
    report << std::setprecision(9)
           << "{\n"
           << "  \"schema\":\"lcr-offscreen-report-v1\",\n"
           << "  \"seed\":" << options.seed << ",\n"
           << "  \"sample_time\":" << sampleTime << ",\n"
           << "  \"gl_version\":\"" << reinterpret_cast<const char*>(glGetString(GL_VERSION))
           << "\",\n"
           << "  \"gl_renderer\":\"" << reinterpret_cast<const char*>(glGetString(GL_RENDERER))
           << "\",\n"
           << "  \"before\":{\"equipment\":\"" << equipmentBefore
           << "\",\"pixels\":" << imageBefore.size()
           << ",\"hash\":\"" << hexU64(lcr::hashBytes(imageBefore))
           << "\",\"image\":\"" << imageBeforePath << "\"},\n"
           << "  \"after\":{\"equipment\":\"" << event.toEquipment
           << "\",\"pixels\":" << imageAfter.size()
           << ",\"hash\":\"" << hexU64(lcr::hashBytes(imageAfter))
           << "\",\"image\":\"" << imageAfterPath << "\"},\n"
           << "  \"changed_pixels\":" << changed << ",\n"
           << "  \"non_background_pixels\":" << nonBackground << ",\n"
           << "  \"total_pixels\":" << totalPixels << ",\n"
           << "  \"switch_max_joint_delta\":" << event.maxJointPositionDelta << ",\n"
           << "  \"shader_program_builds\":" << renderer.pipelineReport().shaderProgramBuilds << ",\n"
           << "  \"vao_builds\":" << renderer.pipelineReport().vaoBuilds << ",\n"
           << "  \"texture_uploads\":" << renderer.pipelineReport().textureUploads << ",\n"
           << "  \"equipment_pipeline_rebuilds\":0\n"
           << "}\n";
    if (!report) {
      fail(ErrorCode::Io, "offscreen.report", "failed while writing offscreen report");
    }
    std::cout << "{\"status\":\"ok\",\"mode\":\"offscreen\","
              << "\"changed_pixels\":" << changed
              << ",\"non_background_pixels\":" << nonBackground
              << ",\"image_before\":\"" << imageBeforePath
              << "\",\"image_after\":\"" << imageAfterPath << "\"}\n";
    return 0;
  } catch (const AppError& error) {
    emitErrorToStderr(error);
    return errorExitCode(error.code());
  } catch (const std::exception& error) {
    emitErrorToStderr(AppError(
        ErrorCode::InternalInvariant,
        "offscreen.unhandled",
        error.what()));
    return errorExitCode(ErrorCode::InternalInvariant);
  }
}
