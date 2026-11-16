#include "bmp.hpp"
#include "config.hpp"
#include "error.hpp"
#include "model.hpp"
#include "renderer.hpp"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

std::string printablePath(const std::filesystem::path& path) {
    return path.u8string();
}

int runRenderer(const std::filesystem::path& configPath) {
    const preview::RenderConfig config = preview::loadConfig(configPath);
    const preview::Model model = preview::loadModel(config.objPath);
    const preview::Image image = preview::renderScene(config, model);
    preview::writeBmp24(config.outputPath, image);
    std::cout << "Rendered " << config.width << "x" << config.height
              << " to " << printablePath(config.outputPath) << "\n";
    return 0;
}

} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    if (argc != 2) {
        std::cerr << "E_USAGE: expected exactly one JSON config path\n";
        return preview::exitCodeFor(preview::ErrorCode::Usage);
    }
    try {
        return runRenderer(std::filesystem::path(argv[1]));
    } catch (const preview::AppError& error) {
        std::cerr << preview::errorCodeName(error.code()) << ": " << error.what() << "\n";
        return preview::exitCodeFor(error.code());
    } catch (const std::exception& error) {
        std::cerr << "E_INTERNAL: " << error.what() << "\n";
        return preview::exitCodeFor(preview::ErrorCode::Internal);
    }
}
#else
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "E_USAGE: expected exactly one JSON config path\n";
        return preview::exitCodeFor(preview::ErrorCode::Usage);
    }
    try {
        return runRenderer(std::filesystem::u8path(argv[1]));
    } catch (const preview::AppError& error) {
        std::cerr << preview::errorCodeName(error.code()) << ": " << error.what() << "\n";
        return preview::exitCodeFor(error.code());
    } catch (const std::exception& error) {
        std::cerr << "E_INTERNAL: " << error.what() << "\n";
        return preview::exitCodeFor(preview::ErrorCode::Internal);
    }
}
#endif
