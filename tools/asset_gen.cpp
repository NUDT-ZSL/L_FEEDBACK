#include "error.hpp"
#include "sample_generator.hpp"

#include <charconv>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace {

void usage() {
    std::cout
        << "Usage: asset_generator --preset NAME --out FILE "
           "[--order 0..5]\n"
        << "       asset_generator --list\n\n"
        << "Composition orders: 0=TRS 1=TSR 2=RTS 3=RST 4=STR 5=SRT\n";
}

bool parseOrder(std::string_view text, skel::CompositionOrder& order) {
    std::uint32_t value = 0;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} ||
        result.ptr != text.data() + text.size() || value > 5U) {
        return false;
    }
    order = static_cast<skel::CompositionOrder>(value);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string preset;
        std::filesystem::path output;
        skel::CompositionOrder order = skel::CompositionOrder::TRS;
        for (int index = 1; index < argc; ++index) {
            const std::string_view option = argv[index];
            if (option == "--list") {
                for (const skel::GeneratorPreset& item :
                     skel::generatorPresets()) {
                    std::cout << item.name << "\t" << item.description
                              << "\n";
                }
                return 0;
            }
            if (option == "--help" || option == "-h") {
                usage();
                return 0;
            }
            if (option == "--preset" && index + 1 < argc) {
                preset = argv[++index];
            } else if (option == "--out" && index + 1 < argc) {
                output = argv[++index];
            } else if (option == "--order" && index + 1 < argc) {
                if (!parseOrder(argv[++index], order)) {
                    std::cerr << "invalid --order, expected 0..5\n";
                    return 2;
                }
            } else {
                std::cerr << "unknown or incomplete option: " << option
                          << "\n";
                usage();
                return 2;
            }
        }
        if (preset.empty() || output.empty()) {
            usage();
            return 2;
        }

        skel::Error error;
        if (!skel::generateAssetFile(preset, order, output, error)) {
            std::cerr << skel::errorCodeName(error.code) << " ("
                      << static_cast<int>(error.code) << "): "
                      << error.message << "\n";
            return skel::processExitCode(error.code);
        }
        std::cout << "generated " << preset << " asset: "
                  << output.string() << "\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "fatal: " << exception.what() << "\n";
        return 1;
    }
}
