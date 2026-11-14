#include <cerrno>
#include <cfenv>
#include <clocale>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <locale>
#include <string>

#include "baker/atomic_io.h"
#include "baker/error.h"
#include "baker/interrupt.h"
#include "baker/model.h"
#include "baker/scene_generator.h"

namespace {

void usage(std::ostream& out) {
    out << "usage: scene_gen --out SCENE [--width N] [--height N] [--layers N] "
           "[--tex-size N] [--seed N]\n";
}

[[noreturn]] void argument_error(const std::string& message) {
    throw baker::BakerError(baker::ErrorCode::kArgs, message);
}

std::uint64_t parse_u64(const char* text, const char* option) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        argument_error(std::string("invalid value for ") + option);
    }
    return static_cast<std::uint64_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
    (void)std::setlocale(LC_ALL, "C");
    std::locale::global(std::locale::classic());
    if (std::fesetround(FE_TONEAREST) != 0) {
        std::cerr << "ERR_IO: cannot establish round-to-nearest mode\n";
        return static_cast<int>(baker::ErrorCode::kIo);
    }
    baker::install_signal_handlers();
    try {
        std::filesystem::path output;
        baker::GeneratorOptions options;
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            const auto value_after = [&](const char* name) -> const char* {
                if (index + 1 >= argc) {
                    argument_error(std::string("missing value for ") + name);
                }
                return argv[++index];
            };

            if (option == "--out") {
                output = value_after("--out");
            } else if (option == "--width") {
                options.width = static_cast<std::uint32_t>(
                    parse_u64(value_after("--width"), "--width"));
            } else if (option == "--height") {
                options.height = static_cast<std::uint32_t>(
                    parse_u64(value_after("--height"), "--height"));
            } else if (option == "--layers") {
                options.layers = static_cast<std::uint32_t>(
                    parse_u64(value_after("--layers"), "--layers"));
            } else if (option == "--tex-size") {
                options.tex_size = static_cast<std::uint32_t>(
                    parse_u64(value_after("--tex-size"), "--tex-size"));
            } else if (option == "--seed") {
                options.seed = parse_u64(value_after("--seed"), "--seed");
            } else if (option == "--help" || option == "-h") {
                usage(std::cout);
                return 0;
            } else {
                argument_error("unknown option '" + option + "'");
            }
        }

        if (output.empty()) {
            argument_error("--out is required");
        }
        if (options.width == 0 || options.width > baker::kMaxDimension ||
            options.height == 0 || options.height > baker::kMaxDimension ||
            options.layers == 0 || options.layers > baker::kMaxLayers ||
            options.tex_size == 0 || options.tex_size > baker::kMaxTextureSize) {
            argument_error("generator dimensions are outside the supported range");
        }

        const std::vector<std::uint8_t> bytes =
            baker::generate_scene_bytes(options);
        baker::write_atomic_file(output, bytes, {});
        std::cout << "[scene_gen] output=" << output.string()
                  << " bytes=" << bytes.size() << " seed=" << options.seed << '\n';
        return 0;
    } catch (const baker::Interrupted& error) {
        std::cerr << "ERR_INTERRUPTED: " << error.what() << '\n';
        return 128 + error.signal_number();
    } catch (const baker::BakerError& error) {
        std::cerr << baker::error_name(error.code()) << ": " << error.what() << '\n';
        return static_cast<int>(error.code());
    } catch (const std::exception& error) {
        std::cerr << "ERR_IO: " << error.what() << '\n';
        return static_cast<int>(baker::ErrorCode::kIo);
    }
}
