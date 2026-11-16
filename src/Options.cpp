#include "Options.h"

#include <cerrno>
#include <cwchar>
#include <cwctype>
#include <optional>

namespace {

std::wstring Trim(std::wstring value) {
    auto isSpace = [](wchar_t ch) { return std::iswspace(static_cast<wint_t>(ch)) != 0; };
    while (!value.empty() && isSpace(value.front())) {
        value.erase(value.begin());
    }
    while (!value.empty() && isSpace(value.back())) {
        value.pop_back();
    }
    return value;
}

long long ParseInteger(const std::wstring& text, const wchar_t* optionName) {
    const std::wstring trimmed = Trim(text);
    if (trimmed.empty()) {
        throw std::invalid_argument(NarrowAscii(optionName) + " requires an integer value");
    }

    errno = 0;
    wchar_t* end = nullptr;
    const long long value = std::wcstoll(trimmed.c_str(), &end, 10);
    if (errno == ERANGE || end == trimmed.c_str() || *end != L'\0') {
        throw std::invalid_argument(NarrowAscii(optionName) + " has an invalid integer value: " +
                                    NarrowAscii(trimmed));
    }
    return value;
}

double ParseDouble(const std::wstring& text, const wchar_t* optionName) {
    const std::wstring trimmed = Trim(text);
    if (trimmed.empty()) {
        throw std::invalid_argument(NarrowAscii(optionName) + " requires a numeric value");
    }

    errno = 0;
    wchar_t* end = nullptr;
    const double value = std::wcstod(trimmed.c_str(), &end);
    if (errno == ERANGE || end == trimmed.c_str() || *end != L'\0' || !std::isfinite(value)) {
        throw std::invalid_argument(NarrowAscii(optionName) +
                                    " has an invalid numeric value: " +
                                    NarrowAscii(trimmed));
    }
    return value;
}

std::array<double, 3> ParseSplits(const std::wstring& text) {
    std::array<double, 3> values{};
    size_t itemCount = 0;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(L',', start);
        const std::wstring item =
            Trim(text.substr(start, comma == std::wstring::npos ? std::wstring::npos : comma - start));
        if (itemCount >= values.size()) {
            throw std::invalid_argument("--cascade-splits must contain exactly 3 comma-separated values");
        }
        if (item.empty()) {
            throw std::invalid_argument("--cascade-splits contains an empty item");
        }
        values[itemCount] = ParseDouble(item, L"--cascade-splits");
        ++itemCount;
        if (comma == std::wstring::npos) {
            break;
        }
        start = comma + 1;
    }

    if (itemCount != 3) {
        throw std::invalid_argument("--cascade-splits must contain exactly 3 comma-separated values");
    }
    for (size_t i = 0; i < values.size(); ++i) {
        if (!(values[i] > 0.0 && values[i] < 1.0)) {
            throw std::invalid_argument("--cascade-splits values must each be inside (0,1)");
        }
        if (i > 0 && !(values[i] > values[i - 1])) {
            throw std::invalid_argument("--cascade-splits values must be strictly increasing");
        }
    }
    return values;
}

struct ParsedArgument {
    std::wstring name;
    std::wstring value;
};

ParsedArgument ReadArgument(int argc, wchar_t** argv, int& index) {
    std::wstring argument = argv[index];
    const size_t equals = argument.find(L'=');
    if (equals != std::wstring::npos) {
        return {argument.substr(0, equals), argument.substr(equals + 1)};
    }
    if (index + 1 >= argc) {
        throw std::invalid_argument(NarrowAscii(argument) + " requires a value");
    }
    ++index;
    return {std::move(argument), argv[index]};
}

}  // namespace

ParseResult ParseCommandLine(int argc, wchar_t** argv) {
    ParseResult result;
    std::optional<long long> lossFrame;

    for (int i = 1; i < argc; ++i) {
        const std::wstring raw = argv[i];
        if (raw == L"--help" || raw == L"-h") {
            result.showHelp = true;
            return result;
        }
        if (raw.rfind(L"--", 0) != 0) {
            throw std::invalid_argument("unexpected positional argument: " + NarrowAscii(raw));
        }

        const ParsedArgument argument = ReadArgument(argc, argv, i);
        if (argument.name == L"--frames") {
            const long long value = ParseInteger(argument.value, L"--frames");
            if (value <= 0) {
                throw std::invalid_argument("--frames must be greater than zero");
            }
            if (value > static_cast<long long>(std::numeric_limits<uint32_t>::max())) {
                throw std::invalid_argument("--frames is too large");
            }
            result.options.frames = static_cast<uint32_t>(value);
        } else if (argument.name == L"--output-dir") {
            if (argument.value.empty()) {
                throw std::invalid_argument("--output-dir must not be empty");
            }
            result.options.outputDirectory = std::filesystem::path(argument.value);
        } else if (argument.name == L"--cascade-splits") {
            result.options.cascadeSplits = ParseSplits(argument.value);
        } else if (argument.name == L"--slope-offset") {
            const double value = ParseDouble(argument.value, L"--slope-offset");
            if (value < 0.0) {
                throw std::invalid_argument("--slope-offset must be non-negative");
            }
            result.options.slopeOffset = value;
        } else if (argument.name == L"--const-offset") {
            const double value = ParseDouble(argument.value, L"--const-offset");
            if (value < 0.0) {
                throw std::invalid_argument("--const-offset must be non-negative");
            }
            result.options.constOffset = value;
        } else if (argument.name == L"--simulate-device-loss-at") {
            lossFrame = ParseInteger(argument.value, L"--simulate-device-loss-at");
            if (lossFrame.value() < 0) {
                throw std::invalid_argument("--simulate-device-loss-at must be non-negative");
            }
        } else {
            throw std::invalid_argument("unknown option: " + NarrowAscii(argument.name));
        }
    }

    if (lossFrame.has_value()) {
        if (lossFrame.value() >= static_cast<long long>(result.options.frames)) {
            throw std::invalid_argument(
                "--simulate-device-loss-at must be less than --frames");
        }
        result.options.hasDeviceLossInjection = true;
        result.options.deviceLossInjectionFrame = static_cast<uint32_t>(lossFrame.value());
    }

    return result;
}

void PrintUsage(std::wostream& output) {
    output
        << L"Usage: deterministic_csm [options]\n"
        << L"  --frames N                     Number of frames (default 60)\n"
        << L"  --output-dir DIR               PNG/JSON output directory (default output)\n"
        << L"  --cascade-splits A,B,C         Three increasing camera-depth ratios (default 0.1,0.3,0.6)\n"
        << L"  --slope-offset X               Shadow slope-scaled depth bias (default 2.0)\n"
        << L"  --const-offset X               Shadow constant normalized depth bias (default 0.0015)\n"
        << L"  --simulate-device-loss-at N    Tear down and rebuild the device after frame N\n"
        << L"  --help                         Show this help\n";
}

