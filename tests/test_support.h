#pragma once

#include <chrono>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>

namespace test {

class Failure : public std::runtime_error {
public:
    explicit Failure(const std::string& message) : std::runtime_error(message) {}
};

inline void require(bool condition, const std::string& message) {
    if (!condition) {
        throw Failure(message);
    }
}

inline void requireNear(double actual, double expected, double tolerance,
                        const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::ostringstream out;
        out << message << " (actual=" << actual << ", expected=" << expected
            << ", tolerance=" << tolerance << ")";
        throw Failure(out.str());
    }
}

inline std::filesystem::path makeTempDirectory(const std::string& name) {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("automix_" + name + "_" + std::to_string(stamp));
    std::filesystem::create_directories(path);
    return path;
}

}  // namespace test
