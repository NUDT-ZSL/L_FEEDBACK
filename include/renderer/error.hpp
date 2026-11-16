#pragma once

#include <stdexcept>
#include <string>

namespace renderer {

class AppError : public std::runtime_error {
public:
    AppError(std::string code, const std::string& message)
        : std::runtime_error(message), code_(std::move(code)) {}

    const std::string& code() const noexcept {
        return code_;
    }

private:
    std::string code_;
};

[[noreturn]] inline void fail(const std::string& code, const std::string& message) {
    throw AppError(code, message);
}

}  // namespace renderer
