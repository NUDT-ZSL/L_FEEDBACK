#include "baker/atomic_io.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

#include "baker/error.h"
#include "baker/interrupt.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace baker {
namespace {

std::atomic<std::uint64_t> g_temp_counter{0};

std::uint32_t parse_delay_environment() {
    const char* text = std::getenv("BAKER_IO_DELAY_MS");
    if (text == nullptr || *text == '\0') {
        return 0;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > 600000ul) {
        return 0;
    }
    return static_cast<std::uint32_t>(value);
}

void delay_slice(std::uint32_t delay_ms) {
    if (delay_ms == 0) {
        return;
    }
    std::uint32_t remaining = delay_ms;
    while (remaining > 0) {
        throw_if_interrupted();
        const std::uint32_t slice = std::min<std::uint32_t>(remaining, 10);
        std::this_thread::sleep_for(std::chrono::milliseconds(slice));
        remaining -= slice;
    }
    throw_if_interrupted();
}

std::filesystem::path make_temp_path(const std::filesystem::path& target) {
#ifdef _WIN32
    const unsigned long process_id = GetCurrentProcessId();
#else
    const unsigned long process_id = static_cast<unsigned long>(getpid());
#endif
    const std::uint64_t counter =
        g_temp_counter.fetch_add(1, std::memory_order_relaxed);
    return std::filesystem::path(target.string() + ".tmp." +
                                 std::to_string(process_id) + "." +
                                 std::to_string(counter));
}

#ifdef _WIN32

std::string windows_error_text(DWORD code) {
    return "Windows error " + std::to_string(static_cast<unsigned long>(code));
}

void write_atomic_file_platform(const std::filesystem::path& target,
                                std::span<const std::uint8_t> bytes,
                                const AtomicWriteOptions& options) {
    std::filesystem::path temp;
    HANDLE file = INVALID_HANDLE_VALUE;
    for (int attempt = 0; attempt < 16; ++attempt) {
        temp = make_temp_path(target);
        file = CreateFileW(temp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file != INVALID_HANDLE_VALUE) {
            break;
        }
        if (GetLastError() != ERROR_FILE_EXISTS &&
            GetLastError() != ERROR_ALREADY_EXISTS) {
            break;
        }
    }
    if (file == INVALID_HANDLE_VALUE) {
        throw BakerError(ErrorCode::kIo,
                         "cannot create temporary output: " +
                             windows_error_text(GetLastError()));
    }

    bool success = false;
    try {
        delay_slice(options.delay_ms);
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            throw_if_interrupted();
            const std::size_t remaining = bytes.size() - offset;
            const DWORD chunk = static_cast<DWORD>(
                std::min<std::size_t>(remaining, 64u * 1024u));
            DWORD written = 0;
            if (!WriteFile(file, bytes.data() + offset, chunk, &written, nullptr) ||
                written == 0) {
                throw BakerError(ErrorCode::kIo,
                                 "temporary output write failed: " +
                                     windows_error_text(GetLastError()));
            }
            offset += written;
            delay_slice(options.delay_ms);
        }
        if (!FlushFileBuffers(file)) {
            throw BakerError(ErrorCode::kIo,
                             "temporary output flush failed: " +
                                 windows_error_text(GetLastError()));
        }
        CloseHandle(file);
        file = INVALID_HANDLE_VALUE;
        throw_if_interrupted();
        if (!MoveFileExW(temp.c_str(), target.c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            throw BakerError(ErrorCode::kIo,
                             "atomic replace failed: " +
                                 windows_error_text(GetLastError()));
        }
        success = true;
    } catch (...) {
        if (file != INVALID_HANDLE_VALUE) {
            CloseHandle(file);
        }
        if (!success) {
            DeleteFileW(temp.c_str());
        }
        throw;
    }
}

#else

std::string posix_error_text(const char* operation) {
    return std::string(operation) + ": " + std::strerror(errno);
}

void write_atomic_file_platform(const std::filesystem::path& target,
                                std::span<const std::uint8_t> bytes,
                                const AtomicWriteOptions& options) {
    std::filesystem::path temp;
    std::string temp_text;
    int fd = -1;
    for (int attempt = 0; attempt < 16; ++attempt) {
        temp = make_temp_path(target);
        temp_text = temp.string();
        fd = ::open(temp_text.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (fd >= 0) {
            break;
        }
        if (errno != EEXIST) {
            break;
        }
    }
    if (fd < 0) {
        throw BakerError(ErrorCode::kIo, posix_error_text("cannot create temporary output"));
    }

    bool renamed = false;
    try {
        delay_slice(options.delay_ms);
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            throw_if_interrupted();
            const std::size_t remaining = bytes.size() - offset;
            const std::size_t chunk = std::min<std::size_t>(remaining, 64u * 1024u);
            const ssize_t written = ::write(fd, bytes.data() + offset, chunk);
            if (written < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw BakerError(ErrorCode::kIo,
                                 posix_error_text("temporary output write failed"));
            }
            if (written == 0) {
                throw BakerError(ErrorCode::kIo, "temporary output write made no progress");
            }
            offset += static_cast<std::size_t>(written);
            delay_slice(options.delay_ms);
        }
        if (::fsync(fd) != 0) {
            throw BakerError(ErrorCode::kIo,
                             posix_error_text("temporary output flush failed"));
        }
        if (::close(fd) != 0) {
            fd = -1;
            throw BakerError(ErrorCode::kIo,
                             posix_error_text("temporary output close failed"));
        }
        fd = -1;
        throw_if_interrupted();
        if (::rename(temp_text.c_str(), target.string().c_str()) != 0) {
            throw BakerError(ErrorCode::kIo, posix_error_text("atomic replace failed"));
        }
        renamed = true;

        const std::filesystem::path parent =
            target.has_parent_path() ? target.parent_path() : std::filesystem::path(".");
        const int directory_fd =
            ::open(parent.string().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory_fd >= 0) {
            (void)::fsync(directory_fd);
            (void)::close(directory_fd);
        }
    } catch (...) {
        if (fd >= 0) {
            (void)::close(fd);
        }
        if (!renamed) {
            (void)::unlink(temp_text.c_str());
        }
        throw;
    }
}

#endif

}  // namespace

std::uint32_t io_delay_ms_from_env() {
    return parse_delay_environment();
}

void write_atomic_file(const std::filesystem::path& target,
                       std::span<const std::uint8_t> bytes,
                       const AtomicWriteOptions& options) {
    write_atomic_file_platform(target, bytes, options);
}

}  // namespace baker
