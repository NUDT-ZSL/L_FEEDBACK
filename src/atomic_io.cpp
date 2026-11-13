#include "atomic_io.hpp"

#include <atomic>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <cerrno>
#include <cstdio>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace lfb::io {

namespace {

std::atomic<std::uint64_t> g_tempCounter{0};

std::uint64_t processId() {
#ifdef _WIN32
    return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint64_t>(getpid());
#endif
}

std::filesystem::path makeTempPath(const std::filesystem::path& target) {
    const std::uint64_t sequence = ++g_tempCounter;
    return target.parent_path() /
           (".tmp-" + std::to_string(processId()) + "-" + std::to_string(sequence) + "-" +
            target.filename().string());
}

bool replaceFile(const std::filesystem::path& source,
                 const std::filesystem::path& destination,
                 std::string& error) {
#ifdef _WIN32
    if (MoveFileExW(source.c_str(), destination.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
        std::ostringstream message;
        message << "MoveFileEx failed with error " << GetLastError();
        error = message.str();
        return false;
    }
    return true;
#else
    if (::rename(source.c_str(), destination.c_str()) != 0) {
        error = "rename failed: " + std::string(std::strerror(errno));
        return false;
    }
    return true;
#endif
}

}  // namespace

bool ensureWritableDirectory(const std::filesystem::path& directory, std::string& error) {
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) {
        error = "cannot create directory '" + directory.string() + "': " + ec.message();
        return false;
    }
    if (!std::filesystem::is_directory(directory, ec) || ec) {
        error = "path is not a writable directory: '" + directory.string() + "'";
        return false;
    }

    const std::filesystem::path probe = makeTempPath(directory / "write-probe");
    std::ofstream output(probe, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "cannot create a file in directory '" + directory.string() + "'";
        return false;
    }
    output << "lf-baker-probe";
    output.flush();
    if (!output) {
        error = "cannot write to directory '" + directory.string() + "'";
        output.close();
        std::filesystem::remove(probe, ec);
        return false;
    }
    output.close();
    std::filesystem::remove(probe, ec);
    return true;
}

bool atomicWriteBytes(const std::filesystem::path& path,
                      const std::vector<std::uint8_t>& bytes,
                      std::string& error) {
    std::error_code ec;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) {
            error = "cannot create parent directory: " + ec.message();
            return false;
        }
    }

    const std::filesystem::path temp = makeTempPath(path);
    {
        std::ofstream output(temp, std::ios::binary | std::ios::trunc);
        if (!output) {
            error = "cannot create temporary file '" + temp.string() + "'";
            return false;
        }
        if (!bytes.empty()) {
            output.write(reinterpret_cast<const char*>(bytes.data()),
                         static_cast<std::streamsize>(bytes.size()));
        }
        output.flush();
        if (!output) {
            error = "cannot write temporary file '" + temp.string() + "'";
            output.close();
            std::filesystem::remove(temp, ec);
            return false;
        }
        output.close();
    }

    if (!replaceFile(temp, path, error)) {
        std::filesystem::remove(temp, ec);
        return false;
    }
    return true;
}

bool readBytes(const std::filesystem::path& path,
               std::vector<std::uint8_t>& bytes,
               std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open file '" + path.string() + "'";
        return false;
    }
    bytes.clear();
    bytes.reserve(static_cast<std::size_t>(1u << 16));
    char buffer[65536];
    while (input.read(buffer, sizeof(buffer)) || input.gcount() > 0) {
        const std::streamsize readCount = input.gcount();
        if (readCount > 0) {
            const auto* first = reinterpret_cast<const std::uint8_t*>(buffer);
            bytes.insert(bytes.end(), first, first + readCount);
        }
        if (input.eof()) {
            break;
        }
    }
    if (input.bad()) {
        error = "cannot read file '" + path.string() + "'";
        return false;
    }
    return true;
}

bool readText(const std::filesystem::path& path, std::string& text, std::string& error) {
    std::vector<std::uint8_t> bytes;
    if (!readBytes(path, bytes, error)) {
        return false;
    }
    text.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return true;
}

void cleanupTemporaryFiles(const std::filesystem::path& directory) {
    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec) || ec) {
        return;
    }
    std::filesystem::recursive_directory_iterator iterator(directory, ec);
    const std::filesystem::recursive_directory_iterator end;
    while (!ec && iterator != end) {
        const auto& entry = *iterator;
        if (entry.is_regular_file(ec) && !ec &&
            entry.path().filename().string().rfind(".tmp-", 0) == 0) {
            std::error_code removeError;
            std::filesystem::remove(entry.path(), removeError);
        }
        iterator.increment(ec);
    }
}

}  // namespace lfb::io
