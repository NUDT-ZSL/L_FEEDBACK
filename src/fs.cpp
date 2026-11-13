#include "skinbake/fs.hpp"

#include "skinbake/error.hpp"

#include <cerrno>
#include <cstdio>
#include <fstream>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace skinbake {

std::vector<std::uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw_error(ErrorCode::SnapshotIoError,
                    "unable to open snapshot file: " + path,
                    {{"path", path}, {"errno", std::to_string(errno)}});
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        throw_error(ErrorCode::SnapshotIoError,
                    "unable to determine snapshot size: " + path,
                    {{"path", path}});
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        if (!input) {
            throw_error(ErrorCode::SnapshotIoError,
                        "unable to read complete snapshot: " + path,
                        {{"path", path}});
        }
    }
    return bytes;
}

namespace {

void replace_file(const std::string& tempPath, const std::string& path) {
#ifdef _WIN32
    const int tempSize = MultiByteToWideChar(CP_UTF8, 0, tempPath.c_str(), -1,
                                             nullptr, 0);
    const int pathSize = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1,
                                             nullptr, 0);
    std::wstring tempWide(static_cast<std::size_t>(tempSize), L'\0');
    std::wstring pathWide(static_cast<std::size_t>(pathSize), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, tempPath.c_str(), -1, tempWide.data(),
                        tempSize);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, pathWide.data(), pathSize);
    if (!MoveFileExW(tempWide.c_str(), pathWide.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw_error(ErrorCode::SnapshotIoError,
                    "atomic rename failed for " + path,
                    {{"path", path}, {"win32Error", std::to_string(GetLastError())}});
    }
#else
    if (std::rename(tempPath.c_str(), path.c_str()) != 0) {
        const int savedErrno = errno;
        std::remove(tempPath.c_str());
        throw_error(ErrorCode::SnapshotIoError,
                    "atomic rename failed for " + path,
                    {{"path", path}, {"errno", std::to_string(savedErrno)}});
    }
#endif
}

void sync_file(const std::string& path) {
#ifdef _WIN32
    const int size = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wide(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), size);
    HANDLE handle = CreateFileW(wide.c_str(), GENERIC_READ | GENERIC_WRITE,
                                FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        throw_error(ErrorCode::SnapshotIoError, "unable to sync temporary file",
                    {{"path", path}});
    }
    const BOOL ok = FlushFileBuffers(handle);
    CloseHandle(handle);
    if (!ok) {
        throw_error(ErrorCode::SnapshotIoError, "unable to flush temporary file",
                    {{"path", path}});
    }
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw_error(ErrorCode::SnapshotIoError, "unable to sync temporary file",
                    {{"path", path}, {"errno", std::to_string(errno)}});
    }
    if (::fsync(fd) != 0) {
        const int savedErrno = errno;
        ::close(fd);
        throw_error(ErrorCode::SnapshotIoError, "unable to flush temporary file",
                    {{"path", path}, {"errno", std::to_string(savedErrno)}});
    }
    ::close(fd);
#endif
}

}

void atomic_write_bytes(const std::string& path,
                        const std::vector<std::uint8_t>& bytes) {
    const std::string tempPath = path + ".tmp";
    {
        std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw_error(ErrorCode::SnapshotIoError,
                        "unable to open temporary snapshot: " + tempPath,
                        {{"path", tempPath}});
        }
        if (!bytes.empty()) {
            output.write(reinterpret_cast<const char*>(bytes.data()),
                         static_cast<std::streamsize>(bytes.size()));
        }
        output.flush();
        if (!output) {
            output.close();
            std::remove(tempPath.c_str());
            throw_error(ErrorCode::SnapshotIoError,
                        "unable to write temporary snapshot: " + tempPath,
                        {{"path", tempPath}});
        }
    }
    try {
        sync_file(tempPath);
        replace_file(tempPath, path);
    } catch (...) {
        std::remove(tempPath.c_str());
        throw;
    }
}

void remove_file_if_exists(const std::string& path) {
    std::error_code ignored;
    (void)ignored;
    std::remove(path.c_str());
}

}

