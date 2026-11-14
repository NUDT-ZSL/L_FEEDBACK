#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "baker/atomic_io.h"

namespace baker {

struct WriteJob {
    std::filesystem::path path;
    std::vector<std::uint8_t> bytes;
    std::string label;
};

class AsyncWriter {
public:
    AsyncWriter(std::size_t capacity, bool enabled, std::uint32_t delay_ms);
    ~AsyncWriter();

    AsyncWriter(const AsyncWriter&) = delete;
    AsyncWriter& operator=(const AsyncWriter&) = delete;

    void start();
    void push(WriteJob job);
    void finish();
    void abort() noexcept;

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }
    [[nodiscard]] std::size_t high_watermark() const noexcept { return high_watermark_; }

private:
    void worker_loop();
    void write_job(WriteJob& job);
    void rethrow_failure();

    std::size_t capacity_;
    bool enabled_;
    std::uint32_t delay_ms_;
    bool started_ = false;
    bool stopping_ = false;
    bool aborted_ = false;
    std::size_t high_watermark_ = 0;
    std::deque<WriteJob> queue_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::thread worker_;
    std::exception_ptr failure_;
};

std::size_t queue_capacity_from_env();

}  // namespace baker
