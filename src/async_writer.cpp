#include "baker/async_writer.h"

#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <utility>

#include "baker/error.h"
#include "baker/interrupt.h"

namespace baker {

std::size_t queue_capacity_from_env() {
    const char* text = std::getenv("BAKER_QUEUE_CAPACITY");
    if (text == nullptr || *text == '\0') {
        return 8;
    }
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value == 0 || value > 4096) {
        return 8;
    }
    return static_cast<std::size_t>(value);
}

AsyncWriter::AsyncWriter(std::size_t capacity, bool enabled, std::uint32_t delay_ms)
    : capacity_(std::max<std::size_t>(capacity, 1)),
      enabled_(enabled),
      delay_ms_(delay_ms) {}

AsyncWriter::~AsyncWriter() {
    abort();
}

void AsyncWriter::start() {
    if (!enabled_ || started_) {
        return;
    }
    started_ = true;
    worker_ = std::thread(&AsyncWriter::worker_loop, this);
}

void AsyncWriter::rethrow_failure() {
    if (failure_) {
        std::exception_ptr failure = failure_;
        failure_ = nullptr;
        std::rethrow_exception(failure);
    }
}

void AsyncWriter::write_job(WriteJob& job) {
    write_atomic_file(job.path, job.bytes, AtomicWriteOptions{delay_ms_});
}

void AsyncWriter::push(WriteJob job) {
    throw_if_interrupted();
    if (!enabled_) {
        high_watermark_ = std::max<std::size_t>(high_watermark_, 1);
        write_job(job);
        return;
    }

    std::unique_lock lock(mutex_);
    not_full_.wait(lock, [this] {
        return aborted_ || stopping_ || failure_ || interrupt_requested() ||
               queue_.size() < capacity_;
    });
    if (failure_) {
        rethrow_failure();
    }
    if (aborted_ || stopping_ || interrupt_requested()) {
        throw Interrupted(interrupt_signal() != 0 ? interrupt_signal() : SIGTERM,
                          "writer queue stopped");
    }

    queue_.push_back(std::move(job));
    high_watermark_ = std::max(high_watermark_, queue_.size());
    lock.unlock();
    not_empty_.notify_one();
}

void AsyncWriter::worker_loop() {
    for (;;) {
        WriteJob job;
        {
            std::unique_lock lock(mutex_);
            not_empty_.wait(lock, [this] {
                return aborted_ || stopping_ || !queue_.empty();
            });
            if (aborted_) {
                queue_.clear();
                not_full_.notify_all();
                return;
            }
            if (queue_.empty()) {
                if (stopping_) {
                    not_full_.notify_all();
                    return;
                }
                continue;
            }
            job = std::move(queue_.front());
            queue_.pop_front();
            not_full_.notify_all();
        }

        try {
            write_job(job);
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                failure_ = std::current_exception();
                stopping_ = true;
                queue_.clear();
            }
            not_full_.notify_all();
            not_empty_.notify_all();
            return;
        }
    }
}

void AsyncWriter::finish() {
    if (!enabled_ || !started_) {
        rethrow_failure();
        return;
    }
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    not_empty_.notify_one();
    not_full_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    started_ = false;
    throw_if_interrupted();
    rethrow_failure();
}

void AsyncWriter::abort() noexcept {
    if (!enabled_ || !started_) {
        return;
    }
    {
        std::lock_guard lock(mutex_);
        aborted_ = true;
        stopping_ = true;
        queue_.clear();
    }
    not_empty_.notify_one();
    not_full_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    started_ = false;
}

}  // namespace baker
