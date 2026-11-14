#include "thread_pool.h"

#include <stdexcept>

namespace cloud {

WorkerPool::WorkerPool(std::size_t workerCount) {
    if (workerCount == 0) {
        workerCount = 1;
    }
    workers_.reserve(workerCount);
    for (std::size_t index = 0; index < workerCount; ++index) {
        workers_.emplace_back([this] { workerMain(); });
    }
}

WorkerPool::~WorkerPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    workAvailable_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void WorkerPool::submit(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            throw std::runtime_error("worker pool is stopping");
        }
        ++activeTasks_;
        tasks_.push(std::move(task));
    }
    workAvailable_.notify_one();
}

void WorkerPool::waitIdle() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_.wait(lock, [this] { return activeTasks_ == 0 && tasks_.empty(); });
    rethrowWorkerException();
}

void WorkerPool::workerMain() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            workAvailable_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
            if (stopping_ && tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
        }

        try {
            task();
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!workerException_) {
                workerException_ = std::current_exception();
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            --activeTasks_;
            if (activeTasks_ == 0 && tasks_.empty()) {
                idle_.notify_all();
            }
        }
    }
}

void WorkerPool::rethrowWorkerException() {
    if (workerException_) {
        std::exception_ptr exception = workerException_;
        workerException_ = nullptr;
        std::rethrow_exception(exception);
    }
}

}  // namespace cloud

