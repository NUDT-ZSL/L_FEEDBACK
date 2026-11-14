#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace cloud {

class WorkerPool {
public:
    explicit WorkerPool(std::size_t workerCount);
    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    void submit(std::function<void()> task);
    void waitIdle();
    std::size_t size() const { return workers_.size(); }

private:
    void workerMain();
    void rethrowWorkerException();

    std::mutex mutex_;
    std::condition_variable workAvailable_;
    std::condition_variable idle_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    std::size_t activeTasks_ = 0;
    bool stopping_ = false;
    std::exception_ptr workerException_;
};

}  // namespace cloud

