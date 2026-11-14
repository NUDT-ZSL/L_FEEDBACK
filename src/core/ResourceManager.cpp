#include "indoor/ResourceManager.hpp"

#include <algorithm>
#include <chrono>

namespace indoor {

ResourceManager::ResourceManager() : worker_(&ResourceManager::workerLoop, this) {}

ResourceManager::~ResourceManager() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    workCv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

std::uint64_t ResourceManager::submit(ResourceRequest request) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::uint64_t sequence = nextSequence_++;
    request.submittedFrame = currentFrame_;
    requests_.push_back(request);
    workCv_.notify_one();
    return sequence;
}

void ResourceManager::setCurrentFrame(std::uint64_t frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    currentFrame_ = std::max(currentFrame_, frame);
}

std::vector<ResourceCommand> ResourceManager::pollAtFrameBoundary(
    std::uint64_t newFrame) {
    std::lock_guard<std::mutex> lock(mutex_);
    currentFrame_ = std::max(currentFrame_, newFrame);
    std::vector<ResourceCommand> applied;
    std::deque<ResourceCommand> remaining;
    while (!ready_.empty()) {
        ResourceCommand command = ready_.front();
        ready_.pop_front();
        if (command.applyAtFrame <= newFrame) {
            applied.push_back(command);
        } else {
            remaining.push_back(command);
        }
    }
    ready_.swap(remaining);
    return applied;
}

bool ResourceManager::waitForProcessed(std::uint64_t sequence,
                                       int timeoutMilliseconds) {
    std::unique_lock<std::mutex> lock(mutex_);
    return doneCv_.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds),
                            [&] {
                                return processedSequence_ >= sequence;
                            });
}

std::uint64_t ResourceManager::currentFrame() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return currentFrame_;
}

void ResourceManager::workerLoop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        workCv_.wait(lock, [&] {
            return stopping_ || !requests_.empty();
        });
        if (stopping_ && requests_.empty()) {
            break;
        }

        const ResourceRequest request = requests_.front();
        requests_.pop_front();
        ResourceCommand command;
        command.kind = request.kind;
        command.lightId = request.lightId;
        command.submittedFrame = request.submittedFrame;
        command.applyAtFrame = currentFrame_ + 1;

        // Keep sequence assignment independent from queue size in case several
        // requests arrive before the worker wakes.
        command.sequence = processedSequence_ + 1;
        ready_.push_back(command);
        processedSequence_ = command.sequence;
        doneCv_.notify_all();
    }
}

} // namespace indoor
