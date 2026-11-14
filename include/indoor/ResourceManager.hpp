#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace indoor {

enum class ResourceRequestKind {
    DeleteLight,
    SwitchScene
};

struct ResourceRequest {
    ResourceRequestKind kind = ResourceRequestKind::DeleteLight;
    std::uint32_t lightId = 0;
    std::uint64_t submittedFrame = 0;
};

struct ResourceCommand {
    std::uint64_t sequence = 0;
    ResourceRequestKind kind = ResourceRequestKind::DeleteLight;
    std::uint32_t lightId = 0;
    std::uint64_t submittedFrame = 0;
    std::uint64_t applyAtFrame = 0;
};

// The worker only moves plain commands. It never calls OpenGL. The render
// thread drains commands only from pollAtFrameBoundary(), which is the CPU-side
// frame fence required by the renderer.
class ResourceManager {
public:
    ResourceManager();
    ~ResourceManager();

    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;

    std::uint64_t submit(ResourceRequest request);
    void setCurrentFrame(std::uint64_t frame);
    std::vector<ResourceCommand> pollAtFrameBoundary(std::uint64_t newFrame);

    bool waitForProcessed(std::uint64_t sequence, int timeoutMilliseconds = 2000);
    [[nodiscard]] std::uint64_t currentFrame() const;

private:
    void workerLoop();

    mutable std::mutex mutex_;
    std::condition_variable workCv_;
    std::condition_variable doneCv_;
    std::deque<ResourceRequest> requests_;
    std::deque<ResourceCommand> ready_;
    std::uint64_t nextSequence_ = 1;
    std::uint64_t processedSequence_ = 0;
    std::uint64_t currentFrame_ = 0;
    bool stopping_ = false;
    std::thread worker_;
};

} // namespace indoor
