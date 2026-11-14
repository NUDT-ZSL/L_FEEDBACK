#pragma once

#include <glad/gl.h>

#include <cstdint>
#include <deque>
#include <vector>

class GpuTimer {
public:
    GpuTimer() = default;
    ~GpuTimer();

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void begin(int frameIndex);
    void end(int frameIndex);
    void poll(int currentFrame);
    void finishAndCollect();
    void shutdown();

    double lastMilliseconds() const { return lastMilliseconds_; }
    double averageMilliseconds() const;
    int samplesCollected() const { return samplesCollected_; }
    bool hasResult() const { return samplesCollected_ > 0; }

private:
    struct PendingQuery {
        GLuint id = 0;
        int frameIndex = 0;
    };

    std::deque<PendingQuery> pending_;
    std::deque<double> recent_;
    GLuint activeQuery_ = 0;
    bool queryActive_ = false;
    double lastMilliseconds_ = 0.0;
    int samplesCollected_ = 0;
};

