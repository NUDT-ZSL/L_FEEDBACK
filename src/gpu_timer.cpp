#include "gpu_timer.h"

#include <algorithm>

GpuTimer::~GpuTimer() {
    shutdown();
}

void GpuTimer::begin(int frameIndex) {
    if (queryActive_ || pending_.size() > 48) {
        return;
    }
    glGenQueries(1, &activeQuery_);
    glBeginQuery(GL_TIME_ELAPSED, activeQuery_);
    queryActive_ = true;
    (void)frameIndex;
}

void GpuTimer::end(int frameIndex) {
    if (!queryActive_) {
        return;
    }
    glEndQuery(GL_TIME_ELAPSED);
    pending_.push_back({activeQuery_, frameIndex});
    activeQuery_ = 0;
    queryActive_ = false;
}

void GpuTimer::poll(int currentFrame) {
    while (!pending_.empty() && currentFrame - pending_.front().frameIndex >= 2) {
        GLuint available = GL_FALSE;
        glGetQueryObjectuiv(pending_.front().id, GL_QUERY_RESULT_AVAILABLE,
                            &available);
        if (available == GL_FALSE) {
            break;
        }
        GLuint64 nanoseconds = 0;
        glGetQueryObjectui64v(pending_.front().id, GL_QUERY_RESULT,
                              &nanoseconds);
        const double milliseconds = static_cast<double>(nanoseconds) / 1.0e6;
        recent_.push_back(milliseconds);
        while (recent_.size() > 60) {
            recent_.pop_front();
        }
        lastMilliseconds_ = milliseconds;
        ++samplesCollected_;
        glDeleteQueries(1, &pending_.front().id);
        pending_.pop_front();
    }
}

void GpuTimer::finishAndCollect() {
    if (queryActive_) {
        glEndQuery(GL_TIME_ELAPSED);
        pending_.push_back({activeQuery_, 0});
        activeQuery_ = 0;
        queryActive_ = false;
    }
    if (pending_.empty()) {
        return;
    }
    glFinish();
    while (!pending_.empty()) {
        GLuint64 nanoseconds = 0;
        glGetQueryObjectui64v(pending_.front().id, GL_QUERY_RESULT,
                              &nanoseconds);
        const double milliseconds = static_cast<double>(nanoseconds) / 1.0e6;
        recent_.push_back(milliseconds);
        lastMilliseconds_ = milliseconds;
        ++samplesCollected_;
        glDeleteQueries(1, &pending_.front().id);
        pending_.pop_front();
    }
}

void GpuTimer::shutdown() {
    if (queryActive_) {
        glEndQuery(GL_TIME_ELAPSED);
        queryActive_ = false;
    }
    if (activeQuery_ != 0) {
        glDeleteQueries(1, &activeQuery_);
        activeQuery_ = 0;
    }
    for (PendingQuery& query : pending_) {
        glDeleteQueries(1, &query.id);
    }
    pending_.clear();
}

double GpuTimer::averageMilliseconds() const {
    if (recent_.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double value : recent_) {
        sum += value;
    }
    return sum / static_cast<double>(recent_.size());
}

