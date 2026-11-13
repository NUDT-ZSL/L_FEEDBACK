#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace skinbake {

struct Tangent {
    double timeMs = 0.0;
    double value = 0.0;
};

struct Keyframe {
    std::int64_t timeMs = 0;
    double value = 0.0;
    Tangent inTangent;
    Tangent outTangent;
};

struct CurveContext {
    std::string clip;
    std::size_t boneIndex = 0;
    std::size_t channel = 0;
};

class Curve {
public:
    Curve() = default;
    explicit Curve(std::vector<Keyframe> keys) : keys_(std::move(keys)) {}

    void validate(const CurveContext& context) const;
    void prepare() const;
    double evaluate(double timeMs) const;
    float evaluate_float(double timeMs) const;

    const std::vector<Keyframe>& keys() const noexcept { return keys_; }
    std::vector<Keyframe>& keys() noexcept { return keys_; }
    bool empty() const noexcept { return keys_.empty(); }

private:
    struct Segment {
        double x0 = 0.0;
        double x1 = 0.0;
        double x2 = 0.0;
        double x3 = 0.0;
        double y0 = 0.0;
        double y1 = 0.0;
        double y2 = 0.0;
        double y3 = 0.0;
    };

    void rebuild_segments();
    static double segment_x(const Segment& s, double u);
    static double segment_y(const Segment& s, double u);

    std::vector<Keyframe> keys_;
    std::vector<Segment> segments_;
    mutable bool prepared_ = false;
};

}

