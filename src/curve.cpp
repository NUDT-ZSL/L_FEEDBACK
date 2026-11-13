#include "skinbake/curve.hpp"

#include "skinbake/error.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace skinbake {

namespace {

std::string fmt(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6) << value;
    return out.str();
}

}

void Curve::validate(const CurveContext& context) const {
    auto context_details = [&]() {
        return std::vector<std::pair<std::string, std::string>>{
            {"clip", context.clip},
            {"boneIndex", std::to_string(context.boneIndex)},
            {"channel", std::to_string(context.channel)}};
    };

    if (keys_.empty()) {
        auto details = context_details();
        details.emplace_back("reason", "empty");
        throw_error(ErrorCode::CurveEmpty,
                    "curve contains no keyframes", std::move(details));
    }

    for (std::size_t i = 0; i < keys_.size(); ++i) {
        const Keyframe& key = keys_[i];
        if (i > 0 && key.timeMs <= keys_[i - 1].timeMs) {
            auto details = context_details();
            details.emplace_back("index", std::to_string(i));
            details.emplace_back("previousTimeMs", std::to_string(keys_[i - 1].timeMs));
            details.emplace_back("timeMs", std::to_string(key.timeMs));
            throw_error(
                ErrorCode::CurveTimeNotIncreasing,
                "curve timestamps are not strictly increasing at index " +
                    std::to_string(i) + ": " +
                    std::to_string(keys_[i - 1].timeMs) + " -> " +
                    std::to_string(key.timeMs),
                std::move(details));
        }
        if (!std::isfinite(key.value) || !std::isfinite(key.inTangent.timeMs) ||
            !std::isfinite(key.inTangent.value) ||
            !std::isfinite(key.outTangent.timeMs) ||
            !std::isfinite(key.outTangent.value)) {
            auto details = context_details();
            details.emplace_back("index", std::to_string(i));
            throw_error(ErrorCode::CurveNonFinite,
                        "curve keyframe or tangent is not finite",
                        std::move(details));
        }
    }

    for (std::size_t i = 0; i + 1 < keys_.size(); ++i) {
        const Keyframe& left = keys_[i];
        const Keyframe& right = keys_[i + 1];
        const double dx0 = left.outTangent.timeMs;
        const double dx1 = static_cast<double>(right.timeMs - left.timeMs) -
                           left.outTangent.timeMs - right.inTangent.timeMs;
        const double dx2 = right.inTangent.timeMs;
        if (!(dx0 > 0.0) || !(dx1 > 0.0) || !(dx2 > 0.0)) {
            auto details = context_details();
            details.emplace_back("segment", std::to_string(i));
            details.emplace_back("delta0", fmt(dx0));
            details.emplace_back("delta1", fmt(dx1));
            details.emplace_back("delta2", fmt(dx2));
            throw_error(
                ErrorCode::BezierDenominatorZero,
                "bezier parameterization denominator is zero or non-positive in "
                "segment " + std::to_string(i) +
                    " (dx control deltas: " + fmt(dx0) + ", " + fmt(dx1) +
                    ", " + fmt(dx2) + ")",
                std::move(details));
        }
    }
}

void Curve::rebuild_segments() {
    segments_.clear();
    if (keys_.size() < 2) {
        prepared_ = true;
        return;
    }
    segments_.reserve(keys_.size() - 1);
    for (std::size_t i = 0; i + 1 < keys_.size(); ++i) {
        const Keyframe& left = keys_[i];
        const Keyframe& right = keys_[i + 1];
        Segment segment;
        segment.x0 = static_cast<double>(left.timeMs);
        segment.y0 = left.value;
        segment.x1 = segment.x0 + left.outTangent.timeMs;
        segment.y1 = segment.y0 + left.outTangent.value;
        segment.x3 = static_cast<double>(right.timeMs);
        segment.y3 = right.value;
        segment.x2 = segment.x3 - right.inTangent.timeMs;
        segment.y2 = segment.y3 - right.inTangent.value;
        segments_.push_back(segment);
    }
    prepared_ = true;
}

void Curve::prepare() const {
    if (!prepared_) {
        const_cast<Curve*>(this)->rebuild_segments();
    }
}

double Curve::segment_x(const Segment& s, double u) {
    const double v = 1.0 - u;
    return v * v * v * s.x0 +
           3.0 * v * v * u * s.x1 +
           3.0 * v * u * u * s.x2 +
           u * u * u * s.x3;
}

double Curve::segment_y(const Segment& s, double u) {
    const double v = 1.0 - u;
    return v * v * v * s.y0 +
           3.0 * v * v * u * s.y1 +
           3.0 * v * u * u * s.y2 +
           u * u * u * s.y3;
}

double Curve::evaluate(double timeMs) const {
    prepare();
    if (keys_.empty()) {
        throw_error(ErrorCode::CurveEmpty, "cannot evaluate an empty curve");
    }
    if (timeMs <= static_cast<double>(keys_.front().timeMs)) {
        return keys_.front().value;
    }
    if (timeMs >= static_cast<double>(keys_.back().timeMs)) {
        return keys_.back().value;
    }

    auto it = std::upper_bound(
        keys_.begin(), keys_.end(), timeMs,
        [](double t, const Keyframe& key) {
            return t < static_cast<double>(key.timeMs);
        });
    const std::size_t rightIndex =
        static_cast<std::size_t>(std::distance(keys_.begin(), it));
    if (rightIndex == 0) {
        return keys_.front().value;
    }
    if (rightIndex >= keys_.size()) {
        return keys_.back().value;
    }
    if (timeMs == static_cast<double>(keys_[rightIndex].timeMs)) {
        return keys_[rightIndex].value;
    }
    if (timeMs == static_cast<double>(keys_[rightIndex - 1].timeMs)) {
        return keys_[rightIndex - 1].value;
    }

    const Segment& segment = segments_[rightIndex - 1];
    double lo = 0.0;
    double hi = 1.0;
    for (int iteration = 0; iteration < 80; ++iteration) {
        const double mid = lo + (hi - lo) * 0.5;
        const double x = segment_x(segment, mid);
        if (x < timeMs) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const double u = lo + (hi - lo) * 0.5;
    return segment_y(segment, u);
}

float Curve::evaluate_float(double timeMs) const {
    return static_cast<float>(evaluate(timeMs));
}

}

