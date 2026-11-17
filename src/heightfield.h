#pragma once

#include "config.h"
#include "math.h"

#include <cstdint>
#include <vector>

namespace ct {

class HeightField {
public:
    explicit HeightField(const HeightfieldConfig& config);

    int size() const { return size_; }
    double cells() const { return static_cast<double>(size_ - 1); }
    double amplitude() const { return config_.amplitude; }

    double at(int x, int z) const;
    double sampleBilinear(double x, double z) const;
    Vec3 normalAt(double x, double z) const;
    std::uint32_t materialAt(double x, double z) const;

    double minimum() const { return minimum_; }
    double maximum() const { return maximum_; }
    double mean() const { return mean_; }

private:
    double rawHeight(double x, double z) const;

    HeightfieldConfig config_;
    int size_ = 0;
    std::vector<double> heights_;
    double minimum_ = 0.0;
    double maximum_ = 0.0;
    double mean_ = 0.0;
};

}  // namespace ct
