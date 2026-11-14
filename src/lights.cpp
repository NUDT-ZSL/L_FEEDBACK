#include "lights.h"

#include <algorithm>

namespace df {
namespace {

bool sameLight(const Light& a, const Light& b) {
    if (a.id != b.id || a.type != b.type) return false;
    if (!approxEqual(a.position, b.position) || !approxEqual(a.color, b.color)) return false;
    if (!approxEqual(a.direction, b.direction)) return false;
    return approxEqual(a.intensity, b.intensity);
}

} // namespace

void LightSnapshot::sync(const std::vector<Light>& lights) {
    bool changed = false;
    if (lights.size() != lights_.size()) {
        changed = true;
    } else {
        for (std::size_t i = 0; i < lights.size(); ++i) {
            if (!sameLight(lights[i], lights_[i])) {
                changed = true;
                break;
            }
        }
    }
    if (!changed) return;
    lights_ = lights;
    ++version_;
    dirty_ = true;
}

void LightSnapshot::forceDirty() {
    if (!lights_.empty()) ++version_;
    dirty_ = true;
}

void ensureLightingUpload(LightSnapshot& snapshot, LightUploadStats& stats) {
    if (stats.lightingVersion == snapshot.version()) return;
    stats.lightingVersion = snapshot.version();
    ++stats.lightingUploads;
}

void ensureFogUpload(LightSnapshot& snapshot, LightUploadStats& stats) {
    if (stats.fogVersion == snapshot.version()) return;
    stats.fogVersion = snapshot.version();
    ++stats.fogUploads;
}

} // namespace df

