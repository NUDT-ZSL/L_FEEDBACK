#pragma once

#include "scene.h"

#include <cstdint>
#include <vector>

namespace df {

class LightSnapshot {
public:
    void sync(const std::vector<Light>& lights);
    void forceDirty();

    uint64_t version() const { return version_; }
    bool dirty() const { return dirty_; }
    void clearDirty() { dirty_ = false; }
    const std::vector<Light>& lights() const { return lights_; }

private:
    std::vector<Light> lights_;
    uint64_t version_ = 0;
    bool dirty_ = true;
};

struct LightUploadStats {
    uint64_t lightingUploads = 0;
    uint64_t fogUploads = 0;
    uint64_t lightingVersion = 0;
    uint64_t fogVersion = 0;
};

void ensureLightingUpload(LightSnapshot& snapshot, LightUploadStats& stats);
void ensureFogUpload(LightSnapshot& snapshot, LightUploadStats& stats);

} // namespace df

