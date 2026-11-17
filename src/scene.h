#pragma once

#include "animation.h"
#include "types.h"

namespace sr {

class DeterministicScene {
public:
    explicit DeterministicScene(int initialVariant = 0);

    void rebuildCharacter(int variant);
    void rebuildCharacter(const MeshParams& params);

    const MeshData& characterBindMesh() const noexcept { return characterBind_; }
    const MeshData& occluderMesh() const noexcept { return occluders_; }
    const MeshData& groundMesh() const noexcept { return ground_; }
    const Skeleton& skeleton() const noexcept { return skeleton_; }
    const MeshRevisionInfo& revisionInfo() const noexcept { return revisionInfo_; }

    Bounds characterBounds(const PoseResult& pose) const;
    Bounds visibleCasterBounds(const PoseResult& pose) const;
    MeshData deformedCharacter(const PoseResult& pose) const;

    Lights lights() const;
    Material characterMaterial() const;
    Material occluderMaterial() const;
    Material groundMaterial() const;

private:
    Skeleton skeleton_;
    MeshData characterBind_;
    MeshData occluders_;
    MeshData ground_;
    Bounds occluderBounds_;
    MeshRevisionInfo revisionInfo_;
    int revision_ = 0;
};

} // namespace sr
