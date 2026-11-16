#include "asset.hpp"

#include <cmath>

namespace skel {

std::size_t Asset::vertexCount() const {
    std::size_t total = 0;
    for (const Mesh& mesh : meshes) {
        total += mesh.vertices.size();
    }
    return total;
}

std::size_t Asset::triangleCount() const {
    std::size_t total = 0;
    for (const Mesh& mesh : meshes) {
        total += mesh.indices.size() / 3U;
    }
    return total;
}

const char* compositionOrderName(CompositionOrder order) {
    switch (order) {
    case CompositionOrder::TRS: return "TRS";
    case CompositionOrder::TSR: return "TSR";
    case CompositionOrder::RTS: return "RTS";
    case CompositionOrder::RST: return "RST";
    case CompositionOrder::STR: return "STR";
    case CompositionOrder::SRT: return "SRT";
    }
    return "INVALID";
}

bool isFinite(const glm::vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

bool isFinite(const glm::quat& value) {
    return std::isfinite(value.w) && std::isfinite(value.x) &&
           std::isfinite(value.y) && std::isfinite(value.z);
}

bool isFinite(const glm::mat4& value) {
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            if (!std::isfinite(value[column][row])) {
                return false;
            }
        }
    }
    return true;
}

} // namespace skel
