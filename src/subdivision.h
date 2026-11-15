#pragma once

#include "mesh.h"

namespace subdiv {

PolygonMesh subdivideCatmullClarkOnce(const PolygonMesh& input);
PolygonMesh subdivideCatmullClark(const PolygonMesh& base, int level);

}  // namespace subdiv

