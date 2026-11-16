#!/usr/bin/env python3
import json
import math
import sys


def main(path: str) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        root = json.load(handle)
    frames = root["frames"]
    assert len(frames) == 120, len(frames)
    assert root["summary"]["validationPassed"] is True
    assert root["summary"]["defaultQualityRestored"] is True
    assert root["summary"]["longestStableMatrixRun"] >= 40
    assert len(root["adaptiveHistory"]) == 120

    previous = None
    previous_frozen = False
    for frame in frames:
        culling = frame["culling"]
        assert culling["visible"] + culling["occluded"] == culling["frustumVisible"]
        assert len(culling["visibleObjectIds"]) == culling["visible"]
        assert len(culling["occludedObjectIds"]) == culling["occluded"]
        assert culling["occludedObjectIds"] == sorted(culling["occludedObjectIds"])
        local = frame["shadows"]["local"]
        assert len(local) >= 4
        assert sum(1 for light in local if light["updated"]) <= 2
        for light in local:
            assert len(light["matrices"]) == (6 if light["type"] == "point" else 1)
        cascades = frame["shadows"]["directional"]["cascades"]
        assert len(cascades) == 3
        assert sum(c["selectedVisibleObjects"] for c in cascades) == culling["visible"]
        matrices = [value for cascade in cascades for value in cascade["matrix"]]
        assert all(math.isfinite(value) for value in matrices)
        frozen = frame["camera"]["frozenFromPrevious"]
        if frozen and previous_frozen and previous is not None:
            assert matrices == previous
        previous = matrices
        previous_frozen = frozen
    print(json.dumps({
        "frames": len(frames),
        "frameStatsHash": root["summary"]["frameStatsHash"],
        "averageVisible": root["summary"]["averageVisible"],
        "longestStableMatrixRun": root["summary"]["longestStableMatrixRun"],
        "totalLocalShadowUpdates": root["summary"]["totalLocalShadowUpdates"],
        "qualityLevelsSeen": root["summary"]["qualityLevelsSeen"],
        "finalQualityLevel": root["summary"]["finalQualityLevel"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/uid166_benchmark.json"))
