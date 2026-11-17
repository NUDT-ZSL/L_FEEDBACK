"""GOP / keyframe analysis on top of probed packet lists."""


def keyframe_times(packets):
    return sorted(p["pts"] for p in packets if p["keyframe"])


def floor_keyframe(keyframes, t, eps=1e-3):
    """Largest keyframe timestamp <= t (with small tolerance), or None."""
    best = None
    for kf in keyframes:
        if kf <= t + eps:
            best = kf
        else:
            break
    return best


def is_keyframe_time(keyframes, t, eps=1e-3):
    return any(abs(kf - t) <= eps for kf in keyframes)
