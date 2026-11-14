"""3D geometry helpers: trajectory sampling, AABB tests, image sources."""

import math

def listener_state_at(keyframes, t):
    """Piecewise-linear listener position/heading at time ``t``."""
    if t <= keyframes[0]["t"]:
        kf = keyframes[0]
        return list(kf["pos"]), kf["heading"]
    if t >= keyframes[-1]["t"]:
        kf = keyframes[-1]
        return list(kf["pos"]), kf["heading"]
    lo = 0
    hi = len(keyframes) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if keyframes[mid]["t"] <= t:
            lo = mid
        else:
            hi = mid
    a = keyframes[lo]
    b = keyframes[hi]
    span = b["t"] - a["t"]
    frac = (t - a["t"]) / span if span > 0 else 0.0
    pos = [a["pos"][i] + frac * (b["pos"][i] - a["pos"][i]) for i in range(3)]
    heading = a["heading"] + frac * (b["heading"] - a["heading"])
    return pos, heading


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def point_in_aabb(p, omin, omax):
    return all(omin[i] < p[i] < omax[i] for i in range(3))


def segment_intersects_aabb(a, b, omin, omax):
    """Slab test for segment a->b against an axis-aligned box."""
    t_min = 0.0
    t_max = 1.0
    for i in range(3):
        d = b[i] - a[i]
        if abs(d) < 1e-12:
            if a[i] < omin[i] or a[i] > omax[i]:
                return False
            continue
        t1 = (omin[i] - a[i]) / d
        t2 = (omax[i] - a[i]) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    return True


def mirror_across_plane(pos, axis, value):
    """Image source position for an axis-aligned reflection plane."""
    img = list(pos)
    img[axis] = 2.0 * value - pos[axis]
    return img
