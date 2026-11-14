"""Vector Base Amplitude Panning (VBAP) for 2.0 and 5.1 layouts."""

import math

import numpy as np

# Channel orders: 2.0 -> [L, R]; 5.1 -> [L, R, C, LFE, LS, RS]
LAYOUTS = {
    "2.0": {
        "channels": 2,
        # (azimuth_deg, channel_index); azimuth 0 = front, + = right
        "speakers": [(-30.0, 0), (30.0, 1)],
    },
    "5.1": {
        "channels": 6,
        "speakers": [(-30.0, 0), (30.0, 1), (0.0, 2), (-110.0, 4), (110.0, 5)],
    },
}


def num_channels(layout):
    return LAYOUTS[layout]["channels"]


def _speaker_vectors(layout):
    entries = sorted(LAYOUTS[layout]["speakers"], key=lambda s: s[0])
    vecs = []
    for az_deg, ch in entries:
        az = math.radians(az_deg)
        vecs.append((np.array([math.sin(az), math.cos(az)]), ch))
    return vecs


def vbap_gains(azimuth_deg, layout):
    """Equal-power normalized VBAP gains for a source azimuth (degrees)."""
    info = LAYOUTS[layout]
    gains = np.zeros(info["channels"], dtype=np.float64)
    spk = _speaker_vectors(layout)
    n = len(spk)
    az = math.radians(azimuth_deg)
    u = np.array([math.sin(az), math.cos(az)])

    best = None  # (min_gain, gains, chans)
    for i in range(n):
        j = (i + 1) % n
        v1, c1 = spk[i]
        v2, c2 = spk[j]
        mat = np.column_stack([v1, v2])
        det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
        if abs(det) < 1e-9:
            continue
        g = np.linalg.solve(mat, u)
        key = min(g[0], g[1])
        if best is None or key > best[0]:
            best = (key, g, (c1, c2))

    _, g, (c1, c2) = best
    if min(g[0], g[1]) >= -1e-9:
        pair = np.array([max(g[0], 0.0), max(g[1], 0.0)])
    else:
        # Source outside every speaker arc (e.g. rear-center in 2.0):
        # fall back to the mirrored (absolute) best-pair solution.
        pair = np.abs(g)
    norm = math.hypot(pair[0], pair[1])
    if norm < 1e-9:
        # Degenerate: pick the single nearest speaker.
        dots = [float(np.dot(v, u)) for v, _ in spk]
        gains[spk[int(np.argmax(dots))][1]] = 1.0
        return gains
    pair /= norm
    gains[c1] = pair[0]
    gains[c2] = pair[1]
    return gains
