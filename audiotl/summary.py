"""Waveform summary: per-window peak and RMS for each channel.

The summary is always computed from the final quantized int16 samples of
the mixed WAV (never from cached or stale intermediate data), so it is
sample-exact consistent with the output regardless of incremental reuse.
The trailing partial window is measured over its actual length.
"""

import numpy as np

SUMMARY_VERSION = 1


def compute_summary(q_int16, rate, channels, window_frames, window_ms):
    total = q_int16.shape[0]
    x = q_int16.astype(np.float64) / 32767.0
    windows = []
    index = 0
    for start in range(0, total, window_frames):
        block = x[start:start + window_frames]
        per_channel = []
        for ch in range(channels):
            col = block[:, ch]
            peak = float(np.max(np.abs(col))) if col.size else 0.0
            rms = float(np.sqrt(np.mean(col * col))) if col.size else 0.0
            per_channel.append({"peak": round(peak, 6), "rms": round(rms, 6)})
        windows.append(
            {
                "index": index,
                "start_frame": start,
                "frames": int(block.shape[0]),
                "channels": per_channel,
            }
        )
        index += 1
    return {
        "version": SUMMARY_VERSION,
        "sample_rate": rate,
        "channels": channels,
        "window_ms": window_ms,
        "window_frames": window_frames,
        "total_frames": int(total),
        "windows": windows,
    }
