from __future__ import annotations

import numpy as np

from .errors import ProcessingError
from .ffmpeg_tools import LoudnessAnalysis, analyze_array, run_loudnorm_pass


TARGET_LUFS = -23.0
LOUDNESS_TOLERANCE_LU = 0.2
MAX_TRUE_PEAK_DBTP = -1.0
_PROCESSING_TRUE_PEAK_DBTP = -1.2


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    factor = np.float32(10.0 ** (gain_db / 20.0))
    return np.ascontiguousarray(audio * factor, dtype=np.float32)


def _fit_length(audio: np.ndarray, expected_samples: int) -> np.ndarray:
    if audio.shape[0] == expected_samples:
        return np.ascontiguousarray(audio, dtype=np.float32)
    if audio.shape[0] > expected_samples:
        return np.ascontiguousarray(audio[:expected_samples], dtype=np.float32)
    padding = np.zeros(
        (expected_samples - audio.shape[0], audio.shape[1]), dtype=np.float32
    )
    return np.ascontiguousarray(np.concatenate((audio, padding), axis=0), dtype=np.float32)


def _passes_targets(
    analysis: LoudnessAnalysis,
    target_lufs: float,
    tolerance_lu: float,
    max_true_peak: float,
) -> bool:
    return (
        abs(analysis.integrated_lufs - target_lufs) <= tolerance_lu + 1e-6
        and analysis.true_peak_dbtp <= max_true_peak + 1e-6
    )


def normalize_master(
    audio: np.ndarray,
    sample_rate: int,
    channels: int,
    frame_rate: float,
    *,
    target_lufs: float = TARGET_LUFS,
    tolerance_lu: float = LOUDNESS_TOLERANCE_LU,
    max_true_peak_dbtp: float = MAX_TRUE_PEAK_DBTP,
) -> tuple[np.ndarray, dict]:
    before = analyze_array(audio, sample_rate, channels, frame_rate)
    requested_gain = target_lufs - before.integrated_lufs
    details: dict = {
        "method": "static_gain",
        "target_lufs": target_lufs,
        "tolerance_lu": tolerance_lu,
        "max_true_peak_dbtp": max_true_peak_dbtp,
        "requested_static_gain_db": round(requested_gain, 3),
        "before": before.to_report(),
    }

    if before.true_peak_dbtp <= -100.0 or before.integrated_lufs <= -70.0:
        details.update(
            {
                "method": "silent_master",
                "gain_db": 0.0,
                "effective_gain_db": 0.0,
                "after": before.to_report(),
                "normalization_type": "silent",
            }
        )
        return np.array(audio, dtype=np.float32, copy=True), details

    if before.true_peak_dbtp + requested_gain <= _PROCESSING_TRUE_PEAK_DBTP + 1e-9:
        candidate = apply_gain(audio, requested_gain)
        static_analysis = analyze_array(candidate, sample_rate, channels, frame_rate)
        correction = target_lufs - static_analysis.integrated_lufs
        if (
            abs(correction) > 0.02
            and static_analysis.true_peak_dbtp
            + correction
            <= _PROCESSING_TRUE_PEAK_DBTP + 1e-9
        ):
            candidate = apply_gain(candidate, correction)
            static_analysis = analyze_array(candidate, sample_rate, channels, frame_rate)
        if _passes_targets(
            static_analysis, target_lufs, tolerance_lu, max_true_peak_dbtp
        ):
            details.update(
                {
                    "gain_db": round(requested_gain + correction, 3),
                    "effective_gain_db": round(
                        static_analysis.integrated_lufs - before.integrated_lufs, 3
                    ),
                    "after": static_analysis.to_report(),
                    "normalization_type": "linear",
                }
            )
            return candidate, details

    candidate = np.ascontiguousarray(audio, dtype=np.float32)
    latest_analysis = before
    latest_payload: dict = {}
    for attempt in range(3):
        processing_peak = _PROCESSING_TRUE_PEAK_DBTP - 0.1 * attempt
        _, pass1 = run_loudnorm_pass(
            candidate,
            sample_rate,
            channels,
            target_lufs=target_lufs,
            true_peak_dbtp=processing_peak,
            capture_audio=False,
        )
        normalized, pass2 = run_loudnorm_pass(
            candidate,
            sample_rate,
            channels,
            target_lufs=target_lufs,
            true_peak_dbtp=processing_peak,
            measured=pass1,
            capture_audio=True,
        )
        if normalized is None:
            raise ProcessingError(3050, "FFmpeg loudnorm returned no audio")
        normalized = _fit_length(normalized, candidate.shape[0])
        latest_analysis = analyze_array(normalized, sample_rate, channels, frame_rate)
        latest_payload = pass2
        if _passes_targets(
            latest_analysis, target_lufs, tolerance_lu, max_true_peak_dbtp
        ):
            candidate = normalized
            break

        correction = target_lufs - latest_analysis.integrated_lufs
        if (
            abs(correction) > 0.02
            and latest_analysis.true_peak_dbtp + correction
            <= _PROCESSING_TRUE_PEAK_DBTP + 1e-9
        ):
            corrected = apply_gain(normalized, correction)
            corrected_analysis = analyze_array(
                corrected, sample_rate, channels, frame_rate
            )
            if _passes_targets(
                corrected_analysis, target_lufs, tolerance_lu, max_true_peak_dbtp
            ):
                latest_analysis = corrected_analysis
                normalized = corrected
                candidate = corrected
                break
        candidate = normalized
    else:
        candidate = normalized

    if not _passes_targets(
        latest_analysis, target_lufs, tolerance_lu, max_true_peak_dbtp
    ):
        raise ProcessingError(
            3051,
            "normalized master did not meet -23 LUFS +/-0.2 LU and <= -1 dBTP",
            {
                "integrated_lufs": round(latest_analysis.integrated_lufs, 3),
                "true_peak_dbtp": round(latest_analysis.true_peak_dbtp, 3),
            },
        )

    effective_gain = latest_analysis.integrated_lufs - before.integrated_lufs
    details.update(
        {
            "method": "two_pass_loudnorm",
            "gain_db": round(effective_gain, 3),
            "effective_gain_db": round(effective_gain, 3),
            "after": latest_analysis.to_report(),
            "normalization_type": latest_payload.get(
                "normalization_type", "unknown"
            ),
            "loudnorm_output_i": latest_payload.get("output_i"),
            "loudnorm_output_tp": latest_payload.get("output_tp"),
        }
    )
    return candidate, details


def verification(
    analysis: LoudnessAnalysis,
    *,
    target_lufs: float = TARGET_LUFS,
    tolerance_lu: float = LOUDNESS_TOLERANCE_LU,
    max_true_peak_dbtp: float = MAX_TRUE_PEAK_DBTP,
) -> dict:
    loudness_error = analysis.integrated_lufs - target_lufs
    return {
        "integrated_lufs": round(analysis.integrated_lufs, 3),
        "true_peak_dbtp": round(analysis.true_peak_dbtp, 3),
        "loudness_error_lu": round(loudness_error, 3),
        "integrated_within_tolerance": abs(loudness_error) <= tolerance_lu + 1e-6,
        "true_peak_within_limit": analysis.true_peak_dbtp
        <= max_true_peak_dbtp + 1e-6,
        "passed": abs(loudness_error) <= tolerance_lu + 1e-6
        and analysis.true_peak_dbtp <= max_true_peak_dbtp + 1e-6,
    }

