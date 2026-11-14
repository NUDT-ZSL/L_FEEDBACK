"""Mix report assembly, checksums and offline re-verification."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .config import SceneConfig
from .errors import ErrorCode, MixerError
from .loudness import measure_loudness, sample_peak_dbfs, true_peak_dbfs
from .wavio import read_wav


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_fingerprint(samples: np.ndarray, sample_rate: int) -> dict:
    contiguous = np.ascontiguousarray(samples.astype(np.float64))
    return {
        "sample_count": int(contiguous.shape[0]),
        "channels": int(contiguous.shape[1]),
        "sample_rate": int(sample_rate),
        "sample_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "first_8_samples": contiguous[:8].round(10).tolist(),
        "last_8_samples": contiguous[-8:].round(10).tolist(),
    }


def build_report(
    scene: SceneConfig,
    mix_report: dict,
    loudness_stats: dict,
    final_samples: np.ndarray,
    output_path: Path,
    extra: dict | None = None,
) -> dict:
    fingerprint = _audio_fingerprint(final_samples, scene.project_sample_rate)
    report = sanitize_json({
        "schema_version": "1.0",
        "status": "ok",
        "scene_id": scene.scene_id,
        "project_sample_rate": scene.project_sample_rate,
        "project_frames": mix_report["project_frames"],
        "duration_seconds": scene.duration,
        "output": {
            "path": str(output_path),
            "sample_width_bytes": scene.output_sample_width,
            "channels": 2,
            "file_sha256": sha256_file(output_path),
        },
        "audio_fingerprint": fingerprint,
        "loudness": loudness_stats,
        "pre_normalisation": {
            "peak_linear": mix_report["pre_normalisation_peak_linear"],
            "peak_dbfs": mix_report["pre_normalisation_peak_dbfs"],
        },
        "tracks": mix_report["tracks"],
        "events": mix_report["events"],
        "validation": {
            "peak_limit_dbfs": scene.loudness.peak_limit_dbfs,
            "target_lufs": scene.loudness.target_lufs,
            "frame_count_matches_duration": (
                fingerprint["sample_count"] == mix_report["project_frames"]
            ),
        },
    })
    if extra:
        report.update(sanitize_json(extra))
    report["report_sha256"] = _canonical_hash({k: v for k, v in report.items() if k != "report_sha256"})
    return report


def _canonical_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def verify_report(report_path: str | Path) -> dict:
    """Re-open the WAV and recompute every key claim stored in the report."""
    report_path = Path(report_path)
    if not report_path.is_file():
        raise MixerError(
            ErrorCode.MISSING_ASSET, f"report not found: {report_path}", {"path": str(report_path)}
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "ok":
        raise MixerError(
            ErrorCode.INVALID_CONFIG,
            "cannot verify a failure report",
            {"scene_id": report.get("scene_id"), "error_code": report.get("error_code")},
        )

    expected_hash = report.pop("report_sha256", None)
    recomputed_hash = _canonical_hash(report)
    report["report_sha256"] = expected_hash

    output_path = Path(report["output"]["path"])
    samples, sample_rate, sample_width = read_wav(output_path)
    fingerprint = _audio_fingerprint(samples, sample_rate)
    stored_fingerprint = report["audio_fingerprint"]

    checks = {
        "report_sha256_match": expected_hash == recomputed_hash,
        "wav_file_sha256_match": sha256_file(output_path) == report["output"]["file_sha256"],
        "sample_sha256_match": fingerprint["sample_sha256"] == stored_fingerprint["sample_sha256"],
        "sample_rate_match": sample_rate == report["project_sample_rate"],
        "sample_width_match": sample_width == report["output"]["sample_width_bytes"],
        "frame_count_match": samples.shape[0] == report["project_frames"],
        "channels_match": samples.shape[1] == 2,
    }

    measured = measure_loudness(samples, sample_rate)
    use_true_peak = report["loudness"].get("peak_measurement") == "true_peak_4x"
    peak = (
        true_peak_dbfs(samples, sample_rate) if use_true_peak else sample_peak_dbfs(samples)
    )
    checks["loudness_matches_report"] = (
        measured["integrated_lufs"] is not None
        and abs(measured["integrated_lufs"] - report["loudness"]["loudness_after_lufs"]) < 0.05
    )
    checks["peak_within_limit"] = peak <= report["validation"]["peak_limit_dbfs"] + 1e-6
    checks["all_passed"] = all(checks.values())

    return {
        "scene_id": report["scene_id"],
        "report_path": str(report_path),
        "output_path": str(output_path),
        "checks": checks,
        "measured": {
            "integrated_lufs": measured["integrated_lufs"],
            "peak_dbfs": peak,
            "frames": samples.shape[0],
            "sample_rate": sample_rate,
        },
    }


def build_failure_report(scene_id: str, error: MixerError, output_path: Path | None) -> dict:
    payload = error.to_payload()
    payload["scene_id"] = scene_id
    payload["output_path"] = str(output_path) if output_path else None
    payload["output_wav_present"] = bool(output_path and output_path.exists())
    return payload


def build_batch_summary(
    results: list[dict], summary_path: str | Path
) -> dict:
    succeeded = [item for item in results if item["status"] == "ok"]
    failed = [item for item in results if item["status"] != "ok"]
    summary = {
        "schema_version": "1.0",
        "status": "ok" if not failed else "partial_failure",
        "total_scenes": len(results),
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
        "succeeded": [
            {
                "scene_id": item["scene_id"],
                "report_path": item["report_path"],
                "output_path": item["output_path"],
                "report_sha256": item.get("report_sha256"),
            }
            for item in succeeded
        ],
        "failed": [
            {
                "scene_id": item["scene_id"],
                "error_code": item["error_code"],
                "message": item["message"],
                "report_path": item["report_path"],
                "output_wav_present": item.get("output_wav_present", False),
            }
            for item in failed
        ],
    }
    write_json(summary_path, summary)
    return summary


def sanitize_json(value):
    """Convert non-finite floats to ``null`` so the report stays valid JSON."""
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
