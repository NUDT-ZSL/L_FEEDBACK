"""Reference-report versus current-render comparison with noise classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import ConfigError
from .report import REPORT_SCHEMA, REPORT_SCHEMA_VERSION


REFERENCE_LAYOUT_KEY = "gbuffer_layout"


def load_reference_report(path: str | Path) -> tuple[dict[str, Any], Path]:
    report_path = Path(path)
    if not report_path.is_file():
        raise ConfigError(f"{report_path}: reference report does not exist")
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{report_path}:{exc.lineno}:{exc.colno}: invalid report JSON: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ConfigError(f"{report_path}: cannot read reference report: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise ConfigError(f"{report_path}: not a {REPORT_SCHEMA} document")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ConfigError(
            f"{report_path}: unsupported schema_version {report.get('schema_version')}"
        )
    return report, report_path.parent


def compare_reports(
    reference: dict[str, Any],
    reference_dir: Path,
    current: dict[str, Any],
    current_dir: Path,
    *,
    threshold: float = 0.01,
) -> dict[str, Any]:
    substantive_reasons: list[dict[str, Any]] = []
    reference_checksums = reference.get("scene", {}).get("checksums", {})
    current_checksums = current.get("scene", {}).get("checksums", {})
    for key in sorted(set(reference_checksums) | set(current_checksums)):
        if reference_checksums.get(key) != current_checksums.get(key):
            substantive_reasons.append(
                {
                    "kind": "checksum_changed",
                    "checksum": key,
                    "reference": reference_checksums.get(key),
                    "current": current_checksums.get(key),
                }
            )
    if _canonical(reference.get(REFERENCE_LAYOUT_KEY)) != _canonical(current.get(REFERENCE_LAYOUT_KEY)):
        substantive_reasons.append(
            {"kind": "gbuffer_layout_changed", "detail": "G-buffer attachment contract differs"}
        )
    ref_rendering = reference.get("rendering", {})
    cur_rendering = current.get("rendering", {})
    for key in ("width", "height", "ssaa", "tone_mapper"):
        if ref_rendering.get(key) != cur_rendering.get(key):
            substantive_reasons.append(
                {
                    "kind": "render_setting_changed",
                    "setting": key,
                    "reference": ref_rendering.get(key),
                    "current": cur_rendering.get(key),
                }
            )

    artifact_diffs: dict[str, Any] = {}
    reference_artifacts = reference.get("artifacts", {})
    current_artifacts = current.get("artifacts", {})
    common = sorted(set(reference_artifacts) & set(current_artifacts))
    missing_reference = sorted(set(current_artifacts) - set(reference_artifacts))
    missing_current = sorted(set(reference_artifacts) - set(current_artifacts))
    significant_artifacts = []
    compared = 0
    for name in common:
        ref_meta = reference_artifacts[name]
        cur_meta = current_artifacts[name]
        ref_path = reference_dir / str(ref_meta.get("file", ""))
        cur_path = current_dir / str(cur_meta.get("file", ""))
        if not ref_path.is_file() or not cur_path.is_file():
            artifact_diffs[name] = {
                "status": "missing_image",
                "reference_file_exists": ref_path.is_file(),
                "current_file_exists": cur_path.is_file(),
            }
            continue
        try:
            ref_image = _load_rgb(ref_path)
            cur_image = _load_rgb(cur_path)
        except (OSError, UnidentifiedImageError) as exc:
            artifact_diffs[name] = {"status": "unreadable_image", "error": str(exc)}
            continue
        if ref_image.shape != cur_image.shape:
            substantive_reasons.append(
                {
                    "kind": "artifact_dimensions_changed",
                    "artifact": name,
                    "reference_shape": list(ref_image.shape),
                    "current_shape": list(cur_image.shape),
                }
            )
            artifact_diffs[name] = {"status": "dimension_mismatch"}
            continue
        diff = np.abs(ref_image.astype(np.float64) - cur_image.astype(np.float64)) / 255.0
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        fraction = float(np.mean(np.max(diff, axis=2) > threshold))
        significant = max_diff > threshold and fraction > 0.0005
        if significant:
            significant_artifacts.append(name)
        artifact_diffs[name] = {
            "status": "significant_difference" if significant else "within_tolerance",
            "mean_abs": mean_diff,
            "max_abs": max_diff,
            "fraction_pixels_over_threshold": fraction,
            "reference_sha256": ref_meta.get("sha256"),
            "current_sha256": cur_meta.get("sha256"),
            "hash_changed": ref_meta.get("sha256") != cur_meta.get("sha256"),
        }
        compared += 1

    if significant_artifacts:
        substantive_reasons.append(
            {
                "kind": "pixel_difference_exceeds_threshold",
                "artifacts": significant_artifacts,
                "threshold": threshold,
                "min_pixel_fraction": 0.0005,
            }
        )
    if missing_reference or missing_current:
        substantive_reasons.append(
            {
                "kind": "artifact_set_changed",
                "missing_from_reference": missing_reference,
                "missing_from_current": missing_current,
            }
        )

    max_artifact_diff = max(
        (item["max_abs"] for item in artifact_diffs.values() if "max_abs" in item),
        default=0.0,
    )
    mean_artifact_diff = max(
        (item["mean_abs"] for item in artifact_diffs.values() if "mean_abs" in item),
        default=0.0,
    )
    changed_hashes = [
        name
        for name, item in artifact_diffs.items()
        if item.get("hash_changed") and item.get("status") == "within_tolerance"
    ]
    status = "substantive_difference" if substantive_reasons else "pass"
    classification = "substantive" if substantive_reasons else (
        "floating_point_noise" if changed_hashes else "identical"
    )
    return {
        "status": status,
        "classification": classification,
        "threshold": threshold,
        "reference": {
            "checksums": reference_checksums,
            "artifacts": len(reference_artifacts),
        },
        "current": {
            "checksums": current_checksums,
            "artifacts": len(current_artifacts),
        },
        "comparison": {
            "compared_artifacts": compared,
            "max_abs_pixel_difference": max_artifact_diff,
            "max_mean_abs_pixel_difference": mean_artifact_diff,
            "hashes_changed_within_tolerance": changed_hashes,
            "artifacts": artifact_diffs,
        },
        "substantive_reasons": substantive_reasons,
    }


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
