"""End-to-end scene/batch orchestration with strict error containment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import SceneConfig, load_batch, load_scene
from .errors import MixerError
from .generator import generate_assets
from .loudness import normalise_loudness
from .mixer import mix_scene
from .report import (
    build_batch_summary,
    build_failure_report,
    build_report,
    sanitize_json,
    verify_report,
    write_json,
)
from .resampler import engine_description, ffmpeg_version
from .wavio import quantize_pcm, remove_if_exists, write_wav_atomic


def _paths(scene: SceneConfig, base_dir: Path) -> tuple[Path, Path]:
    output = Path(scene.output_path)
    report = Path(scene.report_path)
    if not output.is_absolute():
        output = base_dir / output
    if not report.is_absolute():
        report = base_dir / report
    return output, report


def run_scene(
    scene: SceneConfig,
    base_dir: str | Path,
    *,
    generate: bool | None = None,
) -> dict:
    """Process one scene. Always returns a JSON-ready result payload.

    On failure the target WAV (and any stale WAV at that path) is removed
    before the error report is written, so an incomplete WAV never remains.
    """
    base = Path(base_dir)
    output_path, report_path = _paths(scene, base)
    try:
        should_generate = scene.generate_assets if generate is None else generate
        if should_generate:
            generate_assets(scene, base)

        mixed, mix_report = mix_scene(scene, base)
        normalised, loudness_stats = normalise_loudness(
            mixed,
            scene.project_sample_rate,
            scene.loudness.target_lufs,
            scene.loudness.peak_limit_dbfs,
            scene.loudness.peak_shortfall_tolerance_lu,
            scene.loudness.true_peak,
        )

        # Quantise once, hash the exact PCM-domain samples, then write the WAV
        # and the report that fingerprints it. Both writes are atomic, so any
        # failure above means neither artefact is refreshed.
        pcm_ints = quantize_pcm(normalised, scene.output_sample_width)
        quantised = pcm_ints.astype(np.float64) / (1 << (8 * scene.output_sample_width - 1))
        write_wav_atomic(output_path, quantised, scene.project_sample_rate, scene.output_sample_width)
        environment = {
            "ffmpeg": ffmpeg_version(),
            "resampler_engine": engine_description(),
        }
        report = build_report(
            scene,
            mix_report,
            sanitize_json(loudness_stats),
            quantised,
            output_path,
            extra={"environment": environment},
        )
        write_json(report_path, report)
        return {
            "status": "ok",
            "scene_id": scene.scene_id,
            "report_path": str(report_path),
            "output_path": str(output_path),
            "report_sha256": report["report_sha256"],
        }
    except MixerError as exc:
        remove_if_exists(output_path)
        failure = build_failure_report(scene.scene_id, exc, output_path)
        write_json(report_path, failure)
        return {
            "status": "error",
            "scene_id": scene.scene_id,
            "error_code": exc.code,
            "message": exc.message,
            "details": sanitize_json(exc.details),
            "report_path": str(report_path),
            "output_path": str(output_path),
            "output_wav_present": output_path.exists(),
        }


def run_config_file(
    config_path: str | Path,
    base_dir: str | Path | None = None,
    generate: bool | None = None,
) -> dict:
    config = Path(config_path)
    base = Path(base_dir) if base_dir is not None else config.parent
    scene = load_scene(config)
    return run_scene(scene, base, generate=generate)


def run_batch_file(
    batch_config_path: str | Path,
    summary_path: str | Path | None = None,
) -> tuple[dict, list[dict]]:
    config = Path(batch_config_path)
    scenes = load_batch(config)
    results: list[dict] = []
    for scene in scenes:
        results.append(run_scene(scene, config.parent))
    summary_target = (
        Path(summary_path)
        if summary_path
        else config.parent / "batch_summary.json"
    )
    summary = build_batch_summary(results, summary_target)
    return summary, results


def verify_config_reports(report_paths: list[str]) -> list[dict]:
    return [verify_report(path) for path in report_paths]
