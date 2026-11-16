"""Per-language processing pipeline and deterministic parallel orchestration."""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import __version__
from .audio import DetectConfig, detect_segments, read_wav, write_wav
from .errors import (
    DubError,
    EXIT_FATAL,
    EXIT_OK,
    EXIT_PARTIAL,
    E_SAMPLE_RATE_MISMATCH,
    E_SEGMENT_EXTRA,
    E_SEGMENT_MISSING,
    E_STRETCH_TOLERANCE,
)
from .mux import extract_audio, mux, probe_duration, probe_stream_durations
from .report import file_digest, write_report
from .stretch import assemble_track
from .subtitles import (
    enforce_constraints,
    load_subtitles,
    retime_entries,
    validate_within_video,
    write_srt,
)


@dataclass(frozen=True)
class JobConfig:
    sample_rate: int = 48000
    tolerance: float = 0.05          # max |final boundary - target boundary| (s)
    max_sub_duration: float = 7.0    # max subtitle display duration (s)
    min_sub_duration: float = 1.0    # min subtitle display duration (s)
    jobs: int = 4
    detect: DetectConfig = field(default_factory=DetectConfig)

    def to_report_dict(self) -> dict:
        # NOTE: `jobs` is deliberately excluded so the report is identical
        # for any concurrency level.
        return {
            "sample_rate": self.sample_rate,
            "tolerance": self.tolerance,
            "max_sub_duration": self.max_sub_duration,
            "min_sub_duration": self.min_sub_duration,
            "detection": self.detect.to_dict(),
        }


def _segment_slices(segments, rate, samples):
    out = []
    for start, end in segments:
        out.append(samples[round(start * rate): round(end * rate)])
    return out


def process_language(
    lang: str,
    wav_path: str,
    subs_path: str,
    ref_segments: list[tuple[float, float]],
    video_duration: float,
    video_path: str,
    out_dir: str,
    tmp_root: str,
    cfg: JobConfig,
) -> dict:
    """Process one language end-to-end. Never raises for business errors."""
    result: dict = {"status": "failed", "errors": []}
    tmp_dir = os.path.join(tmp_root, lang)
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        samples, rate = read_wav(wav_path)
        if rate != cfg.sample_rate:
            raise DubError(
                E_SAMPLE_RATE_MISMATCH,
                f"dub sample rate {rate} != required {cfg.sample_rate}",
            )
        segments, leading = detect_segments(samples, rate, cfg.detect)
        result["detection"] = {
            "threshold": cfg.detect.threshold,
            "leading_silence": round(leading, 6),
            "segments": [
                {
                    "index": i,
                    "start": round(s, 6),
                    "end": round(e, 6),
                    "duration": round(e - s, 6),
                }
                for i, (s, e) in enumerate(segments)
            ],
        }
        if len(segments) < len(ref_segments):
            raise DubError(
                E_SEGMENT_MISSING,
                f"dub has {len(segments)} segments, reference needs {len(ref_segments)}",
            )
        if len(segments) > len(ref_segments):
            raise DubError(
                E_SEGMENT_EXTRA,
                f"dub has {len(segments)} segments, reference has {len(ref_segments)}",
            )

        seg_samples = _segment_slices(segments, rate, samples)
        track, records = assemble_track(seg_samples, ref_segments, video_duration, rate)
        for rec in records:
            if rec["boundary_error"] > cfg.tolerance:
                raise DubError(
                    E_STRETCH_TOLERANCE,
                    f"segment {rec['index']} boundary error "
                    f"{rec['boundary_error']:.6f}s exceeds tolerance {cfg.tolerance}s",
                )
        result["stretch"] = {"tolerance": cfg.tolerance, "segments": records}

        entries = load_subtitles(subs_path)
        validate_within_video(entries, video_duration)
        ratios = [rec["ratio"] for rec in records]
        retimed = retime_entries(entries, ref_segments, ratios)
        final_entries, adjustments = enforce_constraints(
            retimed, cfg.max_sub_duration, cfg.min_sub_duration, video_duration
        )
        result["subtitles"] = {
            "input_count": len(entries),
            "output_count": len(final_entries),
            "adjustments": adjustments,
        }

        aligned_wav = os.path.join(tmp_dir, "aligned.wav")
        srt_path = os.path.join(tmp_dir, "subs.srt")
        write_wav(aligned_wav, track, rate)
        write_srt(final_entries, srt_path)

        out_name = f"{lang}.mp4"
        out_path = os.path.join(out_dir, out_name)
        mux(video_path, aligned_wav, srt_path, out_path)
        durations = probe_stream_durations(out_path)
        audio_dur = durations.get("audio", 0.0)
        video_dur = durations.get("video", video_duration)
        result["output"] = {
            "file": out_name,
            "audio_duration": round(audio_dur, 6),
            "video_duration": round(video_dur, 6),
            "duration_diff": round(audio_dur - video_dur, 6),
        }
        subs_json = os.path.join(out_dir, f"{lang}.subtitles.json")
        with open(subs_json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(final_entries, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        result["status"] = "ok"
    except DubError as exc:
        result["errors"].append(exc.to_dict())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return result


def prepare_reference(video_path: str, tmp_root: str, cfg: JobConfig):
    """Extract reference dialogue segments and video duration deterministically."""
    os.makedirs(tmp_root, exist_ok=True)
    ref_wav = os.path.join(tmp_root, "reference.wav")
    extract_audio(video_path, ref_wav, cfg.sample_rate)
    samples, rate = read_wav(ref_wav)
    segments, _ = detect_segments(samples, rate, cfg.detect)
    duration = probe_duration(video_path)
    return segments, duration


def run(
    video_path: str,
    languages: dict[str, tuple[str, str]],
    out_dir: str,
    report_path: str,
    cfg: JobConfig,
) -> int:
    """Run the full pipeline. Returns the process exit code."""
    os.makedirs(out_dir, exist_ok=True)
    tmp_root = os.path.join(out_dir, ".tmp")
    report: dict = {
        "tool": {"name": "dubmux", "version": __version__},
        "config": cfg.to_report_dict(),
        "inputs": {
            "video": file_digest(video_path),
            "languages": {
                lang: {"dub": file_digest(w), "subtitles": file_digest(s)}
                for lang, (w, s) in sorted(languages.items())
            },
        },
    }
    try:
        ref_segments, video_duration = prepare_reference(video_path, tmp_root, cfg)
    except DubError as exc:
        report["reference"] = {"error": exc.to_dict()}
        report["languages"] = {}
        report["summary"] = {"total": len(languages), "succeeded": 0,
                             "failed": len(languages), "exit_code": EXIT_FATAL}
        write_report(report_path, report)
        print(f"[fatal] {exc.code}: {exc.message}")
        return EXIT_FATAL
    report["reference"] = {
        "video_duration": round(video_duration, 6),
        "sample_rate": cfg.sample_rate,
        "detection": cfg.detect.to_dict(),
        "segments": [
            {"index": i, "start": round(s, 6), "end": round(e, 6),
             "duration": round(e - s, 6)}
            for i, (s, e) in enumerate(ref_segments)
        ],
    }

    lang_results: dict[str, dict] = {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, cfg.jobs)) as pool:
            futures = {
                lang: pool.submit(
                    process_language, lang, w, s, ref_segments, video_duration,
                    video_path, out_dir, tmp_root, cfg,
                )
                for lang, (w, s) in sorted(languages.items())
            }
            for lang, fut in futures.items():
                try:
                    lang_results[lang] = fut.result()
                except Exception as exc:  # defensive: keep one language from killing others
                    lang_results[lang] = {
                        "status": "failed",
                        "errors": [{"code": "E1999", "message": f"unexpected: {exc}"}],
                    }
    except KeyboardInterrupt:
        print("[fatal] aborted by user; temporary files cleaned up")
        return EXIT_FATAL
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    report["languages"] = {lang: lang_results[lang] for lang in sorted(lang_results)}
    succeeded = sum(1 for r in lang_results.values() if r["status"] == "ok")
    failed = len(lang_results) - succeeded
    if failed == 0:
        exit_code = EXIT_OK
    elif succeeded == 0:
        exit_code = EXIT_FATAL
    else:
        exit_code = EXIT_PARTIAL
    report["summary"] = {
        "total": len(lang_results),
        "succeeded": succeeded,
        "failed": failed,
        "exit_code": exit_code,
    }
    write_report(report_path, report)
    for lang in sorted(lang_results):
        res = lang_results[lang]
        if res["status"] == "ok":
            print(f"[ok] {lang}: {res['output']['file']}")
        else:
            for err in res["errors"]:
                print(f"[failed] {lang}: {err['code']}: {err['message']}")
    print(f"[summary] {succeeded} ok, {failed} failed, exit={exit_code}")
    return exit_code
