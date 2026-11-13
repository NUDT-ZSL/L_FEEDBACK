"""Deterministic local fixtures and end-to-end self verification."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import struct
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from . import codes
from .model import PipelineConfig
from .runner import PipelineRunner
from .tools import Toolchain
from .util import atomic_write_json, probe_wav, utc_now


MARKER = ".audio-pipeline-selftest"
DURATION_SEC = 4.0
DIALOGUE_FREQ = 220.0
MUSIC_FREQ = 660.0
AMBIENCE_FREQ = 330.0


@dataclass
class Fixture:
    job_id: str
    manifest: dict[str, Any]
    expected_status: str
    expected_error: Optional[str] = None
    expected_clamp: str = "none"
    fade_overlap: bool = False


def _write_tone(
    path: Path,
    *,
    sample_rate: int,
    channels: int,
    frequency: float,
    amplitude: float,
    phase: float,
    duration: float = DURATION_SEC,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(round(duration * sample_rate))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        blocks: list[bytes] = []
        for index in range(frames):
            t = index / sample_rate
            base = math.sin(2.0 * math.pi * frequency * t + phase)
            if channels == 1:
                left = right = base * amplitude
            elif channels == 2:
                left = base * amplitude * 0.92
                right = base * amplitude
            else:
                left = base * amplitude * 0.92
                right = base * amplitude
            left_sample = max(-32768, min(32767, int(round(left * 32767.0))))
            right_sample = max(-32768, min(32767, int(round(right * 32767.0))))
            if channels == 1:
                blocks.append(struct.pack("<h", left_sample))
            elif channels == 2:
                blocks.append(struct.pack("<hh", left_sample, right_sample))
            else:
                middle_sample = max(-32768, min(32767, int(round((left + right) * 0.5 * 32767.0))))
                blocks.append(struct.pack("<hhh", left_sample, right_sample, middle_sample))
        output.writeframes(b"".join(blocks))


def _manifest(
    *,
    dialogue: str,
    music: str,
    ambience: str,
    target: float,
    fade_start: float,
    fade_duration: float,
    music_gain: float,
    ambience_gain: float,
    max_attenuation: float,
    max_boost: float,
    nested: bool = False,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "target_lufs": target,
        "music_fade_start_sec": fade_start,
        "music_fade_duration_sec": fade_duration,
        "music_gain_db": music_gain,
        "ambience_gain_db": ambience_gain,
        "max_attenuation_db": max_attenuation,
        "max_boost_db": max_boost,
    }
    if nested:
        values["tracks"] = {"dialogue": dialogue, "music": music, "ambience": ambience}
    else:
        values.update(
            {
                "dialogue_path": dialogue,
                "music_path": music,
                "ambience_path": ambience,
            }
        )
    return values


def _write_show(base: Path, fixture: Fixture, rng: random.Random) -> None:
    show_dir = base / fixture.job_id
    show_dir.mkdir(parents=True, exist_ok=True)
    values = fixture.manifest
    dialogue = values.get("dialogue_path") or values.get("tracks", {}).get("dialogue")
    music = values.get("music_path") or values.get("tracks", {}).get("music")
    ambience = values.get("ambience_path") or values.get("tracks", {}).get("ambience")
    if not isinstance(dialogue, str) or not isinstance(music, str) or not isinstance(ambience, str):
        raise AssertionError("fixture paths must be strings")
    profiles = {
        "01_stereo48_normal": ((48000, 2), (44100, 1), (48000, 1)),
        "02_fade_overlap_bed": ((44100, 1), (48000, 2), (44100, 2)),
        "03_clamp_boost": ((48000, 1), (44100, 2), (48000, 2)),
        "04_clamp_attenuation": ((44100, 2), (48000, 1), (44100, 1)),
    }
    profile = profiles.get(fixture.job_id, ((48000, 2), (44100, 1), (48000, 1)))
    if fixture.expected_error != codes.MANIFEST_INVALID:
        dialogue_spec, music_spec, ambience_spec = profile
        _write_tone(
            show_dir / dialogue,
            sample_rate=dialogue_spec[0],
            channels=dialogue_spec[1],
            frequency=DIALOGUE_FREQ,
            amplitude=0.16,
            phase=rng.random() * math.tau,
        )
        _write_tone(
            show_dir / music,
            sample_rate=music_spec[0],
            channels=music_spec[1],
            frequency=MUSIC_FREQ,
            amplitude=0.24,
            phase=rng.random() * math.tau,
        )
        _write_tone(
            show_dir / ambience,
            sample_rate=ambience_spec[0],
            channels=ambience_spec[1],
            frequency=AMBIENCE_FREQ,
            amplitude=0.07,
            phase=rng.random() * math.tau,
        )
    (show_dir / "manifest.json").write_text(
        json.dumps(fixture.manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _prepare_fixtures(base: Path) -> list[Fixture]:
    rng = random.Random(0xA0D10F1C)
    shows = base / "shows"
    fixtures = [
        Fixture(
            "01_stereo48_normal",
            _manifest(
                dialogue="audio/dialogue.wav",
                music="audio/music.wav",
                ambience="audio/ambience.wav",
                target=-16.0,
                fade_start=1.0,
                fade_duration=1.5,
                music_gain=-6.0,
                ambience_gain=-12.0,
                max_attenuation=12.0,
                max_boost=12.0,
            ),
            "completed",
        ),
        Fixture(
            "02_fade_overlap_bed",
            _manifest(
                dialogue="dialogue.wav",
                music="bed/music.wav",
                ambience="room/ambience.wav",
                target=-15.0,
                fade_start=1.0,
                fade_duration=1.5,
                music_gain=-3.0,
                ambience_gain=-14.0,
                max_attenuation=20.0,
                max_boost=20.0,
                nested=True,
            ),
            "completed",
            fade_overlap=True,
        ),
        Fixture(
            "03_clamp_boost",
            _manifest(
                dialogue="dialogue.wav",
                music="music.wav",
                ambience="ambience.wav",
                target=-5.0,
                fade_start=2.0,
                fade_duration=1.0,
                music_gain=-8.0,
                ambience_gain=-12.0,
                max_attenuation=12.0,
                max_boost=6.0,
            ),
            "completed",
            expected_clamp="max_boost",
        ),
        Fixture(
            "04_clamp_attenuation",
            _manifest(
                dialogue="dialogue.wav",
                music="music.wav",
                ambience="ambience.wav",
                target=-50.0,
                fade_start=1.5,
                fade_duration=1.0,
                music_gain=-6.0,
                ambience_gain=-12.0,
                max_attenuation=6.0,
                max_boost=12.0,
            ),
            "completed",
            expected_clamp="max_attenuation",
        ),
        Fixture(
            "05_missing_source",
            _manifest(
                dialogue="missing-dialogue.wav",
                music="music.wav",
                ambience="ambience.wav",
                target=-16.0,
                fade_start=1.0,
                fade_duration=1.0,
                music_gain=-6.0,
                ambience_gain=-12.0,
                max_attenuation=12.0,
                max_boost=12.0,
            ),
            "failed",
            expected_error=codes.SOURCE_MISSING,
        ),
        Fixture(
            "06_invalid_manifest",
            {
                "dialogue_path": "dialogue.wav",
                "music_path": "music.wav",
                "ambience_path": "ambience.wav",
                "music_fade_start_sec": 1.0,
                "music_fade_duration_sec": 1.0,
                "music_gain_db": -6.0,
                "ambience_gain_db": -12.0,
            },
            "failed",
            expected_error=codes.MANIFEST_INVALID,
        ),
        Fixture(
            "07_unsupported_rate",
            _manifest(
                dialogue="dialogue-32k.wav",
                music="music.wav",
                ambience="ambience.wav",
                target=-16.0,
                fade_start=1.0,
                fade_duration=1.0,
                music_gain=-6.0,
                ambience_gain=-12.0,
                max_attenuation=12.0,
                max_boost=12.0,
            ),
            "failed",
            expected_error=codes.UNSUPPORTED_AUDIO_FORMAT,
        ),
        Fixture(
            "08_unsupported_channels",
            _manifest(
                dialogue="dialogue-3ch.wav",
                music="music.wav",
                ambience="ambience.wav",
                target=-16.0,
                fade_start=1.0,
                fade_duration=1.0,
                music_gain=-6.0,
                ambience_gain=-12.0,
                max_attenuation=12.0,
                max_boost=12.0,
            ),
            "failed",
            expected_error=codes.UNSUPPORTED_AUDIO_FORMAT,
        ),
    ]

    for fixture in fixtures:
        show_dir = shows / fixture.job_id
        if fixture.expected_error == codes.MANIFEST_INVALID:
            show_dir.mkdir(parents=True, exist_ok=True)
            (show_dir / "manifest.json").write_text(
                json.dumps(fixture.manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            continue
        if fixture.job_id == "05_missing_source":
            show_dir.mkdir(parents=True, exist_ok=True)
            _write_tone(
                show_dir / "music.wav",
                sample_rate=44100,
                channels=2,
                frequency=MUSIC_FREQ,
                amplitude=0.24,
                phase=rng.random() * math.tau,
            )
            _write_tone(
                show_dir / "ambience.wav",
                sample_rate=48000,
                channels=1,
                frequency=AMBIENCE_FREQ,
                amplitude=0.07,
                phase=rng.random() * math.tau,
            )
            (show_dir / "manifest.json").write_text(
                json.dumps(fixture.manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            continue
        if fixture.job_id == "07_unsupported_rate":
            show_dir.mkdir(parents=True, exist_ok=True)
            _write_tone(
                show_dir / "dialogue-32k.wav",
                sample_rate=32000,
                channels=2,
                frequency=DIALOGUE_FREQ,
                amplitude=0.16,
                phase=rng.random() * math.tau,
            )
            _write_tone(
                show_dir / "music.wav",
                sample_rate=44100,
                channels=1,
                frequency=MUSIC_FREQ,
                amplitude=0.24,
                phase=rng.random() * math.tau,
            )
            _write_tone(
                show_dir / "ambience.wav",
                sample_rate=48000,
                channels=1,
                frequency=AMBIENCE_FREQ,
                amplitude=0.07,
                phase=rng.random() * math.tau,
            )
            (show_dir / "manifest.json").write_text(
                json.dumps(fixture.manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            continue
        if fixture.job_id == "08_unsupported_channels":
            show_dir.mkdir(parents=True, exist_ok=True)
            _write_tone(
                show_dir / "dialogue-3ch.wav",
                sample_rate=48000,
                channels=3,
                frequency=DIALOGUE_FREQ,
                amplitude=0.16,
                phase=rng.random() * math.tau,
            )
            _write_tone(
                show_dir / "music.wav",
                sample_rate=44100,
                channels=1,
                frequency=MUSIC_FREQ,
                amplitude=0.24,
                phase=rng.random() * math.tau,
            )
            _write_tone(
                show_dir / "ambience.wav",
                sample_rate=48000,
                channels=1,
                frequency=AMBIENCE_FREQ,
                amplitude=0.07,
                phase=rng.random() * math.tau,
            )
            (show_dir / "manifest.json").write_text(
                json.dumps(fixture.manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            continue
        _write_show(shows, fixture, rng)
    return fixtures


def _runner(
    base: Path,
    ffmpeg: Path,
    ffprobe: Path,
    *,
    jobs: int,
    result_name: str,
    journal: Optional[Path] = None,
    temp_limit_mb: int = 256,
) -> PipelineRunner:
    return PipelineRunner(
        PipelineConfig(
            root=base / "shows",
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            jobs=jobs,
            work_dir=base / "work",
            job_temp_limit_mb=temp_limit_mb,
            min_free_mb=16,
            journal_path=journal or base / "shows" / "audio_pipeline_jobs.json",
            result_path=base / result_name,
        )
    )


def _by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {job["id"]: job for job in result["jobs"]}


def _read_left_channel(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 2 or source.getsampwidth() != 2:
            raise AssertionError("expected stereo PCM16 output")
        rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    values = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    return [values[index] for index in range(0, len(values), 2)], rate


def _tone_amplitude(samples: list[int], rate: int, start_sec: float, end_sec: float, frequency: float) -> float:
    start = int(round(start_sec * rate))
    end = int(round(end_sec * rate))
    count = end - start
    if count <= 0 or end > len(samples):
        return 0.0
    real = 0.0
    imaginary = 0.0
    for index in range(count):
        angle = 2.0 * math.pi * frequency * index / rate
        value = float(samples[start + index])
        real += value * math.cos(angle)
        imaginary -= value * math.sin(angle)
    return 2.0 * math.hypot(real, imaginary) / count


def _check_fade(path: Path) -> tuple[bool, dict[str, float]]:
    samples, rate = _read_left_channel(path)
    before = _tone_amplitude(samples, rate, 0.20, 0.40, MUSIC_FREQ)
    middle = _tone_amplitude(samples, rate, 1.70, 1.90, MUSIC_FREQ)
    after = _tone_amplitude(samples, rate, 2.70, 2.90, MUSIC_FREQ)
    ratio_mid = middle / before if before > 0 else 0.0
    ratio_after = after / before if before > 0 else 1.0
    ok = before > 100.0 and 0.25 <= ratio_mid <= 0.70 and ratio_after <= 0.05
    return ok, {
        "before_amplitude": round(before, 3),
        "mid_fade_amplitude": round(middle, 3),
        "after_fade_amplitude": round(after, 3),
        "mid_ratio": round(ratio_mid, 4),
        "after_ratio": round(ratio_after, 6),
    }


def _journal_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [entry["id"] for entry in data["jobs"]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio_pipeline.py self-test",
        description="Generate deterministic local fixtures and verify the full pipeline.",
    )
    parser.add_argument("--dir", default=".audio-pipeline-selftest", help="self-test workspace")
    parser.add_argument("--ffmpeg", help="path to the local ffmpeg executable")
    parser.add_argument("--ffprobe", help="path to the local ffprobe executable")
    parser.add_argument("--tools-dir", help="directory containing ffmpeg and ffprobe")
    parser.add_argument("--jobs", type=int, default=3, help="concurrency for fixture runs (default: 3)")
    parser.add_argument("--json", action="store_true", help="print the self-test report as JSON")
    return parser


def _prepare_base(path: Path) -> Path:
    base = path.resolve()
    if base == Path(base.anchor) or base == Path.home():
        raise RuntimeError(f"refusing to use filesystem root or home directory as self-test workspace: {base}")
    if base.exists():
        marker = base / MARKER
        if marker.is_file():
            shutil.rmtree(base)
        elif any(base.iterdir()):
            raise RuntimeError(
                f"refusing to erase non-empty directory without {MARKER} marker: {base}"
            )
    base.mkdir(parents=True, exist_ok=True)
    (base / MARKER).write_text("generated by audio_pipeline.py self-test\n", encoding="utf-8")
    (base / "shows").mkdir(exist_ok=True)
    (base / "work").mkdir(exist_ok=True)
    return base


def self_test_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    ffmpeg = Path(args.ffmpeg) if args.ffmpeg else None
    ffprobe = Path(args.ffprobe) if args.ffprobe else None
    if args.tools_dir:
        tools_dir = Path(args.tools_dir)
        ffmpeg = ffmpeg or tools_dir / "ffmpeg"
        ffprobe = ffprobe or tools_dir / "ffprobe"
    if ffmpeg is None or ffprobe is None:
        parser.error("provide --ffmpeg and --ffprobe (or --tools-dir)")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    started = utc_now()
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(condition), "detail": detail})

    try:
        base = _prepare_base(Path(args.dir))
    except (OSError, RuntimeError) as exc:
        print(f"SELF-TEST ERROR: {exc}", file=sys.stderr)
        return codes.EXIT_FATAL

    fixtures = _prepare_fixtures(base)
    expected = {fixture.job_id: fixture for fixture in fixtures}
    journal_path = base / "shows" / "audio_pipeline_jobs.json"

    first = _runner(base, ffmpeg, ffprobe, jobs=args.jobs, result_name="first_result.json").run()
    if first.get("fatal_error"):
        print(
            "SELF-TEST ERROR: initial pipeline run failed: "
            + json.dumps(first["fatal_error"], ensure_ascii=False),
            file=sys.stderr,
        )
        return codes.EXIT_FATAL
    first_jobs = _by_id(first)
    check("first_run_has_all_jobs", len(first_jobs) == len(fixtures), len(first_jobs))
    check("first_run_exit_partial", first["exit_code"] == codes.EXIT_PARTIAL, first["exit_code"])
    check(
        "first_run_counts",
        first["counts"] == {"total": 8, "completed": 4, "skipped": 0, "failed": 4, "interrupted": 0},
        first["counts"],
    )

    for fixture in fixtures:
        job = first_jobs.get(fixture.job_id)
        check(f"{fixture.job_id}_present", job is not None)
        if job is None:
            continue
        check(f"{fixture.job_id}_status", job["status"] == fixture.expected_status, job["status"])
        if fixture.expected_error:
            check(
                f"{fixture.job_id}_error_code",
                (job.get("error") or {}).get("code") == fixture.expected_error,
                job.get("error"),
            )
            check(f"{fixture.job_id}_exit_code", job["job_exit_code"] == 1, job["job_exit_code"])
        else:
            check(f"{fixture.job_id}_job_exit_code", job["job_exit_code"] == 0, job["job_exit_code"])
            check(
                f"{fixture.job_id}_loudness_measured",
                isinstance(job.get("measured_dialogue_lufs"), (int, float)),
                job.get("measured_dialogue_lufs"),
            )
            check(
                f"{fixture.job_id}_gain_applied",
                isinstance(job.get("applied_gain_db"), (int, float)),
                job.get("applied_gain_db"),
            )
            if fixture.expected_clamp == "none":
                target = float(fixture.manifest.get("target_lufs", -16.0))
                expected_applied = target - float(job["measured_dialogue_lufs"])
                check(
                    f"{fixture.job_id}_gain_matches_dialogue_baseline",
                    abs(float(job["applied_gain_db"]) - expected_applied) <= 1e-3,
                    {"actual": job.get("applied_gain_db"), "expected": round(expected_applied, 4)},
                )
            output = base / "shows" / str(job["output"])
            try:
                wav_info = probe_wav(output)
                artifact_ok = (
                    wav_info["sample_rate"] == 48000
                    and wav_info["channels"] == 2
                    and wav_info["bits_per_sample"] == 16
                    and wav_info["riff_size"] == wav_info["file_size"]
                    and wav_info["data_size"] + 44 == wav_info["file_size"]
                    and job.get("output_validated") is True
                    and job.get("output_size") == wav_info["file_size"]
                )
                check(f"{fixture.job_id}_valid_wav", artifact_ok, wav_info)
            except (OSError, ValueError, wave.Error) as exc:
                check(f"{fixture.job_id}_valid_wav", False, str(exc))

        if fixture.expected_clamp != "none":
            expected_gain = (
                expected[fixture.job_id].manifest["max_boost_db"]
                if fixture.expected_clamp == "max_boost"
                else -expected[fixture.job_id].manifest["max_attenuation_db"]
            )
            check(f"{fixture.job_id}_clamped_flag", job.get("clamped") is True, job.get("clamped"))
            check(
                f"{fixture.job_id}_clamp_direction",
                job.get("clamp") == fixture.expected_clamp,
                job.get("clamp"),
            )
            check(
                f"{fixture.job_id}_clamp_value",
                abs(float(job.get("applied_gain_db", 999999)) - expected_gain) <= 1e-3,
                {"actual": job.get("applied_gain_db"), "expected": expected_gain},
            )
        elif job["status"] == "completed":
            check(f"{fixture.job_id}_not_clamped", job.get("clamped") is False, job.get("clamped"))

    fade_metrics: Optional[dict[str, float]] = None
    fade_job = first_jobs.get("02_fade_overlap_bed")
    if fade_job and fade_job.get("output"):
        try:
            fade_ok, fade_metrics = _check_fade(base / "shows" / fade_job["output"])
            check("linear_fade_with_dialogue_overlap", fade_ok, fade_metrics)
        except (OSError, ValueError, wave.Error) as exc:
            check("linear_fade_with_dialogue_overlap", False, str(exc))
    else:
        check("linear_fade_with_dialogue_overlap", False, "missing output")

    completed_jobs = [job for job in first["jobs"] if job["status"] == "completed"]
    direct_toolchain = Toolchain(ffmpeg, ffprobe)
    dialogue_sources = {
        "01_stereo48_normal": "audio/dialogue.wav",
        "02_fade_overlap_bed": "dialogue.wav",
        "03_clamp_boost": "dialogue.wav",
        "04_clamp_attenuation": "dialogue.wav",
    }
    direct_checks: list[dict[str, Any]] = []
    for job in completed_jobs:
        source = base / "shows" / job["id"] / dialogue_sources[job["id"]]
        try:
            direct_lufs = direct_toolchain.measure_integrated_lufs(source)
            delta = round(float(job["measured_dialogue_lufs"]) - direct_lufs, 4)
        except Exception as exc:
            direct_checks.append({"id": job["id"], "error": str(exc), "delta": None})
            continue
        direct_checks.append(
            {
                "id": job["id"],
                "pipeline": job.get("measured_dialogue_lufs"),
                "dialogue_source": round(direct_lufs, 4),
                "delta": delta,
            }
        )
    check(
        "dialogue_loudness_measured_without_mix_or_fade",
        direct_checks
        and all(item["delta"] is not None and abs(item["delta"]) <= 0.2 for item in direct_checks),
        direct_checks,
    )
    profile_set = {
        (track["sample_rate"], track["channels"])
        for job in completed_jobs
        for track in job.get("tracks", [])
    }
    profile_detail = sorted([list(item) for item in profile_set])
    check(
        "fixtures_cover_44100_and_48000",
        {44100, 48000}.issubset({item[0] for item in profile_set}),
        profile_detail,
    )
    check(
        "fixtures_cover_mono_and_stereo",
        {1, 2}.issubset({item[1] for item in profile_set}),
        profile_detail,
    )

    ids_first = _journal_ids(journal_path)
    check("journal_no_duplicate_ids", len(ids_first) == len(set(ids_first)), ids_first)
    check("journal_has_one_entry_per_job", len(ids_first) == len(fixtures), len(ids_first))

    second = _runner(base, ffmpeg, ffprobe, jobs=args.jobs, result_name="resume_result.json").run()
    second_jobs = _by_id(second)
    check("resume_exit_partial", second["exit_code"] == codes.EXIT_PARTIAL, second["exit_code"])
    check(
        "resume_counts",
        second["counts"] == {"total": 8, "completed": 0, "skipped": 4, "failed": 4, "interrupted": 0},
        second["counts"],
    )
    for fixture in fixtures:
        if fixture.expected_status == "completed":
            check(
                f"{fixture.job_id}_skipped_on_resume",
                second_jobs[fixture.job_id]["status"] == "skipped",
                second_jobs[fixture.job_id]["status"],
            )
    ids_second = _journal_ids(journal_path)
    check("resume_no_duplicate_ids", len(ids_second) == len(set(ids_second)), ids_second)
    check("resume_one_entry_per_job", len(ids_second) == len(fixtures), len(ids_second))

    changed_manifest_path = base / "shows" / "01_stereo48_normal" / "manifest.json"
    changed_manifest = json.loads(changed_manifest_path.read_text(encoding="utf-8"))
    changed_manifest["music_gain_db"] = -9.0
    changed_manifest_path.write_text(
        json.dumps(changed_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    changed = _runner(base, ffmpeg, ffprobe, jobs=args.jobs, result_name="changed_result.json").run()
    changed_jobs = _by_id(changed)
    check(
        "changed_fingerprint_reruns_job",
        changed_jobs["01_stereo48_normal"]["status"] == "completed",
        changed_jobs["01_stereo48_normal"]["status"],
    )
    check(
        "unchanged_jobs_still_skip",
        all(changed_jobs[job_id]["status"] == "skipped" for job_id in ("02_fade_overlap_bed", "03_clamp_boost", "04_clamp_attenuation")),
        {job_id: changed_jobs[job_id]["status"] for job_id in changed_jobs},
    )
    ids_changed = _journal_ids(journal_path)
    check("changed_job_replaces_not_appends", len(ids_changed) == len(set(ids_changed)) == 8, ids_changed)

    changed_source = base / "shows" / "02_fade_overlap_bed" / "bed" / "music.wav"
    _write_tone(
        changed_source,
        sample_rate=48000,
        channels=2,
        frequency=MUSIC_FREQ,
        amplitude=0.20,
        phase=1.0,
    )
    source_changed = _runner(
        base, ffmpeg, ffprobe, jobs=args.jobs, result_name="source_changed_result.json"
    ).run()
    source_changed_jobs = _by_id(source_changed)
    check(
        "source_content_change_reruns_job",
        source_changed_jobs["02_fade_overlap_bed"]["status"] == "completed",
        source_changed_jobs["02_fade_overlap_bed"]["status"],
    )
    check(
        "source_content_change_keeps_other_jobs_skipped",
        all(
            source_changed_jobs[job_id]["status"] == "skipped"
            for job_id in ("01_stereo48_normal", "03_clamp_boost", "04_clamp_attenuation")
        ),
        {job_id: source_changed_jobs[job_id]["status"] for job_id in source_changed_jobs},
    )

    deleted_output = base / "shows" / str(changed_jobs["02_fade_overlap_bed"]["output"])
    deleted_output.unlink()
    deleted = _runner(base, ffmpeg, ffprobe, jobs=args.jobs, result_name="deleted_result.json").run()
    deleted_jobs = _by_id(deleted)
    check(
        "missing_artifact_reruns_job",
        deleted_jobs["02_fade_overlap_bed"]["status"] == "completed",
        deleted_jobs["02_fade_overlap_bed"]["status"],
    )
    check(
        "missing_artifact_does_not_duplicate_journal",
        len(set(_journal_ids(journal_path))) == 8,
        _journal_ids(journal_path),
    )

    stale = deleted_output.parent / ".mix.part-deadbeef.wav"
    stale.write_bytes(b"not-a-wav")
    stale_work = base / "work" / "jobs" / "job-deadbeef"
    stale_work.mkdir(parents=True, exist_ok=True)
    (stale_work / "decode.partial").write_bytes(b"partial")
    after_stale = _runner(base, ffmpeg, ffprobe, jobs=args.jobs, result_name="stale_result.json").run()
    check("stale_final_part_does_not_break_skip", after_stale["counts"]["skipped"] == 4, after_stale["counts"])
    check("stale_files_not_referenced", all((job.get("output") or "").find(".part") < 0 for job in after_stale["jobs"]))

    temp_test = _runner(
        base,
        ffmpeg,
        ffprobe,
        jobs=args.jobs,
        result_name="temp_limit_result.json",
        journal=base / "temp_limit_journal.json",
        temp_limit_mb=1,
    ).run()
    temp_jobs = _by_id(temp_test)
    check(
        "temp_space_business_code",
        temp_jobs["01_stereo48_normal"]["status"] == "failed"
        and temp_jobs["01_stereo48_normal"]["error"]["code"] == codes.TEMP_SPACE_INSUFFICIENT,
        temp_jobs["01_stereo48_normal"].get("error"),
    )

    missing_tool = PipelineRunner(
        PipelineConfig(
            root=base / "shows",
            ffmpeg=base / "does-not-exist-ffmpeg",
            ffprobe=ffprobe,
            jobs=args.jobs,
            work_dir=base / "work",
            journal_path=base / "fatal_journal.json",
            result_path=base / "fatal_result.json",
        )
    ).run()
    check("missing_ffmpeg_is_fatal", missing_tool["exit_code"] == codes.EXIT_FATAL, missing_tool["exit_code"])
    check(
        "missing_ffmpeg_code",
        (missing_tool.get("fatal_error") or {}).get("code") == codes.FFMPEG_UNAVAILABLE,
        missing_tool.get("fatal_error"),
    )
    check(
        "fatal_counts_consistent",
        missing_tool["counts"]["total"] == missing_tool["counts"]["failed"] == 8,
        missing_tool["counts"],
    )

    finished = utc_now()
    passed = all(item["ok"] for item in checks)
    report = {
        "ok": passed,
        "started_at": started,
        "finished_at": finished,
        "fixture_root": str(base),
        "first_pipeline_exit_code": first["exit_code"],
        "first_counts": first["counts"],
        "fade_metrics": fade_metrics,
        "jobs": [
            {
                "id": job["id"],
                "status": job["status"],
                "job_exit_code": job["job_exit_code"],
                "measured_dialogue_lufs": job["measured_dialogue_lufs"],
                "requested_gain_db": job["requested_gain_db"],
                "applied_gain_db": job["applied_gain_db"],
                "clamped": job["clamped"],
                "clamp": job["clamp"],
                "error_code": (job.get("error") or {}).get("code"),
            }
            for job in first["jobs"]
        ],
        "resume": {
            "second_run_counts": second["counts"],
            "changed_job_status": changed_jobs["01_stereo48_normal"]["status"],
            "source_changed_job_status": source_changed_jobs["02_fade_overlap_bed"]["status"],
            "deleted_output_status": deleted_jobs["02_fade_overlap_bed"]["status"],
            "stale_run_counts": after_stale["counts"],
            "journal_ids": _journal_ids(journal_path),
        },
        "checks": checks,
        "passed_checks": sum(1 for item in checks if item["ok"]),
        "total_checks": len(checks),
    }
    try:
        atomic_write_json(base / "selftest_report.json", report)
    except OSError as exc:
        print(f"SELF-TEST ERROR: cannot write report: {exc}", file=sys.stderr)
        return codes.EXIT_FATAL

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"SELF-TEST {'PASS' if passed else 'FAIL'}")
        print(f"fixture_root={base}")
        print(f"first_pipeline_exit_code={first['exit_code']} counts={first['counts']}")
        print("id | lufs | requested_db | applied_db | clamped | status | job_exit")
        for entry in report["jobs"]:
            print(
                f"{entry['id']} | {entry['measured_dialogue_lufs']} | {entry['requested_gain_db']} | "
                f"{entry['applied_gain_db']} | {entry['clamped']}:{entry['clamp']} | "
                f"{entry['status']} | {entry['job_exit_code']} | {entry['error_code'] or '-'}"
            )
        print(f"fade_metrics={json.dumps(fade_metrics, sort_keys=True)}")
        print(f"resume_counts={second['counts']}")
        print(f"changed_job={changed_jobs['01_stereo48_normal']['status']}")
        print(f"source_changed_job={source_changed_jobs['02_fade_overlap_bed']['status']}")
        print(f"deleted_output={deleted_jobs['02_fade_overlap_bed']['status']}")
        print(f"stale_run_counts={after_stale['counts']}")
        print(f"checks={report['passed_checks']}/{report['total_checks']}")
        for item in checks:
            if not item["ok"]:
                print(f"FAILED CHECK {item['name']}: {item['detail']}")
        print(f"report={base / 'selftest_report.json'}")
    return codes.EXIT_OK if passed else codes.EXIT_PARTIAL
