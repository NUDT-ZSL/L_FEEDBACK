import json
from pathlib import Path

import numpy as np
import soundfile as sf

from locdub import ffmpeg_utils
from locdub.cli import main
from locdub.constants import EXIT_FATAL, EXIT_OK, EXIT_PARTIAL

SR = 44100


def _write_tone(path: Path, seconds: float, frequency: float) -> None:
    frames = int(SR * seconds)
    t = np.arange(frames) / SR
    data = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data[:, None], SR, subtype="PCM_16")


def _build_fixture(tmp_path: Path, *, omit=()):
    video = tmp_path / "source.mp4"
    ffmpeg_utils.make_solid_video(video, 12000)
    specs = [
        ("s1", 500, 2500, 1.6, 330.0),
        ("s2", 3000, 5500, 2.0, 440.0),
        ("s3", 6000, 9000, 2.4, 520.0),
        ("s4", 9500, 11500, 1.2, 392.0),
    ]
    languages = {"en": 1.0, "ja": 1.25}
    timeline_segments = []
    for segment_id, start, end, spoken, freq in specs:
        timeline_segments.append({
            "id": segment_id, "start_ms": start, "end_ms": end,
            "audio": f"{segment_id}.wav", "ratio_range": [0.5, 2.0],
        })
        for language, factor in languages.items():
            if segment_id in omit and language == "ja":
                continue
            _write_tone(tmp_path / "dubs" / language / f"{segment_id}.wav",
                        spoken * factor, freq)
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps({
        "video_duration_ms": 12000,
        "segments": timeline_segments,
    }), encoding="utf-8")
    return video, timeline


def test_full_success_exit_zero(tmp_path):
    video, timeline = _build_fixture(tmp_path)
    out = tmp_path / "out"
    code = main([
        "--timeline", str(timeline), "--video", str(video),
        "--lang", f"en={tmp_path/'dubs'/'en'}",
        "--lang", f"ja={tmp_path/'dubs'/'ja'}",
        "--out", str(out), "--workers", "2", "--gain-db", "-1",
        "--quiet",
    ])
    assert code == EXIT_OK
    report = json.loads((out / "report.json").read_text())
    assert report["summary"]["overall_status"] == "success"
    assert [lang["language"] for lang in report["languages"]] == ["en", "ja"]
    for language in report["languages"]:
        assert language["skipped_segments"] == 0
        assert Path(language["output_file"]).exists()


def test_missing_dub_is_skipped_with_placeholder(tmp_path):
    video, timeline = _build_fixture(tmp_path, omit=("s3",))
    out = tmp_path / "out"
    code = main([
        "--timeline", str(timeline), "--video", str(video),
        "--lang", f"en={tmp_path/'dubs'/'en'}",
        "--lang", f"ja={tmp_path/'dubs'/'ja'}",
        "--out", str(out), "--quiet",
    ])
    assert code == EXIT_PARTIAL
    report = json.loads((out / "report.json").read_text())
    ja = next(lang for lang in report["languages"] if lang["language"] == "ja")
    en = next(lang for lang in report["languages"] if lang["language"] == "en")
    assert ja["overall_status"] == "partial_success"
    assert en["overall_status"] == "success"
    skipped = next(s for s in ja["segments"] if s["segment_id"] == "s3")
    assert skipped["status"] == "skipped"
    assert skipped["silence_placeholder"] is True
    assert skipped["errors"]
    # MP4 for the partial language still exists.
    assert Path(ja["output_file"]).exists()
    # Error log file was written.
    assert (out / "error.log").exists()


def test_bad_timeline_json_is_fatal(tmp_path):
    video = tmp_path / "source.mp4"
    ffmpeg_utils.make_solid_video(video, 5000)
    timeline = tmp_path / "broken.json"
    timeline.write_text("{oops", encoding="utf-8")
    out = tmp_path / "out"
    code = main([
        "--timeline", str(timeline), "--video", str(video),
        "--lang", f"en={tmp_path/'dubs'}",
        "--out", str(out), "--quiet",
    ])
    assert code == EXIT_FATAL


def test_missing_video_is_fatal(tmp_path):
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps({
        "segments": [{"id": "a", "start_ms": 0, "end_ms": 100,
                      "audio": "a.wav"}],
    }), encoding="utf-8")
    code = main([
        "--timeline", str(timeline),
        "--video", str(tmp_path / "nope.mp4"),
        "--lang", f"en={tmp_path/'dubs'}",
        "--out", str(tmp_path / "out"), "--quiet",
    ])
    assert code == EXIT_FATAL


def test_report_status_vocabulary_consistent(tmp_path):
    video, timeline = _build_fixture(tmp_path, omit=("s2",))
    out = tmp_path / "out"
    main([
        "--timeline", str(timeline), "--video", str(video),
        "--lang", f"en={tmp_path/'dubs'/'en'}",
        "--lang", f"ja={tmp_path/'dubs'/'ja'}",
        "--out", str(out), "--quiet",
    ])
    report = json.loads((out / "report.json").read_text())
    allowed = set(report["summary"]["status_values"].values())
    for language in report["languages"]:
        assert language["overall_status"] in allowed


def test_non_wav_dub_is_skipped(tmp_path):
    video = tmp_path / "source.mp4"
    ffmpeg_utils.make_solid_video(video, 6000)
    dub_dir = tmp_path / "dubs" / "en"
    dub_dir.mkdir(parents=True)
    # A valid WAV segment plus a non-WAV segment.
    _write_tone(dub_dir / "good.wav", 1.0, 440.0)
    (dub_dir / "bad.mp3").write_text("not a wav", encoding="utf-8")
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps({
        "video_duration_ms": 6000,
        "segments": [
            {"id": "good", "start_ms": 500, "end_ms": 2000,
             "audio": "good.wav"},
            {"id": "bad", "start_ms": 3000, "end_ms": 4500,
             "audio": "bad.mp3"},
        ],
    }), encoding="utf-8")
    out = tmp_path / "out"
    code = main([
        "--timeline", str(timeline), "--video", str(video),
        "--lang", f"en={dub_dir}", "--out", str(out), "--quiet",
    ])
    assert code == EXIT_PARTIAL
    report = json.loads((out / "report.json").read_text())
    segments = {s["segment_id"]: s for s in report["languages"][0]["segments"]}
    assert segments["good"]["status"] == "ok"
    assert segments["bad"]["status"] == "skipped"
    assert segments["bad"]["silence_placeholder"] is True
