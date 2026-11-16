"""End-to-end verification of dubmux against deterministic local samples."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from dubmux import REPORT_FORMAT_VERSION
from dubmux.audio import DetectConfig, detect_segments, read_wav
from dubmux.errors import (
    EXIT_OK,
    EXIT_PARTIAL,
    E_SAMPLE_RATE_MISMATCH,
    E_SEGMENT_MISSING,
    E_SUBTITLE_OVERFLOW,
)
from tools import make_samples as ms

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(args, cwd=REPO_ROOT):
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "dubmux", *args],
        capture_output=True, text=True, cwd=cwd, env=env,
    )


def lang_args(paths, langs):
    args = []
    for lang in langs:
        args += ["--lang", f"{lang}={paths['dubs'][lang]}",
                 "--subs", f"{lang}={paths['subs'][lang]}"]
    return args


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    out = tmp_path_factory.mktemp("samples")
    return ms.generate(str(out))


@pytest.fixture(scope="module")
def standard_run(samples, tmp_path_factory):
    """Standard run; the report path is pre-seeded with a stale-version file."""
    out = tmp_path_factory.mktemp("out_standard")
    report = out / "report.json"
    report.write_text('{"format_version": 999, "stale": true}\njunk trailing\n')
    proc = run_cli([
        "--video", samples["video"], "--out", str(out), "--report", str(report),
        "--jobs", "2", *lang_args(samples, ["en", "ja", "de"]),
    ])
    data = json.loads(report.read_text())
    return {"proc": proc, "out": out, "report": data}


# --- requirement 2: deterministic silence detection -------------------------

def test_detection_deterministic(samples):
    cfg = DetectConfig()
    for lang, spec in ms.LANGUAGES.items():
        samples_a, rate = read_wav(samples["dubs"][lang])
        segs1, lead1 = detect_segments(samples_a, rate, cfg)
        segs2, lead2 = detect_segments(samples_a, rate, cfg)
        assert segs1 == segs2 and lead1 == lead2
        assert len(segs1) == len(ms.REFERENCE_SEGMENTS)
        assert abs(lead1 - spec["leading_silence"]) < 0.05


# --- requirement 3: local time stretch alignment ----------------------------

def test_stretch_alignment(standard_run):
    assert standard_run["proc"].returncode == EXIT_OK, standard_run["proc"].stdout
    report = standard_run["report"]
    tolerance = report["config"]["tolerance"]
    ref = report["reference"]["segments"]
    for lang, spec in ms.LANGUAGES.items():
        lang_report = report["languages"][lang]
        assert lang_report["status"] == "ok"
        stretched = lang_report["stretch"]["segments"]
        assert len(stretched) == len(ref)
        for rec, target in zip(stretched, ref):
            assert rec["boundary_error"] <= tolerance
            assert abs(rec["final_duration"] - target["duration"]) <= tolerance
            expected_ratio = 1.0 / spec["factor"]
            assert abs(rec["ratio"] - expected_ratio) < 0.02
        expected_leading = spec["leading_silence"]
        assert abs(lang_report["detection"]["leading_silence"] - expected_leading) < 0.05


# --- requirement 5: muxed output sync / duration metadata -------------------

def test_mux_duration_metadata(standard_run):
    report = standard_run["report"]
    for lang in ms.LANGUAGES:
        out = report["languages"][lang]["output"]
        assert abs(out["duration_diff"]) < 0.05
        mp4 = standard_run["out"] / out["file"]
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
             "-of", "json", str(mp4)],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(probe.stdout)["streams"]
        kinds = {s["codec_type"]: s["codec_name"] for s in streams}
        assert kinds["video"] == "h264"      # copied, not re-encoded
        assert kinds["audio"] == "aac"
        assert "subtitle" in kinds


# --- requirement 4: subtitle retiming constraints ---------------------------

@pytest.fixture(scope="module")
def constraint_run(samples, tmp_path_factory):
    """Run with tight subtitle constraints plus a merge-triggering language."""
    out = tmp_path_factory.mktemp("out_constraints")
    # Extra language reusing the en dub, with two short adjacent entries.
    seg0_start = ms.REFERENCE_SEGMENTS[0][0]
    xx_subs = [
        {"id": "xx-a", "start": seg0_start + 0.2, "end": seg0_start + 0.4,
         "text": "short one", "segment": 0},
        {"id": "xx-b", "start": seg0_start + 0.6, "end": seg0_start + 0.9,
         "text": "short two", "segment": 0},
    ] + ms.make_subtitles("xx")[1:]
    xx_path = os.path.join(str(out), "subs_xx.json")
    with open(xx_path, "w", encoding="utf-8") as fh:
        json.dump(xx_subs, fh)
    report = out / "report.json"
    proc = run_cli([
        "--video", samples["video"], "--out", str(out), "--report", str(report),
        "--max-sub-duration", "1.2", "--min-sub-duration", "0.5",
        *lang_args(samples, ["en", "ja", "de"]),
        "--lang", f"xx={samples['dubs']['en']}", "--subs", f"xx={xx_path}",
    ])
    return {"proc": proc, "out": out, "report": json.loads(report.read_text())}


def test_subtitle_constraints(constraint_run):
    assert constraint_run["proc"].returncode == EXIT_OK, constraint_run["proc"].stdout
    report = constraint_run["report"]
    max_d = report["config"]["max_sub_duration"]
    min_d = report["config"]["min_sub_duration"]
    for lang in ["en", "ja", "de", "xx"]:
        subs_file = constraint_run["out"] / f"{lang}.subtitles.json"
        entries = json.loads(subs_file.read_text())
        for e in entries:
            dur = e["end"] - e["start"]
            assert dur <= max_d + 1e-6, (lang, e)
            assert dur >= min_d - 1e-6, (lang, e)
        assert report["languages"][lang]["subtitles"]["adjustments"], lang
    rules = {a["rule"] for a in report["languages"]["xx"]["subtitles"]["adjustments"]}
    assert "merge_min_duration" in rules
    split_rules = {a["rule"] for a in report["languages"]["en"]["subtitles"]["adjustments"]}
    assert "split_max_duration" in split_rules


# --- requirement 7: parallel consistency ------------------------------------

@pytest.fixture(scope="module")
def parallel_runs(samples, tmp_path_factory):
    results = {}
    for jobs in (1, 4):
        out = tmp_path_factory.mktemp(f"out_jobs{jobs}")
        report = out / "report.json"
        proc = run_cli([
            "--video", samples["video"], "--out", str(out), "--report", str(report),
            "--jobs", str(jobs), *lang_args(samples, ["en", "ja", "de"]),
        ])
        results[jobs] = {"proc": proc, "out": out,
                         "report": json.loads(report.read_text())}
    return results


def test_parallel_consistency(parallel_runs):
    r1, r4 = parallel_runs[1], parallel_runs[4]
    assert r1["proc"].returncode == EXIT_OK
    assert r4["proc"].returncode == EXIT_OK
    assert r1["report"] == r4["report"]
    for lang in ms.LANGUAGES:
        for suffix in (".mp4", ".subtitles.json"):
            f1 = r1["out"] / f"{lang}{suffix}"
            f4 = r4["out"] / f"{lang}{suffix}"
            assert f1.read_bytes() == f4.read_bytes(), f"{lang}{suffix}"
        # temp directories are cleaned up
    for r in (r1, r4):
        assert not (r["out"] / ".tmp").exists()


# --- requirement 6: business error codes and exit codes ---------------------

@pytest.fixture(scope="module")
def error_run(samples, tmp_path_factory):
    out = tmp_path_factory.mktemp("out_errors")
    tmp = str(out)
    # missing segment: dub with only 3 of 4 segments
    miss_wav = os.path.join(tmp, "dub_miss.wav")
    from dubmux.audio import write_wav
    write_wav(miss_wav, ms.build_dub_track(
        ms.REFERENCE_SEGMENTS[:3], 1.0, 0.3, seed=5000), ms.SAMPLE_RATE)
    # sample rate mismatch: 44100 Hz
    badrate_wav = os.path.join(tmp, "dub_badrate.wav")
    write_wav(badrate_wav, ms.build_dub_track(
        ms.REFERENCE_SEGMENTS, 1.0, 0.3, seed=6000, rate=44100), 44100)
    # subtitle overflow: last entry beyond video duration
    overflow_subs = ms.make_subtitles("overflow")
    overflow_subs[-1]["end"] = ms.VIDEO_DURATION + 1.0
    overflow_path = os.path.join(tmp, "subs_overflow.json")
    with open(overflow_path, "w", encoding="utf-8") as fh:
        json.dump(overflow_subs, fh)
    en_subs = samples["subs"]["en"]
    report = out / "report.json"
    proc = run_cli([
        "--video", samples["video"], "--out", str(out), "--report", str(report),
        "--lang", f"en={samples['dubs']['en']}", "--subs", f"en={en_subs}",
        "--lang", f"miss={miss_wav}", "--subs", f"miss={en_subs}",
        "--lang", f"badrate={badrate_wav}", "--subs", f"badrate={en_subs}",
        "--lang", f"overflow={samples['dubs']['de']}",
        "--subs", f"overflow={overflow_path}",
    ])
    return {"proc": proc, "out": out, "report": json.loads(report.read_text())}


def test_error_codes_and_exit(error_run):
    proc = error_run["proc"]
    assert proc.returncode == EXIT_PARTIAL, proc.stdout
    report = error_run["report"]
    langs = report["languages"]
    assert langs["en"]["status"] == "ok"          # one failure does not block others
    assert langs["miss"]["errors"][0]["code"] == E_SEGMENT_MISSING
    assert langs["badrate"]["errors"][0]["code"] == E_SAMPLE_RATE_MISMATCH
    assert langs["overflow"]["errors"][0]["code"] == E_SUBTITLE_OVERFLOW
    assert report["summary"]["exit_code"] == EXIT_PARTIAL
    # stable codes also appear on process output
    for code in (E_SEGMENT_MISSING, E_SAMPLE_RATE_MISMATCH, E_SUBTITLE_OVERFLOW):
        assert code in proc.stdout
    # failed languages left no half-written outputs
    for lang in ("miss", "badrate", "overflow"):
        assert not (error_run["out"] / f"{lang}.mp4").exists()


# --- requirement 8: report versioning and snapshot replacement --------------

def test_report_version_and_snapshot(standard_run):
    report = standard_run["report"]
    assert report["format_version"] == REPORT_FORMAT_VERSION
    assert report["previous_format_version"] == 999
    raw = (standard_run["out"] / "report.json").read_text()
    assert "junk trailing" not in raw and "stale" not in raw
    # input digests for reproducibility
    assert len(report["inputs"]["video"]["sha256"]) == 64
    for lang in ms.LANGUAGES:
        digest = report["inputs"]["languages"][lang]
        assert len(digest["dub"]["sha256"]) == 64
        assert len(digest["subtitles"]["sha256"]) == 64
