import json

import numpy as np
import soundfile as sf
import pytest

from dub_replace.errors import (
    ChannelLayoutError,
    CorruptAudioError,
    MissingAudioError,
    SampleRateError,
)
from dub_replace.metadata import parse_metadata
from dub_replace.processor import process_animation
from dub_replace.synth import SAMPLE_RATE, speech_like, write_corrupt_wav, write_pcm16
from dub_replace.util import ms_to_samples

from tests.conftest import (
    build_stereo_job,
    make_meta,
    stereo_channels,
    surround_channels,
    write_meta,
)


def test_end_to_end_stereo_replaces_only_voice(stereo_job):
    manifest_path, job_dir, spec = stereo_job
    meta = parse_metadata(manifest_path.parent / "anim_stereo" / "meta.json")
    output = job_dir / "anim_stereo" / "out.wav"
    report_path = job_dir / "reports" / "anim_stereo.json"
    report = process_animation(
        meta, output, report_path, work_dir=job_dir / "tmp"
    )
    data, sr = sf.read(output, dtype="float64")
    assert sr == SAMPLE_RATE
    assert data.ndim == 2 and data.shape[1] == 2
    # SFX channel is bit-identical to the original timeline.
    original, _ = sf.read(meta.original_track, dtype="float64")
    assert np.array_equal(data[:, 0], original[:, 0])
    # L1 replaced, L2 fallback (no dialect clip provided).
    methods = {s["line_id"]: s["method"] for s in report["sentences"]}
    assert methods == {"L1": "replaced", "L2": "fallback"}
    l1 = report["sentences"][0]
    assert l1["actual_start_ms"] == 100 and l1["actual_end_ms"] == 500
    assert l1["stretch_ratio"] == pytest.approx(0.4 / 0.3, rel=1e-3)
    assert isinstance(l1["peak_dbfs"], float)
    assert report["summary"]["replaced"] == 1
    assert report["summary"]["fallback"] == 1


def test_5point1_preserves_other_channels(tmp_path):
    anim = tmp_path / "surround"
    anim.mkdir(parents=True)
    n = SAMPLE_RATE
    channels = surround_channels()
    track = np.stack(
        [
            0.2 * np.sin(2 * np.pi * 100 * np.arange(n) / n) for _ in range(6)
        ],
        axis=1,
    )
    s = ms_to_samples(100, SAMPLE_RATE)
    e = ms_to_samples(700, SAMPLE_RATE)
    track[s:e, 2] = speech_like(0.6, 220, amp=0.4, seed=5)
    write_pcm16(anim / "original.wav", track)
    write_pcm16(anim / "d.wav", speech_like(0.4, 170, amp=0.4, seed=6))
    dialogue = [{"line_id": "S1", "start_ms": 100, "end_ms": 700, "text": "x"}]
    clips = [{"clip_id": "D", "line_id": "S1", "start_ms": 100, "path": "d.wav"}]
    meta_path = write_meta(anim, make_meta("surround", "5.1", channels, dialogue, clips))
    meta = parse_metadata(meta_path)
    output = tmp_path / "out" / "surround.wav"
    report = process_animation(meta, output, tmp_path / "r.json", tmp_path / "tmp")
    data, sr = sf.read(output, dtype="float64")
    assert data.shape == (n, 6)
    # Every non-target channel stays on its original timeline (output is
    # quantised to PCM_16, so compare against the quantisation step).
    for ch in (0, 1, 3, 4, 5):
        assert np.max(np.abs(data[:, ch] - track[:, ch])) <= 1.0 / 32768.0 + 1e-9
    assert not np.array_equal(data[:, 2], track[:, 2])
    assert report["input"]["channel_layout"] == "5.1"


def test_missing_dialect_falls_back_with_warning(tmp_path):
    manifest_path, job_dir, _ = build_stereo_job(tmp_path)
    anim_dir = tmp_path / "anim_stereo"
    meta_path = anim_dir / "meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["dialect_clips"][0]["path"] = "nope.wav"
    write_meta(anim_dir, payload)
    meta = parse_metadata(meta_path)
    report = process_animation(
        meta, job_dir / "o.wav", job_dir / "r.json", job_dir / "tmp"
    )
    assert report["sentences"][0]["method"] == "fallback"
    codes = [ev["code"] for ev in report["events"]]
    assert "E_MISSING_AUDIO" in codes


def test_corrupt_original_raises_corrupt_error(tmp_path):
    manifest_path, job_dir, _ = build_stereo_job(tmp_path)
    anim_dir = tmp_path / "anim_stereo"
    write_corrupt_wav(anim_dir / "original.wav")
    meta = parse_metadata(anim_dir / "meta.json")
    with pytest.raises(CorruptAudioError) as exc:
        process_animation(meta, job_dir / "o.wav", job_dir / "r.json", job_dir / "tmp")
    assert exc.value.code == "E_CORRUPT"


def test_sample_rate_mismatch_original_is_fatal(tmp_path):
    manifest_path, job_dir, _ = build_stereo_job(tmp_path)
    anim_dir = tmp_path / "anim_stereo"
    write_pcm16(anim_dir / "original.wav",
                np.zeros((1000, 2)), sr=24000)
    meta = parse_metadata(anim_dir / "meta.json")
    with pytest.raises(SampleRateError) as exc:
        process_animation(meta, job_dir / "o.wav", job_dir / "r.json", job_dir / "tmp")
    assert exc.value.code == "E_SAMPLE_RATE"


def test_channel_count_mismatch_is_fatal(tmp_path):
    manifest_path, job_dir, _ = build_stereo_job(tmp_path)
    anim_dir = tmp_path / "anim_stereo"
    write_pcm16(anim_dir / "original.wav", np.zeros((1000, 1)), sr=SAMPLE_RATE)
    meta = parse_metadata(anim_dir / "meta.json")
    with pytest.raises(ChannelLayoutError) as exc:
        process_animation(meta, job_dir / "o.wav", job_dir / "r.json", job_dir / "tmp")
    assert exc.value.code == "E_CHANNEL"


def test_stretch_clamp_event_recorded(tmp_path):
    # 2 s dialect vs 0.4 s slot and max ratio 1.5 -> E_STRETCH trim event.
    manifest_path, job_dir, _ = build_stereo_job(
        tmp_path, dialect_seconds=2.0, limits=(0.5, 1.5)
    )
    meta = parse_metadata(tmp_path / "anim_stereo" / "meta.json")
    report = process_animation(
        meta, job_dir / "o.wav", job_dir / "r.json", job_dir / "tmp"
    )
    sentence = report["sentences"][0]
    assert sentence["limit_event"] in {"trim", "pad"}
    assert any(ev["code"] == "E_STRETCH" for ev in sentence["events"])


def test_extra_clip_discarded_with_nearest(tmp_path):
    manifest_path, job_dir, _ = build_stereo_job(tmp_path)
    anim_dir = tmp_path / "anim_stereo"
    payload = json.loads((anim_dir / "meta.json").read_text(encoding="utf-8"))
    write_pcm16(anim_dir / "extra.wav", speech_like(0.2, 150, amp=0.3, seed=99))
    payload["dialect_clips"].append(
        {"clip_id": "DX", "line_id": "GHOST", "start_ms": 850, "path": "extra.wav"}
    )
    write_meta(anim_dir, payload)
    meta = parse_metadata(anim_dir / "meta.json")
    report = process_animation(
        meta, job_dir / "o.wav", job_dir / "r.json", job_dir / "tmp"
    )
    assert len(report["discarded_clips"]) == 1
    assert report["discarded_clips"][0]["nearest_line_id"] == "L2"


def test_dialect_sample_rate_mismatch_resamples_with_warning(tmp_path):
    manifest_path, job_dir, _ = build_stereo_job(tmp_path)
    anim_dir = tmp_path / "anim_stereo"
    # Replace the dialect clip with a genuine 24 kHz file.
    write_pcm16(anim_dir / "d1.wav", speech_like(0.3, 150, sr=24000, amp=0.4, seed=11), sr=24000)
    meta = parse_metadata(anim_dir / "meta.json")
    report = process_animation(
        meta, job_dir / "o.wav", job_dir / "r.json", job_dir / "tmp"
    )
    sentence = report["sentences"][0]
    assert sentence["method"] == "replaced"
    assert any(ev["code"] == "E_SAMPLE_RATE" for ev in sentence["events"])
    # Output still runs at the metadata-declared rate.
    data, sr = sf.read(job_dir / "o.wav")
    assert sr == SAMPLE_RATE
    assert data.shape[1] == 2


def test_mono_layout_output(tmp_path):
    anim = tmp_path / "mono_anim"
    anim.mkdir(parents=True)
    n = SAMPLE_RATE
    voice = np.zeros(n)
    s, e = ms_to_samples(100, SAMPLE_RATE), ms_to_samples(500, SAMPLE_RATE)
    voice[s:e] = speech_like(0.4, 200, amp=0.4, seed=3)
    write_pcm16(anim / "original.wav", voice)
    write_pcm16(anim / "d.wav", speech_like(0.3, 160, amp=0.4, seed=4))
    dialogue = [{"line_id": "M1", "start_ms": 100, "end_ms": 500, "text": "mono"}]
    channels = [{"index": 0, "label": "mono", "role": "target_voice", "language": "zh-CN"}]
    clips = [{"clip_id": "D", "line_id": "M1", "start_ms": 100, "path": "d.wav"}]
    meta_path = write_meta(anim, make_meta("mono_anim", "mono", channels, dialogue, clips))
    meta = parse_metadata(meta_path)
    output = tmp_path / "out" / "mono.wav"
    report = process_animation(meta, output, tmp_path / "r.json", tmp_path / "tmp")
    data, sr = sf.read(output, dtype="float64")
    assert data.ndim == 1 and data.shape == (n,)
    assert sr == SAMPLE_RATE
    assert report["input"]["channel_layout"] == "mono"
    assert report["sentences"][0]["method"] == "replaced"
