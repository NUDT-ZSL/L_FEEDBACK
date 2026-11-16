from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cutscene_dub.adapt import adapt_dialogues
from cutscene_dub.config import RenderConfig
from cutscene_dub.errors import BusinessError
from cutscene_dub.model import load_timeline
from cutscene_dub.render import render_project

from .helpers import (
    dialogue,
    read_wav,
    silence,
    timeline_payload,
    tone,
    write_json,
    write_mono_wav,
)


def _prepare_timeline(
    tmp_path: Path,
    *,
    start_s: float,
    end_s: float,
    duration_s: float = 4.0,
) -> tuple[Path, Path]:
    timeline_path = tmp_path / "timeline.json"
    write_json(
        timeline_path,
        timeline_payload(
            [dialogue("dlg_001", start_s, end_s)],
            duration_s=duration_s,
        ),
    )
    audio_root = tmp_path / "samples"
    write_mono_wav(
        audio_root / "bgm.wav",
        tone(duration_s, frequency=110.0, amplitude=0.18),
    )
    return timeline_path, audio_root


def test_silence_is_trimmed_before_stretching(tmp_path: Path) -> None:
    timeline_path, audio_root = _prepare_timeline(
        tmp_path, start_s=0.5, end_s=1.5
    )
    source = np.concatenate(
        [silence(0.2), tone(0.8, amplitude=0.5), silence(0.5)]
    )
    write_mono_wav(audio_root / "en-US" / "dlg_001.wav", source)
    timeline = load_timeline(timeline_path)
    config = RenderConfig(max_speedup=1.2)

    analyses = adapt_dialogues(timeline, audio_root=audio_root, config=config)

    assert len(analyses) == 1
    item = analyses[0]
    assert item.unadaptable is False
    assert item.trim.total_silence_frames > 0
    assert item.trim.trimmed_frames <= item.window_frames
    assert item.applied_ratio == 1.0
    report = item.as_report()
    assert report["original"]["duration_s"] == pytest.approx(1.5)
    assert report["silence_trim"]["total_s"] > 0.6
    assert report["final"]["start_s"] == pytest.approx(0.5)


def test_time_stretch_reports_actual_ratio_and_exact_window(tmp_path: Path) -> None:
    timeline_path, audio_root = _prepare_timeline(
        tmp_path, start_s=0.5, end_s=1.4
    )
    source = np.concatenate(
        [silence(0.05), tone(0.95, amplitude=0.5), silence(0.05)]
    )
    write_mono_wav(audio_root / "en-US" / "dlg_001.wav", source)
    timeline = load_timeline(timeline_path)
    config = RenderConfig(max_speedup=1.2)

    analyses = adapt_dialogues(timeline, audio_root=audio_root, config=config)

    item = analyses[0]
    assert item.unadaptable is False
    assert item.requested_ratio > 1.0
    assert item.applied_ratio == pytest.approx(item.requested_ratio)
    assert item.final_frames == item.window_frames
    assert item.as_report()["stretch"]["filter"] == "atempo"


def test_fully_silent_overlong_clip_trims_to_zero_without_nan_json(
    tmp_path: Path,
) -> None:
    timeline_path, audio_root = _prepare_timeline(
        tmp_path, start_s=0.5, end_s=1.5
    )
    write_mono_wav(audio_root / "en-US" / "dlg_001.wav", silence(2.0))
    timeline = load_timeline(timeline_path)

    item = adapt_dialogues(
        timeline, audio_root=audio_root, config=RenderConfig()
    )[0]

    assert item.unadaptable is False
    assert item.trim.fully_silent is True
    assert item.final_frames == 0
    json.dumps(item.as_report(), allow_nan=False)


def test_unadaptable_dialogue_fails_without_publishing_audio(tmp_path: Path) -> None:
    timeline_path, audio_root = _prepare_timeline(
        tmp_path, start_s=0.5, end_s=1.4
    )
    source = np.concatenate(
        [silence(0.03), tone(1.30, amplitude=0.5), silence(0.03)]
    )
    write_mono_wav(audio_root / "en-US" / "dlg_001.wav", source)
    timeline = load_timeline(timeline_path)
    output = tmp_path / "build" / "cutscene.wav"
    report_path = tmp_path / "build" / "report.json"

    with pytest.raises(BusinessError) as captured:
        render_project(
            timeline,
            audio_root=audio_root,
            bgm_path=audio_root / "bgm.wav",
            output_path=output,
            report_path=report_path,
            config=RenderConfig(max_speedup=1.15),
        )

    assert captured.value.code == "E_DIALOGUE_UNFIT"
    assert captured.value.entry_id == "dlg_001"
    assert not output.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["alignment"]["unadaptable_count"] == 1
    assert report["alignment"]["dialogues"][0]["unadaptable"] is True
    assert report["errors"][0]["code"] == "E_DIALOGUE_UNFIT"


def _generate_project(tmp_path: Path) -> tuple[Path, Path]:
    payload = timeline_payload(
        [
            dialogue("dlg_001", 0.5, 1.8, source_duration_s=1.55),
            dialogue("dlg_002", 2.1, 3.3, source_duration_s=1.35),
        ],
        duration_s=4.0,
    )
    timeline_path = tmp_path / "timeline.json"
    write_json(timeline_path, payload)
    from cutscene_dub.samples import generate_samples

    timeline = load_timeline(timeline_path)
    audio_root = tmp_path / "samples"
    generate_samples(timeline, audio_root=audio_root, seed=77)
    return timeline_path, audio_root


def test_stereo_mix_duration_report_and_peak_limit(tmp_path: Path) -> None:
    timeline_path, audio_root = _generate_project(tmp_path)
    timeline = load_timeline(timeline_path)
    output = tmp_path / "build" / "cutscene.wav"
    report_path = tmp_path / "build" / "alignment_report.json"
    config = RenderConfig(
        duck_depth_db=16.0,
        dialogue_gain_db=20.0,
        peak_ceiling_dbfs=-1.0,
    )

    summary = render_project(
        timeline,
        audio_root=audio_root,
        bgm_path=audio_root / "bgm.wav",
        output_path=output,
        report_path=report_path,
        config=config,
    )

    assert summary["ok"] is True
    pcm, rate, channels, width = read_wav(output)
    assert rate == 48_000
    assert channels == 2
    assert width == 2
    assert pcm.shape == (round(timeline.duration_s * rate), 2)
    peak_db = 20.0 * np.log10(np.max(np.abs(pcm.astype(np.float64))) / 32768.0)
    assert peak_db <= -1.0 + 1e-9
    assert np.array_equal(pcm[:, 0], pcm[:, 1])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["output"]["timeline_duration_error_frames"] == 0
    assert report["output"]["channels"] == 2
    assert report["output"]["bit_depth"] == 16
    assert report["mix"]["global_attenuation_db"] < 0
    assert report["ducking"]["measured_dialogue_attenuation_db"] == pytest.approx(-16.0)
    assert report["ducking"]["depth_db"] == -16.0
    assert report["mix"]["dialogue_to_music_ratio_db"] > 0
    assert len(report["ducking"]["intervals"]) == 2
    assert report["alignment"]["unadaptable_count"] == 0
    fields = {
        "original",
        "silence_trim",
        "stretch",
        "final",
        "unadaptable",
        "status",
    }
    for item in report["alignment"]["dialogues"]:
        assert fields.issubset(item.keys())
        assert item["silence_trim"]["total_s"] >= 0
        assert item["stretch"]["applied_ratio"] is not None
        assert item["final"]["within_subtitle"] is True


def test_rendering_is_idempotent_and_repairs_corrupt_output(tmp_path: Path) -> None:
    timeline_path, audio_root = _generate_project(tmp_path)
    timeline = load_timeline(timeline_path)
    output = tmp_path / "build" / "cutscene.wav"
    report_path = tmp_path / "build" / "alignment_report.json"

    first = render_project(
        timeline,
        audio_root=audio_root,
        bgm_path=audio_root / "bgm.wav",
        output_path=output,
        report_path=report_path,
        config=RenderConfig(),
    )
    second = render_project(
        timeline,
        audio_root=audio_root,
        bgm_path=audio_root / "bgm.wav",
        output_path=output,
        report_path=report_path,
        config=RenderConfig(),
    )
    assert first["reused"] is False
    assert second["reused"] is True
    assert first["audio_sha256"] == second["audio_sha256"]

    output.write_bytes(b"corrupt")
    repaired = render_project(
        timeline,
        audio_root=audio_root,
        bgm_path=audio_root / "bgm.wav",
        output_path=output,
        report_path=report_path,
        config=RenderConfig(),
    )
    assert repaired["reused"] is False
    pcm, rate, channels, width = read_wav(output)
    assert rate == 48_000 and channels == 2 and width == 2
    assert pcm.shape[0] == round(timeline.duration_s * rate)

