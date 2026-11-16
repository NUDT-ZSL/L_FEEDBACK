from __future__ import annotations

import hashlib
from pathlib import Path

from cutscene_dub.model import load_timeline
from cutscene_dub.samples import generate_samples

from .helpers import dialogue, timeline_payload, write_json


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sample_generation_is_byte_deterministic(tmp_path: Path) -> None:
    payload = timeline_payload(
        [
            dialogue("dlg_001", 0.5, 2.5, source_duration_s=2.8),
            dialogue("dlg_002", 2.8, 3.8, source_duration_s=1.3),
        ],
        duration_s=5.0,
    )
    timeline_path = tmp_path / "timeline.json"
    write_json(timeline_path, payload)
    timeline = load_timeline(timeline_path)

    first = tmp_path / "samples_a"
    second = tmp_path / "samples_b"
    summary_a = generate_samples(timeline, audio_root=first, seed=189)
    summary_b = generate_samples(timeline, audio_root=second, seed=189)

    assert summary_a["ok"] and summary_b["ok"]
    expected = sorted(
        str(path.relative_to(first))
        for path in first.rglob("*.wav")
    )
    actual = sorted(
        str(path.relative_to(second))
        for path in second.rglob("*.wav")
    )
    assert actual == expected
    for relative in expected:
        assert _sha(first / relative) == _sha(second / relative)

    # Same seed and same timeline also reuse complete artifacts deterministically.
    summary_reuse = generate_samples(timeline, audio_root=first, seed=189)
    assert summary_reuse["reused"] is True
    for record in summary_reuse["files"]:
        path = Path(record["path"])
        assert _sha(path) == record["sha256"]


def test_sample_wav_format_and_seed_sensitivity(tmp_path: Path) -> None:
    payload = timeline_payload(
        [dialogue("dlg_001", 0.5, 2.0)],
        duration_s=3.0,
    )
    timeline_path = tmp_path / "timeline.json"
    write_json(timeline_path, payload)
    timeline = load_timeline(timeline_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_samples(timeline, audio_root=first, seed=1)
    generate_samples(timeline, audio_root=second, seed=2)

    import wave

    with wave.open(str(first / "en-US" / "dlg_001.wav"), "rb") as handle:
        assert handle.getframerate() == 48_000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getcomptype() == "NONE"
    assert _sha(first / "en-US" / "dlg_001.wav") != _sha(
        second / "en-US" / "dlg_001.wav"
    )
    assert _sha(first / "bgm.wav") != _sha(second / "bgm.wav")


def test_generated_voice_duration_varies_by_target_language(tmp_path: Path) -> None:
    payload = timeline_payload(
        [dialogue("dlg_001", 0.5, 2.0)],
        duration_s=3.0,
    )
    en_path = tmp_path / "en.json"
    de_payload = dict(payload)
    de_payload["target_language"] = "de-DE"
    de_path = tmp_path / "de.json"
    write_json(en_path, payload)
    write_json(de_path, de_payload)
    en_timeline = load_timeline(en_path)
    de_timeline = load_timeline(de_path)

    en_root = tmp_path / "en_samples"
    de_root = tmp_path / "de_samples"
    generate_samples(en_timeline, audio_root=en_root, seed=189)
    generate_samples(de_timeline, audio_root=de_root, seed=189)

    import wave

    with wave.open(str(en_root / "en-US" / "dlg_001.wav"), "rb") as handle:
        en_frames = handle.getnframes()
    with wave.open(str(de_root / "de-DE" / "dlg_001.wav"), "rb") as handle:
        de_frames = handle.getnframes()
    assert en_frames != de_frames
    assert _sha(en_root / "en-US" / "dlg_001.wav") != _sha(
        de_root / "de-DE" / "dlg_001.wav"
    )

