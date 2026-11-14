"""Generate deterministic sample projects without external assets."""

import copy


def build_sample_project(output_wav="replay.wav", manifest="manifest.json", work_dir="render_work"):
    return {
        "sample_rate": 16000,
        "channel_layout": "stereo",
        "tick_rate": 1000,
        "channels": [
            {"id": "squad", "gain_db": 0.0, "muted": False},
            {"id": "command", "gain_db": 6.0, "muted": False},
            {"id": "recon", "gain_db": -3.0, "muted": True},
        ],
        "speakers": [
            {
                "id": "alpha",
                "seed": 101,
                "base_frequency": 130.0,
                "harmonic_ratio": 0.35,
                "pitch_semitones": -2.0,
                "source_duration_ms": 800.0,
                "bound_channel": "squad",
                "pan": -0.4,
            },
            {
                "id": "bravo",
                "seed": 202,
                "base_frequency": 165.0,
                "harmonic_ratio": 0.2,
                "pitch_semitones": 3.0,
                "source_duration_ms": 700.0,
                "bound_channel": "command",
                "pan": 0.35,
            },
            {
                "id": "charlie",
                "seed": 303,
                "base_frequency": 110.0,
                "harmonic_ratio": 0.45,
                "pitch_semitones": 0.0,
                "source_duration_ms": 900.0,
                "bound_channel": "recon",
                "pan": 0.0,
            },
        ],
        "events": [
            {"tick": 100, "type": "start", "speaker": "alpha"},
            {"tick": 150, "type": "ptt", "speaker": "alpha"},
            {"tick": 350, "type": "ptt", "speaker": "bravo"},
            {"tick": 400, "type": "start", "speaker": "bravo"},
            {"tick": 700, "type": "switch", "speaker": "alpha", "channel": "command"},
            {"tick": 900, "type": "end", "speaker": "alpha"},
            {"tick": 950, "type": "block", "speaker": "bravo"},
            {"tick": 1200, "type": "unblock", "speaker": "bravo"},
            {"tick": 1350, "type": "end", "speaker": "bravo"},
            {"tick": 1500, "type": "start", "speaker": "charlie"},
            {"tick": 1900, "type": "end", "speaker": "charlie"},
            {"tick": 2000, "type": "start", "speaker": "alpha"},
            {"tick": 2400, "type": "end", "speaker": "alpha"},
        ],
        "export": {
            "output_wav": output_wav,
            "manifest": manifest,
            "work_dir": work_dir,
            "chunk_samples": 4000,
            "intro_ms": 300.0,
            "outro_ms": 400.0,
            "enable_prompts": True,
        },
    }


def build_clipping_project(output_wav="clipping.wav", manifest="clipping_manifest.json", work_dir="clipping_work"):
    project = build_sample_project(output_wav, manifest, work_dir)
    project = copy.deepcopy(project)
    for channel in project["channels"]:
        channel["gain_db"] = 18.0
        channel["muted"] = False
    project["export"]["chunk_samples"] = 2000
    return project


def build_mono_project(output_wav="mono.wav", manifest="mono_manifest.json", work_dir="mono_work"):
    project = copy.deepcopy(build_sample_project(output_wav, manifest, work_dir))
    project["channel_layout"] = "mono"
    return project
