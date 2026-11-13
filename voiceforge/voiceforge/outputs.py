"""Final deterministic artifacts: WAV track, float mix, timeline and summary."""

import json
import os
import struct

from . import OUTPUT_SAMPLE_RATE
from .audio import floats_to_int16, write_wav
from .hashing import sha256_file
from .lock import atomic_write_text

WAV_NAME = "voice_track.wav"
FLOAT_NAME = "mix_float.f32"
TIMELINE_NAME = "timeline.json"
LOUDNESS_NAME = "loudness_report.json"
SUMMARY_NAME = "summary.json"


def _dump_json(path: str, payload) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    atomic_write_text(path, text)
    return text


def _write_float32(path: str, values) -> None:
    packed = b"".join(struct.pack("<f", float(value)) for value in values)
    with open(path, "wb") as handle:
        handle.write(packed)


def write_outputs(result: dict, manifest: dict, input_dir: str,
                  output_dir: str, output_rate: int = OUTPUT_SAMPLE_RATE) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    accumulator = result["accumulator"]
    clip_mask = result["clip_mask"]
    total_samples = result["total_samples"]
    missing_players = sorted(result["missing_players"])

    wav_path = os.path.join(output_dir, WAV_NAME)
    float_path = os.path.join(output_dir, FLOAT_NAME)
    timeline_path = os.path.join(output_dir, TIMELINE_NAME)
    loudness_path = os.path.join(output_dir, LOUDNESS_NAME)
    summary_path = os.path.join(output_dir, SUMMARY_NAME)

    int_samples = floats_to_int16(accumulator)
    write_wav(wav_path, output_rate, int_samples)
    _write_float32(float_path, accumulator)

    timeline_payload = {
        "schema_version": 1,
        "output_sample_rate": output_rate,
        "total_samples": total_samples,
        "event_count": len(result["events"]),
        "events": result["events"],
    }
    _dump_json(timeline_path, timeline_payload)

    loudness_payload = {
        interval_id: result["loudness_map"][interval_id]
        for interval_id in sorted(result["loudness_map"])
    }
    _dump_json(loudness_path, loudness_payload)

    input_digests = []
    for player in sorted(manifest.get("players", []),
                         key=lambda item: item["player_id"]):
        for shard in sorted(player.get("segments", []),
                            key=lambda item: item["segment_id"]):
            file_path = os.path.join(input_dir, shard["file"])
            if os.path.isfile(file_path):
                digest = sha256_file(file_path)
                state = "present"
            else:
                digest = None
                state = "missing"
            input_digests.append({
                "player_id": player["player_id"],
                "segment_id": shard["segment_id"],
                "file": shard["file"],
                "state": state,
                "sha256": digest,
            })

    missing_interval_count = sum(
        1 for interval in manifest["intervals"]
        if interval["player_id"] in result["missing_players"])
    summary = {
        "schema_version": 1,
        "output_sample_rate": output_rate,
        "total_samples": total_samples,
        "interval_count": len(manifest["intervals"]),
        "event_count": len(result["events"]),
        "clipped_samples": int(sum(clip_mask)),
        "missing_source_players": missing_players,
        "missing_source_player_count": len(missing_players),
        "missing_source_interval_count": missing_interval_count,
        "inputs": input_digests,
        "outputs": {
            WAV_NAME: sha256_file(wav_path),
            FLOAT_NAME: sha256_file(float_path),
            TIMELINE_NAME: sha256_file(timeline_path),
            LOUDNESS_NAME: sha256_file(loudness_path),
        },
    }
    _dump_json(summary_path, summary)
    return summary

