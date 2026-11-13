"""Synthesis engine: gated mixing, clipping, loudness and resumability."""

import array
import json
import os

from . import OUTPUT_SAMPLE_RATE
from .audio import int16_to_float, is_clipped, read_wav
from .checkpoint import CheckpointStore, MISSING_DIGEST
from .hashing import sha256_text
from .interrupt import simulated_interrupt_interval
from .resampling import ms_to_samples, render_segment
from .timeline import build_timeline


def canonical_manifest_text(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def load_manifest(manifest_path: str):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest, canonical_manifest_text(manifest)


def _shard_by_segment(manifest: dict) -> dict:
    mapping = {}
    for player in manifest.get("players", []):
        for shard in player.get("segments", []):
            mapping[shard["segment_id"]] = shard
    return mapping


def compute_total_samples(manifest: dict,
                          output_rate: int = OUTPUT_SAMPLE_RATE) -> int:
    end_ms = 0.0
    for player in manifest.get("players", []):
        for shard in player.get("segments", []):
            shard_end = shard["record_offset_ms"] + shard["declared_duration_ms"]
            end_ms = max(end_ms, shard_end)
    for interval in manifest.get("intervals", []):
        end_ms = max(end_ms, interval["end_ms"])
    return max(1, ms_to_samples(end_ms, output_rate) + 1)


def _loudness(contribution, new_clips: int) -> dict:
    peak = 0.0
    square_sum = 0.0
    for value in contribution:
        magnitude = abs(value)
        if magnitude > peak:
            peak = magnitude
        square_sum += value * value
    rms = (square_sum / len(contribution)) ** 0.5 if contribution else 0.0
    return {"peak": peak, "rms": rms, "new_clipped_samples": new_clips}


def synthesize(manifest: dict, manifest_text: str, input_dir: str,
               output_dir: str, interrupt_controller=None,
               output_rate: int = OUTPUT_SAMPLE_RATE):
    """Run (or resume) synthesis. Returns a result dictionary."""
    intervals = sorted(manifest["intervals"],
                       key=lambda item: (item["start_ms"], item["interval_id"]))
    for interval in intervals:
        interval["_start_sample"] = ms_to_samples(interval["start_ms"], output_rate)
        interval["_end_sample"] = ms_to_samples(interval["end_ms"], output_rate)

    total_samples = compute_total_samples(manifest, output_rate)
    shard_by_segment = _shard_by_segment(manifest)
    missing_players = set(manifest.get("missing_players", []))
    order_by_id = {interval["interval_id"]: index
                   for index, interval in enumerate(intervals)}

    def digest_for(interval):
        if interval["player_id"] in missing_players:
            return MISSING_DIGEST
        shard = shard_by_segment.get(interval["segment_id"])
        if shard is None:
            return MISSING_DIGEST
        path = os.path.join(input_dir, shard["file"])
        if not os.path.isfile(path):
            return MISSING_DIGEST
        from .hashing import sha256_file
        return sha256_file(path)

    store = CheckpointStore(output_dir)
    completed, accumulator, clip_mask, loudness_map = store.load_or_reset(
        manifest_text, intervals, digest_for, total_samples)

    processed_now = 0
    interrupted = False
    interrupt_after = simulated_interrupt_interval()

    for order, interval in enumerate(intervals):
        interval_id = interval["interval_id"]
        if interval_id in completed:
            continue

        player_id = interval["player_id"]
        shard = shard_by_segment.get(interval["segment_id"])
        is_missing = (player_id in missing_players or shard is None
                      or not os.path.isfile(
                          os.path.join(input_dir, shard["file"])))
        if is_missing:
            digest = MISSING_DIGEST
            start_sample = interval["_start_sample"]
            contribution = array.array("d")
            loudness = {"peak": 0.0, "rms": 0.0,
                        "new_clipped_samples": 0}
        else:
            digest = digest_for(interval)
            wav = read_wav(os.path.join(input_dir, shard["file"]))
            source_floats = int16_to_float(wav.samples)
            start_sample, rendered = render_segment(
                source_floats, wav.sample_rate,
                shard["record_offset_ms"], shard["declared_duration_ms"],
                output_rate)
            threshold = interval["threshold"]
            gain = interval["gain"]
            contribution = array.array("d")
            new_clips = 0
            for value in rendered:
                gated = value if abs(value) >= threshold else 0.0
                gained = gated * gain
                contribution.append(gained)
            for offset, value in enumerate(contribution):
                output_index = start_sample + offset
                accumulator[output_index] += value
                if is_clipped(accumulator[output_index]) and not clip_mask[output_index]:
                    clip_mask[output_index] = 1
                    new_clips += 1
            loudness = _loudness(contribution, new_clips)

        loudness_map[interval_id] = loudness
        store.write_contribution(interval_id, player_id, start_sample,
                                 digest, contribution, loudness)
        store.write_accumulator(accumulator)
        completed.add(interval_id)
        store.write_state(sha256_text(manifest_text),
                          sorted(completed, key=lambda ident: order_by_id[ident]),
                          total_samples)
        processed_now += 1

        if interrupt_controller is not None and interrupt_controller.requested:
            interrupted = True
            break
        if interrupt_after and processed_now >= interrupt_after:
            interrupted = True
            break

    if interrupted:
        return {
            "interrupted": True,
            "completed": completed,
            "total_samples": total_samples,
            "intervals": intervals,
            "accumulator": accumulator,
            "clip_mask": clip_mask,
            "loudness_map": loudness_map,
            "missing_players": missing_players,
        }

    events = build_timeline(intervals, missing_players, loudness_map,
                            total_samples, output_rate)
    return {
        "interrupted": False,
        "completed": completed,
        "total_samples": total_samples,
        "intervals": intervals,
        "accumulator": accumulator,
        "clip_mask": clip_mask,
        "loudness_map": loudness_map,
        "missing_players": missing_players,
        "events": events,
    }

