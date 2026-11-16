"""End-to-end build pipeline and incremental rebuild logic."""

from __future__ import annotations

import json
import os

import numpy as np

from .assembly import assemble_sections, crossfade_frames
from .dsp import INTERNAL_SAMPLE_RATE
from .errors import BuildError, ExitCode
from .hashing import (
    hash_chart_tree,
    hash_source_files,
    sha256_array,
    sha256_file,
    sha256_json,
    stable_json_bytes,
)
from .locking import OutputLock
from .loudness import loudness_gain
from .markers import build_markers
from .renderer import (
    LayerSpec,
    load_sources,
    parse_layers,
    parse_notes,
    render_section,
)
from .timing import load_chart
from .waveio import write_wav

MANIFEST_NAME = "manifest.json"
CACHE_DIR_NAME = "cache"
STEMS_DIR_NAME = "stems"
MARKERS_NAME = "markers.json"
REPORT_NAME = "normalization.json"


def _atomic_write_bytes(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _atomic_write_json(path: str, payload: dict) -> None:
    _atomic_write_bytes(path, json.dumps(payload, ensure_ascii=False,
                                         indent=2, sort_keys=True).encode("utf-8")
                        + b"\n")


def _save_npy(path: str, array: np.ndarray) -> None:
    _atomic_write_bytes(path, _npy_bytes(array))


def _npy_bytes(array: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _load_npy(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _load_manifest(output_dir: str) -> dict | None:
    path = os.path.join(output_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def _section_cache_key(chart: dict, source_hashes: dict[str, str],
                       section: dict, time_map, layers: list,
                       notes: list[dict]) -> str:
    # Only inputs that actually change this section's rendered samples enter
    # the key.  Editing another section therefore keeps this cache hit.
    start_beat = float(section["start_beat"])
    end_beat = float(section["end_beat"])
    section_notes = [note for note in notes
                     if start_beat <= note["beat"] < end_beat]
    bpm_in_range = [point for point in chart["bpm_points"]
                    if start_beat <= float(point["beat"]) <= end_beat]
    layer_inputs = [
        {
            "id": layer.layer_id,
            "enabled": layer.enabled,
            "gain_db": layer.gain_db,
            "onset_beat": layer.onset_beat,
            "lead_in_seconds": layer.lead_in_seconds,
            "native_bpm_layer": layer.native_bpm,
            "source_hash": source_hashes.get(os.path.basename(layer.path)),
        }
        for layer in layers
    ]
    payload = {
        "source_hashes": source_hashes,
        "section": section,
        "bpm_in_range": bpm_in_range,
        "crossfade_beats": chart.get("crossfade_beats", 0.0),
        "internal_sample_rate": chart.get("internal_sample_rate", 48000),
        "layers": layer_inputs,
        "notes": section_notes,
        "pipeline_version": 1,
    }
    return sha256_json(payload)[:16]


def _validate_layers(chart: dict, layers: list[LayerSpec]) -> None:
    if not any(layer.enabled for layer in layers):
        raise BuildError(ExitCode.CHART_SYNTAX,
                         "at least one layer must be enabled for the master mix")
    target = float(chart.get("target_lufs", -23.0))
    peak = float(chart.get("true_peak_dbfs", -1.0))
    if not np.isfinite(target) or not np.isfinite(peak) or peak >= 0.0:
        raise BuildError(ExitCode.CHART_SYNTAX,
                         "target_lufs must be finite and true_peak_dbfs must be < 0")


def build(chart_path: str, samples_dir: str, output_dir: str,
          force: bool = False) -> dict:
    """Run a full or incremental build; return the summary manifest."""
    os.makedirs(output_dir, exist_ok=True)
    with OutputLock(output_dir):
        return _build_locked(chart_path, samples_dir, output_dir, force)


def _build_locked(chart_path: str, samples_dir: str, output_dir: str,
                  force: bool) -> dict:
    chart, time_map, signatures, sections = load_chart(chart_path)
    layers = parse_layers(chart, samples_dir)
    _validate_layers(chart, layers)
    notes = parse_notes(chart, {layer.layer_id for layer in layers})
    source_hashes = hash_source_files([layer.path for layer in layers])
    chart_hash = hash_chart_tree(chart)

    cache_dir = os.path.join(output_dir, CACHE_DIR_NAME)
    stems_dir = os.path.join(output_dir, STEMS_DIR_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(stems_dir, exist_ok=True)

    previous = None if force else _load_manifest(output_dir)
    sources = load_sources(layers, time_map.sample_rate)

    layer_order = [layer.layer_id for layer in layers]
    enabled_layers = [layer.layer_id for layer in layers if layer.enabled]
    fade_frames = crossfade_frames(
        time_map, float(chart.get("crossfade_beats", 0.0)), time_map.sample_rate)

    rebuilt_sections: list[str] = []
    reused_sections: list[str] = []
    section_stems: list[dict[str, np.ndarray]] = []
    section_cache_records: list[dict] = []

    for index, section in enumerate(sections):
        section_payload = chart["sections"][index]
        cache_key = _section_cache_key(chart, source_hashes, section_payload,
                                       time_map, layers, notes)
        cache_path = os.path.join(cache_dir, f"section_{index:03d}_{cache_key}.npy")
        record = {"section": section.section_id, "index": index,
                  "cache_key": cache_key, "cache_file": os.path.basename(cache_path)}
        previous_records = previous.get("sections", []) if previous else []
        prior = next((item for item in previous_records
                      if item.get("index") == index), None)

        if prior and prior.get("cache_key") == cache_key and \
                os.path.isfile(os.path.join(cache_dir, prior["cache_file"])):
            stacked = _load_npy(os.path.join(cache_dir, prior["cache_file"]))
            stems = {layer_order[channel]: stacked[channel]
                     for channel in range(stacked.shape[0])}
            cache_path = os.path.join(cache_dir, prior["cache_file"])
            record["cache_file"] = prior["cache_file"]
            reused_sections.append(section.section_id)
        else:
            stems = render_section(section, time_map, sources, notes,
                                   time_map.sample_rate)
            stacked = np.stack([stems[layer_id] for layer_id in layer_order], axis=0)
            _save_npy(cache_path, stacked)
            rebuilt_sections.append(section.section_id)
        section_stems.append(stems)
        section_cache_records.append(record)

    # Garbage-collect stale cache files that no current record references.
    live_files = {record["cache_file"] for record in section_cache_records}
    for name in os.listdir(cache_dir):
        if name.endswith(".npy") and name not in live_files:
            os.remove(os.path.join(cache_dir, name))

    full_stems = assemble_sections(section_stems, layer_order, fade_frames)
    master = np.zeros_like(full_stems[0])
    for channel, layer in enumerate(layer_order):
        if layer in enabled_layers:
            master += full_stems[channel]

    target_lufs = float(chart["target_lufs"])
    true_peak = float(chart["true_peak_dbfs"])
    normalization = loudness_gain(master, time_map.sample_rate,
                                  target_lufs, true_peak)
    gain = normalization.applied_gain_linear
    master_normalized = (master * gain).astype(np.float32)

    output_records: list[dict] = []
    master_path = os.path.join(output_dir, "master.wav")
    write_wav(master_path, master_normalized, time_map.sample_rate, 3)
    output_records.append({"role": "master", "path": os.path.relpath(master_path, output_dir),
                           "sha256": sha256_file(master_path),
                           "array_sha256": sha256_array(master_normalized)})

    # Stems use the identical gain so their sum reproduces the master.
    for channel, layer in enumerate(layer_order):
        stem_normalized = (full_stems[channel] * gain).astype(np.float32)
        stem_path = os.path.join(stems_dir, f"{layer}.wav")
        write_wav(stem_path, stem_normalized, time_map.sample_rate, 3)
        output_records.append({
            "role": "stem",
            "layer": layer,
            "enabled": layer in enabled_layers,
            "path": os.path.relpath(stem_path, output_dir),
            "sha256": sha256_file(stem_path),
            "array_sha256": sha256_array(stem_normalized),
        })

    markers = build_markers(time_map, sections, signatures)
    marker_payload = {
        "sample_rate": time_map.sample_rate,
        "total_frames": int(master_normalized.shape[0]),
        "markers": markers,
    }
    markers_path = os.path.join(output_dir, MARKERS_NAME)
    _atomic_write_json(markers_path, marker_payload)

    report_path = os.path.join(output_dir, REPORT_NAME)
    report = normalization.as_dict()
    report.update({
        "gain_applied_to_master_and_stems": gain,
        "sample_rate": time_map.sample_rate,
        "frames": int(master_normalized.shape[0]),
        "enabled_layers": enabled_layers,
        "all_layers": layer_order,
    })
    _atomic_write_json(report_path, report)

    manifest = {
        "schema_version": 1,
        "chart_path": os.path.abspath(chart_path),
        "samples_dir": os.path.abspath(samples_dir),
        "chart_hash": chart_hash,
        "source_hashes": source_hashes,
        "sections": section_cache_records,
        "outputs": output_records,
        "markers_file": MARKERS_NAME,
        "markers_sha256": sha256_file(markers_path),
        "normalization_file": REPORT_NAME,
        "crossfade_frames": fade_frames,
        "total_frames": int(master_normalized.shape[0]),
        "sample_rate": time_map.sample_rate,
    }
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    _atomic_write_json(manifest_path, manifest)
    # The on-disk manifest is content-addressed and bit-stable across no-op
    # rebuilds; the per-run action summary is only returned, not persisted.
    summary = dict(manifest)
    summary["rebuilt_sections"] = rebuilt_sections
    summary["reused_sections"] = reused_sections
    return summary
