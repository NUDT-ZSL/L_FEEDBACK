"""Top-level pipeline: validate, align, mix, normalize and report."""

import json
import os

from .beatgrid import BeatGrid
from .config import load_inputs
from .errors import ChartError, OutputNotWritableError
from .mixer import LayerMixer

OUTPUT_WAV_NAME = "mix_output.wav"
REPORT_NAME = "beat_alignment_report.json"


def plan_events(chart, grid):
    """Snap every control event to the beat grid."""
    planned = []
    for event in chart.events:
        snapped = grid.snap(event.time_ms, require_on_beat=True)
        sample_index = grid.sample_index_at_ms(snapped.aligned_ms)
        planned.append(
            {
                "original_time_ms": round(snapped.original_ms, 6),
                "aligned_time_ms": round(snapped.aligned_ms, 6),
                "beat_index": snapped.beat_index,
                "snap_deviation_ms": round(snapped.deviation_ms, 6),
                "layer": event.layer,
                "action": event.action,
                "gain": event.gain,
                "sample_index": sample_index,
            }
        )
    return planned


def build_report(chart, grid, planned_events, mixed_counts, master_gain, peak):
    return {
        "sample_rate": chart.sample_rate,
        "bpm": chart.bpm,
        "beat_period_ms": round(grid.beat_period_ms, 6),
        "total_duration_ms": round(chart.total_duration_ms, 6),
        "total_samples": grid.total_samples(chart.total_duration_ms),
        "crossfade_ms": chart.crossfade_ms,
        "loop_bars": chart.loop_bars,
        "beats_per_bar": chart.beats_per_bar,
        "master_gain": master_gain,
        "measured_peak_linear": peak,
        "event_count": len(planned_events),
        "events": planned_events,
        "layer_mixed_samples": dict(sorted(mixed_counts.items())),
    }


def _ensure_output_dir(output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise OutputNotWritableError(
            f"cannot create output directory {output_dir}: {exc}",
            detail={"path": output_dir},
        ) from exc
    probe = os.path.join(output_dir, ".write_probe")
    try:
        with open(probe, "wb") as handle:
            handle.write(b"0")
        os.remove(probe)
    except OSError as exc:
        raise OutputNotWritableError(
            f"output directory is not writable: {output_dir}",
            detail={"path": output_dir},
        ) from exc


def run(input_dir, output_dir):
    """Execute the full pipeline and return (wav_path, report_path)."""
    chart = load_inputs(input_dir)
    if chart.total_duration_ms <= 0:
        raise ChartError("chart total_duration_ms must be greater than zero")
    grid = BeatGrid(chart.bpm, chart.sample_rate)
    planned_events = plan_events(chart, grid)

    _ensure_output_dir(output_dir)
    wav_path = os.path.join(output_dir, OUTPUT_WAV_NAME)
    report_path = os.path.join(output_dir, REPORT_NAME)
    produced = []
    try:
        mixer = LayerMixer(chart, grid, planned_events, wav_path)
        peak, mixed_counts, segments, sources = mixer.measure_peak()
        master_gain = mixer.write_normalized(segments, sources, peak)
        produced.append(wav_path)
        report = build_report(
            chart, grid, planned_events, mixed_counts, master_gain, peak
        )
        try:
            with open(
                report_path, "w", encoding="utf-8", newline="\n"
            ) as handle:
                json.dump(
                    report,
                    handle,
                    indent=2,
                    sort_keys=False,
                    ensure_ascii=False,
                )
                handle.write("\n")
        except OSError as exc:
            raise OutputNotWritableError(
                f"cannot write report {report_path}: {exc}",
                detail={"path": report_path},
            ) from exc
        produced.append(report_path)
        return wav_path, report_path
    except Exception:
        for partial in produced:
            try:
                if os.path.exists(partial):
                    os.remove(partial)
            except OSError:
                pass
        raise
