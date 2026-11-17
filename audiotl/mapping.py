"""Timeline <-> sample-position mapping.

The mapping is exact to a single output frame even after resampling and
channel conversion, because positions are computed in output frames from
integer frame counts (never from rounded milliseconds).
"""

from .dsp import resample_length

MAPPING_VERSION = 1


def compute_positions(out_frames_list, fade_out_frames_list):
    """Return (starts, total_frames) for segments laid out sequentially.

    Segment i+1 starts at start[i] + frames[i] - fade_out_frames[i]
    (the crossfade overlap with the previous segment).
    """
    starts = []
    cursor = 0
    for index, frames in enumerate(out_frames_list):
        starts.append(cursor)
        overlap = fade_out_frames_list[index] if index + 1 < len(out_frames_list) else 0
        cursor += frames - overlap
    total = starts[-1] + out_frames_list[-1] if out_frames_list else 0
    return starts, total


def build_mapping(manifest, analyzed, starts, total_frames):
    segments = []
    count = len(analyzed)
    for index, info in enumerate(analyzed):
        overlap_next = info["fade_out_frames"] if index + 1 < count else 0
        segments.append(
            {
                "index": index,
                "id": info["id"],
                "output_start_frame": starts[index],
                "output_end_frame": starts[index] + info["out_frames"],
                "valid_frames": info["out_frames"],
                "input_start_frame": 0,
                "input_frames": info["in_frames"],
                "input_rate": info["in_rate"],
                "input_channels": info["in_channels"],
                "fade_in_frames": info["fade_in_frames"],
                "fade_out_frames": info["fade_out_frames"],
                "overlap_with_next_frames": overlap_next,
            }
        )
    return {
        "version": MAPPING_VERSION,
        "sample_rate": manifest.sample_rate,
        "channels": manifest.channels,
        "total_frames": total_frames,
        "total_ms": total_frames * 1000.0 / manifest.sample_rate,
        "segments": segments,
    }


def segments_at_ms(mapping, ms):
    """Return the segments active at output time `ms` (1 or 2 in a crossfade)."""
    rate = mapping["sample_rate"]
    frame = int(ms * rate / 1000.0)
    return segments_at_frame(mapping, frame)


def segments_at_frame(mapping, frame):
    return [
        s
        for s in mapping["segments"]
        if s["output_start_frame"] <= frame < s["output_end_frame"]
    ]


def output_frame_to_input(segment, output_rate, out_frame):
    """Map an output frame inside a segment to its source (input) frame."""
    local = out_frame - segment["output_start_frame"]
    in_pos = local * segment["input_rate"] / output_rate
    return segment["input_start_frame"] + int(in_pos)


def input_frame_to_output(segment, output_rate, in_frame):
    """Map a source frame to the output frame it lands on (single-frame exact)."""
    local_in = in_frame - segment["input_start_frame"]
    out_len = resample_length(
        segment["input_frames"], segment["input_rate"], output_rate
    )
    local_out = min(
        round(local_in * output_rate / segment["input_rate"]),
        out_len - 1,
    )
    return segment["output_start_frame"] + int(local_out)
