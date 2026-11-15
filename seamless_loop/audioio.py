"""Sample-accurate time conversion and WAV input validation/loading."""

import os

import numpy as np
import soundfile as sf

from .errors import AudioFileError, ParameterRangeError

SUPPORTED_SAMPLE_RATES = (44100, 48000)
SUPPORTED_SUBTYPES = ("PCM_16", "PCM_24", "FLOAT")
SUPPORTED_CHANNELS = (1, 2)


def to_sample_index(time_seconds, sample_rate):
    """Convert a time in seconds to an integer sample index.

    Uses half-away-from-zero rounding so that ``x.5`` always rounds up, which is
    deterministic and independent of the host platform's banker's rounding.
    """
    if not isinstance(time_seconds, (int, float)) or isinstance(
        time_seconds, bool
    ):
        raise ParameterRangeError(
            "time value must be a number, got %r" % (time_seconds,)
        )
    return int(np.floor(float(time_seconds) * sample_rate + 0.5))


def load_and_validate_track(path, start_sample, end_sample, expected_sr=None):
    """Read a PCM WAV track and validate its format against the spec.

    Parameters
    ----------
    path:
        Path to the input WAV file.
    start_sample, end_sample:
        Half-open integer sample interval ``[start_sample, end_sample)`` of
        the loop region.
    expected_sr:
        Sample rate every track must share.  When ``None`` the first track
        establishes the reference rate.

    Returns
    -------
    dict
        ``region`` (float64 ndarray), ``sample_rate``, ``channels``,
        ``subtype``, ``total_frames`` and the source ``path``.
    """
    if not os.path.isfile(path):
        raise AudioFileError("input audio file not found: %s" % path)

    try:
        info = sf.info(path)
    except Exception as exc:  # soundfile raises RuntimeError on bad files.
        raise AudioFileError(
            "cannot read audio file %s: %s" % (path, exc)
        ) from exc

    if info.format != "WAV":
        raise AudioFileError(
            "unsupported container for %s: expected WAV, got %s"
            % (path, info.format)
        )
    if info.samplerate not in SUPPORTED_SAMPLE_RATES:
        raise AudioFileError(
            "unsupported sample rate %d Hz in %s; expected 44100 or 48000"
            % (info.samplerate, path)
        )
    if expected_sr is not None and info.samplerate != expected_sr:
        raise AudioFileError(
            "sample rate mismatch in %s: track is %d Hz but the project uses "
            "%d Hz; all tracks must share one sample rate"
            % (path, info.samplerate, expected_sr)
        )
    if info.subtype not in SUPPORTED_SUBTYPES:
        raise AudioFileError(
            "unsupported bit depth/subtype %s in %s; expected PCM_16, "
            "PCM_24 or FLOAT (32-bit float)" % (info.subtype, path)
        )
    if info.channels not in SUPPORTED_CHANNELS:
        raise AudioFileError(
            "unsupported channel count %d in %s; only mono or stereo are "
            "supported" % (info.channels, path)
        )
    if end_sample > info.frames:
        raise ParameterRangeError(
            "loop end sample %d exceeds duration of %s (%d samples)"
            % (end_sample, path, info.frames)
        )
    if start_sample < 0:
        raise ParameterRangeError(
            "loop start sample must be >= 0, got %d" % start_sample
        )

    try:
        data, sr = sf.read(path, dtype="float64", always_2d=False)
    except Exception as exc:
        raise AudioFileError(
            "failed to decode %s: %s" % (path, exc)
        ) from exc

    region = data[start_sample:end_sample]
    if region.shape[0] != (end_sample - start_sample):
        raise ParameterRangeError(
            "could not slice full loop region from %s: requested %d samples, "
            "got %d"
            % (path, end_sample - start_sample, region.shape[0])
        )

    return {
        "path": path,
        "region": region,
        "sample_rate": sr,
        "channels": info.channels,
        "subtype": info.subtype,
        "total_frames": info.frames,
    }
