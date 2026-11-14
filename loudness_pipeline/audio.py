"""WAV input/output with deterministic format preservation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .errors import PipelineError


SUPPORTED_SUBTYPES = frozenset(
    {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}
)
SUPPORTED_FORMATS = frozenset({"WAV", "WAVEX"})
SUPPORTED_SAMPLE_RATES = frozenset(
    {32000, 44100, 48000, 88200, 96000, 176400, 192000}
)


@dataclass
class AudioFile:
    samples: np.ndarray
    sample_rate: int
    channels: int
    frames: int
    subtype: str
    format: str
    path: Path


def _soundfile_error_text(error: Exception) -> str:
    text = str(error).strip()
    return text if text else error.__class__.__name__


def read_wav(path: Path) -> AudioFile:
    """Read a supported WAV PCM/float file as normalized float64 samples."""

    source = Path(path)
    try:
        if not source.exists():
            raise PipelineError(
                "E_FORMAT", "input file does not exist: {0}".format(source)
            )
        if not source.is_file():
            raise PipelineError(
                "E_FORMAT",
                "input path is not a regular file: {0}".format(source),
            )
        info = sf.info(str(source))
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(
            "E_FORMAT",
            "cannot inspect WAV file '{0}': {1}".format(
                source, _soundfile_error_text(error)
            ),
        ) from error

    if info.frames == 0:
        raise PipelineError("E_EMPTY", "input WAV contains zero audio frames")
    if info.format not in SUPPORTED_FORMATS:
        raise PipelineError(
            "E_FORMAT",
            "unsupported container format '{0}'; expected WAV or WAVEX".format(
                info.format
            ),
        )
    if info.subtype not in SUPPORTED_SUBTYPES:
        raise PipelineError(
            "E_FORMAT",
            "unsupported WAV subtype '{0}'; expected one of: {1}".format(
                info.subtype, ", ".join(sorted(SUPPORTED_SUBTYPES))
            ),
        )
    if info.samplerate not in SUPPORTED_SAMPLE_RATES:
        raise PipelineError(
            "E_SAMPLERATE",
            "unsupported sample rate {0} Hz; expected one of: {1}".format(
                info.samplerate,
                ", ".join(str(rate) for rate in sorted(SUPPORTED_SAMPLE_RATES)),
            ),
        )
    if info.channels <= 0:
        raise PipelineError("E_FORMAT", "WAV header reports no audio channels")

    try:
        samples, sample_rate = sf.read(
            str(source),
            dtype="float64",
            always_2d=True,
            frames=info.frames,
        )
    except Exception as error:
        raise PipelineError(
            "E_FORMAT",
            "cannot read WAV samples from '{0}': {1}".format(
                source, _soundfile_error_text(error)
            ),
        ) from error

    if samples.shape[0] == 0 or samples.size == 0:
        raise PipelineError("E_EMPTY", "input WAV contains zero audio frames")
    if samples.shape[1] != info.channels:
        raise PipelineError(
            "E_FORMAT",
            "decoded channel count does not match the WAV header",
        )
    if not np.all(np.isfinite(samples)):
        raise PipelineError(
            "E_FORMAT",
            "input WAV contains NaN or infinite floating-point samples",
        )

    return AudioFile(
        samples=np.asarray(samples, dtype=np.float64),
        sample_rate=int(sample_rate),
        channels=int(samples.shape[1]),
        frames=int(samples.shape[0]),
        subtype=str(info.subtype),
        format=str(info.format),
        path=source,
    )


def same_file(first: Path, second: Path) -> bool:
    """Return whether two paths identify the same file entry."""

    first_path = Path(first)
    second_path = Path(second)
    try:
        if first_path.exists() and second_path.exists():
            return os.path.samefile(str(first_path), str(second_path))
    except OSError:
        pass
    return first_path.resolve() == second_path.resolve()


def _temporary_path(destination: Path) -> Path:
    destination = Path(destination)
    parent = destination.parent
    if not parent.exists():
        raise PipelineError(
            "E_WRITE",
            "output directory does not exist: {0}".format(parent),
        )
    if not parent.is_dir():
        raise PipelineError(
            "E_WRITE",
            "output parent is not a directory: {0}".format(parent),
        )
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".{0}.".format(destination.name),
            suffix=".tmp",
            dir=str(parent),
        )
        os.close(descriptor)
    except OSError as error:
        raise PipelineError(
            "E_WRITE",
            "cannot create temporary output in '{0}': {1}".format(parent, error),
        ) from error
    return Path(temporary_name)


def write_wav_atomic(
    destination: Path,
    samples: np.ndarray,
    sample_rate: int,
    subtype: str,
    format_name: str,
) -> None:
    """Write then atomically replace ``destination``, cleaning on failure."""

    target = Path(destination)
    temporary = _temporary_path(target)
    try:
        audio = np.asarray(samples, dtype=np.float64)
        if audio.ndim != 2:
            raise ValueError("samples must be a two-dimensional array")
        if audio.shape[0] == 0:
            raise PipelineError("E_EMPTY", "refusing to write an empty WAV file")
        if not np.all(np.isfinite(audio)):
            raise PipelineError(
                "E_WRITE", "refusing to write NaN or infinite samples"
            )
        audio = np.clip(audio, -1.0, 1.0)
        sf.write(
            str(temporary),
            audio,
            int(sample_rate),
            subtype=subtype,
            format=format_name,
        )
        written = sf.info(str(temporary))
        if (
            written.frames != audio.shape[0]
            or written.channels != audio.shape[1]
            or written.samplerate != sample_rate
            or written.subtype != subtype
        ):
            raise PipelineError(
                "E_WRITE",
                "written WAV metadata does not match the requested output",
            )
        os.replace(str(temporary), str(target))
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(
            "E_WRITE",
            "failed to write output WAV '{0}': {1}".format(
                target, _soundfile_error_text(error)
            ),
        ) from error
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass


def write_text_atomic(destination: Path, content: str) -> None:
    """Write UTF-8 text atomically and remove the temporary file on failure."""

    target = Path(destination)
    temporary = _temporary_path(target)
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(
            "E_WRITE",
            "failed to write report '{0}': {1}".format(target, error),
        ) from error
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass

