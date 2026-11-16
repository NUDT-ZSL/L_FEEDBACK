from __future__ import annotations

import hashlib
import os
import struct
import uuid
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .errors import OutputError


def _read_header_line(handle) -> bytes:
    while True:
        line = handle.readline()
        if line == b"":
            raise ValueError("unexpected end of PFM header")
        stripped = line.strip()
        if stripped and not stripped.startswith(b"#"):
            return stripped


def read_pfm(path: str) -> np.ndarray:
    try:
        with open(path, "rb") as handle:
            magic = _read_header_line(handle)
            if magic not in (b"PF", b"Pf"):
                raise ValueError("expected PF or Pf magic")
            dimensions = _read_header_line(handle).split()
            if len(dimensions) != 2:
                raise ValueError("invalid PFM dimensions")
            width, height = int(dimensions[0]), int(dimensions[1])
            scale = float(_read_header_line(handle))
            if width <= 0 or height <= 0:
                raise ValueError("invalid PFM size")
            channels = 3 if magic == b"PF" else 1
            pixel_count = width * height * channels
            dtype = np.dtype("<f4") if scale < 0.0 else np.dtype(">f4")
            data = np.frombuffer(handle.read(pixel_count * 4), dtype=dtype)
            if data.size != pixel_count:
                raise ValueError("truncated PFM pixel data")
            data = data.reshape((height, width, channels))[::-1]
            if channels == 1:
                return np.ascontiguousarray(data[:, :, 0]).astype(np.float32)
            return np.ascontiguousarray(data).astype(np.float32)
    except (OSError, ValueError) as error:
        raise OutputError("cannot read PFM %r: %s" % (path, error))


def write_pfm(path: str, image: np.ndarray) -> None:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3 and array.shape[2] == 3:
        magic = "PF"
        height, width, _ = array.shape
        payload = array[::-1]
    elif array.ndim == 2:
        magic = "Pf"
        height, width = array.shape
        payload = array[::-1, :, None]
    else:
        raise ValueError("PFM image must be HxWx3 color or HxW grayscale")
    header = ("%s\n%d %d\n-1.0\n" % (magic, width, height)).encode("ascii")
    little_endian = np.ascontiguousarray(payload, dtype="<f4")
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(little_endian.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())


def aces_tone_map(linear: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    if exposure <= 0.0:
        raise ValueError("exposure must be positive")
    value = np.maximum(np.asarray(linear, dtype=np.float64) * float(exposure), 0.0)
    numerator = value * (2.51 * value + 0.03)
    denominator = value * (2.43 * value + 0.59) + 0.14
    return np.clip(numerator / denominator, 0.0, 1.0)


def linear_to_srgb_preview(linear: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    mapped = aces_tone_map(linear, exposure)
    gamma_corrected = np.power(mapped, 1.0 / 2.2)
    return np.clip(np.rint(gamma_corrected * 255.0), 0.0, 255.0).astype(np.uint8)


def write_png(path: str, linear: np.ndarray, exposure: float = 1.0) -> None:
    pixels = linear_to_srgb_preview(linear, exposure)
    image = Image.fromarray(pixels, mode="RGB")
    image.save(path, format="PNG")


def pixel_result_hash(image: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(image, dtype="<f4"))
    digest = hashlib.sha256()
    digest.update(struct.pack("<II", array.shape[1], array.shape[0]))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def downsample(image: np.ndarray, supersample: int) -> np.ndarray:
    if supersample == 1:
        return np.ascontiguousarray(image, dtype=np.float32)
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("downsample expects an HxWx3 image")
    height, width, _ = array.shape
    if height % supersample != 0 or width % supersample != 0:
        raise ValueError("image dimensions must be divisible by supersample")
    reshaped = array.reshape(
        height // supersample,
        supersample,
        width // supersample,
        supersample,
        3,
    )
    averaged = reshaped.astype(np.float64).mean(axis=(1, 3))
    return np.ascontiguousarray(averaged, dtype=np.float32)


class AtomicOutputSet:
    """Write several files to temporary paths, then commit with rollback."""

    def __init__(self) -> None:
        self._files: List[Tuple[str, Callable[[str], None], str]] = []
        self._committed = False

    def add(self, target: str, writer: Callable[[str], None]) -> None:
        absolute = os.path.abspath(target)
        if any(existing[0] == absolute for existing in self._files):
            raise OutputError("duplicate output path %r" % target)
        self._files.append((absolute, writer, target))

    def commit(self) -> None:
        if self._committed:
            raise RuntimeError("output set was already committed")
        prepared: List[Tuple[str, str, Callable[[str], None], str]] = []
        temporary_paths: List[str] = []
        token = "%d-%s" % (os.getpid(), uuid.uuid4().hex)
        try:
            for absolute, writer, original in self._files:
                parent = os.path.dirname(absolute) or os.curdir
                try:
                    os.makedirs(parent, exist_ok=True)
                except OSError as error:
                    raise OutputError(
                        "cannot create output directory %r: %s" % (parent, error)
                    )
                if not os.path.isdir(parent):
                    raise OutputError("output parent %r is not a directory" % parent)
                if os.path.isdir(absolute):
                    raise OutputError("output path %r is a directory" % original)
                temporary = os.path.join(
                    parent, ".%s.tmp-%s" % (os.path.basename(absolute), token)
                )
                temporary_paths.append(temporary)
                try:
                    writer(temporary)
                except Exception as error:
                    raise OutputError("cannot encode output %r: %s" % (original, error))
                if not os.path.isfile(temporary):
                    raise OutputError("writer did not produce output %r" % original)
                prepared.append((absolute, temporary, writer, original))

            backups: List[Tuple[str, str]] = []
            installed: List[str] = []
            try:
                for absolute, temporary, _, original in prepared:
                    backup = None
                    if os.path.lexists(absolute):
                        backup = temporary + ".backup"
                        os.replace(absolute, backup)
                        backups.append((absolute, backup))
                    try:
                        os.replace(temporary, absolute)
                    except OSError:
                        if backup is not None:
                            os.replace(backup, absolute)
                            backups = [item for item in backups if item[0] != absolute]
                        raise
                    installed.append(absolute)
            except OSError as error:
                for absolute in installed:
                    try:
                        os.unlink(absolute)
                    except OSError:
                        pass
                for absolute, backup in reversed(backups):
                    try:
                        if os.path.exists(backup):
                            os.replace(backup, absolute)
                    except OSError:
                        pass
                raise OutputError("cannot commit output files: %s" % error)
            for _, backup in backups:
                try:
                    os.unlink(backup)
                except OSError:
                    pass
            self._committed = True
        finally:
            for temporary in temporary_paths:
                if os.path.exists(temporary):
                    try:
                        os.unlink(temporary)
                    except OSError:
                        pass
                backup = temporary + ".backup"
                if os.path.exists(backup):
                    try:
                        os.unlink(backup)
                    except OSError:
                        pass
