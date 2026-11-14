from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import LensConfig, PipelineConfig, format_frame_name
from .errors import InputDataError

DEPTH_EPSILON = 1e-6


@dataclass(frozen=True)
class DepthFrame:
    meters: np.ndarray
    invalid_mask: np.ndarray
    clamped_mask: np.ndarray

    @property
    def invalid_count(self) -> int:
        return int(np.count_nonzero(self.invalid_mask))

    @property
    def clamped_count(self) -> int:
        return int(np.count_nonzero(self.clamped_mask))


def load_depth_raw(path: Path) -> np.ndarray:
    try:
        raw = np.load(path, allow_pickle=False)
    except OSError as exc:
        raise InputDataError(f"cannot read depth map {path}: {exc}") from exc
    except ValueError as exc:
        raise InputDataError(f"cannot decode depth map {path}: {exc}") from exc
    if not isinstance(raw, np.ndarray):
        raise InputDataError(f"depth map {path} did not decode to an array")
    if raw.dtype != np.float32:
        raise InputDataError(
            f"depth map {path} must be float32, got {raw.dtype}"
        )
    if raw.ndim != 2:
        raise InputDataError(f"depth map {path} must be 2-D, got shape {raw.shape}")
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise InputDataError(f"depth map {path} must not be empty")
    return raw


def parse_depth(raw: np.ndarray, lens: LensConfig, *, path: Path | None = None) -> DepthFrame:
    source = str(path) if path is not None else "<array>"
    finite = np.isfinite(raw)
    # Near-zero values are treated as missing metadata rather than as an
    # extremely close surface. This is the documented fallback rule.
    invalid = (~finite) | (raw <= DEPTH_EPSILON)
    clamped = finite & (raw > 1.0)

    normalized = np.where(invalid, 1.0, np.clip(raw, 0.0, 1.0)).astype(np.float64)
    depth_m = lens.near_plane_m + normalized * (
        lens.far_plane_m - lens.near_plane_m
    )
    depth_m = np.where(invalid, lens.invalid_depth_fallback_m, depth_m)
    if not np.all(np.isfinite(depth_m)) or np.any(depth_m <= 0.0):
        raise InputDataError(
            f"depth map {source} produced non-finite or non-positive physical depth"
        )
    return DepthFrame(
        meters=depth_m,
        invalid_mask=invalid,
        clamped_mask=clamped,
    )


def load_color(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    color = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if color is None:
        raise InputDataError(f"cannot read color image {path}")
    if color.shape[:2] != expected_shape:
        raise InputDataError(
            f"color image {path} has shape {color.shape[:2]}, expected {expected_shape}"
        )
    if color.ndim == 2 or (color.ndim == 3 and color.shape[2] == 1):
        color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
    elif color.ndim == 3 and color.shape[2] == 4:
        color = color[:, :, :3]
    elif color.ndim != 3 or color.shape[2] != 3:
        raise InputDataError(f"color image {path} must be grayscale, BGR, or BGRA")
    if color.dtype != np.uint8:
        raise InputDataError(
            f"color image {path} must be uint8 after decoding, got {color.dtype}"
        )
    return np.ascontiguousarray(color)


def preflight_sequence(
    config: PipelineConfig, input_dir: Path
) -> tuple[tuple[int, int], list[Path], list[Path]]:
    color_paths: list[Path] = []
    depth_paths: list[Path] = []
    image_shape: tuple[int, int] | None = None
    for sequence_index in range(config.frame_count):
        frame_index = config.frame_start + sequence_index
        color_path = input_dir / format_frame_name(
            config.color_pattern, sequence_index, frame_index
        )
        depth_path = input_dir / format_frame_name(
            config.depth_pattern, sequence_index, frame_index
        )
        if not color_path.is_file():
            raise InputDataError(f"missing color frame: {color_path}")
        if not depth_path.is_file():
            raise InputDataError(f"missing depth frame: {depth_path}")
        raw = load_depth_raw(depth_path)
        if image_shape is None:
            image_shape = raw.shape
        elif raw.shape != image_shape:
            raise InputDataError(
                f"depth frame {depth_path} shape {raw.shape} does not match "
                f"first frame shape {image_shape}"
            )
        color_paths.append(color_path)
        depth_paths.append(depth_path)
    assert image_shape is not None
    return image_shape, color_paths, depth_paths
