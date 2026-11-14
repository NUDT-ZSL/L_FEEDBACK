"""End-to-end material baking pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, TextIO

import numpy as np

from . import __version__
from .config import BakeConfig, load_config
from .errors import ToolError, invalid_config, io_error, missing_file
from .lighting import evaluate_material
from .pngio import PNGImage, read_png_rgb, write_rgb_png
from .resample import decode_normal_map, normalize_vectors, resize_image


INPUT_NAMES = {
    "albedo": "albedo.png",
    "roughness": "roughness.png",
    "metallic": "metallic.png",
    "normal": "normal.png",
}


def _resolve_directory(
    cli_value: str | Path | None,
    config_value: Path | None,
    field_name: str,
) -> Path:
    value = cli_value if cli_value is not None else config_value
    if value is None:
        raise invalid_config(
            f"{field_name} is required via the command line or configuration file"
        )
    return Path(value)


def _load_inputs(input_dir: Path) -> dict[str, PNGImage]:
    if not input_dir.exists():
        raise missing_file(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise missing_file(f"Input path is not a directory: {input_dir}")

    paths = {name: input_dir / filename for name, filename in INPUT_NAMES.items()}
    for name, path in paths.items():
        if not path.exists() or not path.is_file():
            raise missing_file(f"Required input file not found: {path} ({name})")

    images = {name: read_png_rgb(path) for name, path in paths.items()}
    reference = images["albedo"]
    reference_size = (reference.width, reference.height)
    for name, image in images.items():
        if (image.width, image.height) != reference_size:
            raise invalid_format_dimensions(name, image, reference_size)
    return images


def invalid_format_dimensions(
    name: str, image: PNGImage, expected: tuple[int, int]
) -> ToolError:
    from .errors import invalid_format

    return invalid_format(
        f"{INPUT_NAMES[name]} is {image.width}x{image.height}; "
        f"all maps must match albedo.png ({expected[0]}x{expected[1]})"
    )


def _resample_material(
    images: dict[str, PNGImage],
    config: BakeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    normal_vectors, zero_count = decode_normal_map(
        images["normal"].rgb, flip_green=config.flip_green
    )

    albedo = resize_image(images["albedo"].rgb, config.resolution, config.resolution)
    roughness = resize_image(
        images["roughness"].rgb[:, :, 0], config.resolution, config.resolution
    )
    metallic = resize_image(
        images["metallic"].rgb[:, :, 0], config.resolution, config.resolution
    )
    normal_vectors = resize_image(
        normal_vectors, config.resolution, config.resolution
    )
    normal_vectors, resampled_zero_count = normalize_vectors(normal_vectors)

    return (
        np.clip(albedo, 0.0, 1.0),
        np.clip(roughness, 0.0, 1.0),
        np.clip(metallic, 0.0, 1.0),
        normal_vectors,
        zero_count + resampled_zero_count,
    )


def _json_vector(values: tuple[float, float, float] | list[float]) -> list[float]:
    return [float(value) for value in values]


def _metadata(
    config: BakeConfig,
    images: dict[str, PNGImage],
    zero_normal_count: int,
    output_dir: Path,
) -> dict[str, Any]:
    directional = {
        "direction": _json_vector(config.directional.direction),
        "color": _json_vector(config.directional.color),
        "intensity": config.directional.intensity,
    }
    hemisphere = {
        "sky_color": _json_vector(config.hemisphere.sky_color),
        "ground_color": _json_vector(config.hemisphere.ground_color),
        "intensity": config.hemisphere.intensity,
    }
    images_metadata = {
        name: {
            "file": filename,
            "width": config.resolution,
            "height": config.resolution,
            "bit_depth": config.output_bits,
            "gamma": config.output_gamma,
        }
        for name, filename in config.output_names.items()
    }
    return {
        "schema_version": 1,
        "tool": {"name": "material-baker", "version": __version__},
        "lighting": {
            "directional_light": directional,
            "hemisphere_light": hemisphere,
            "view_direction": _json_vector(config.view_direction),
        },
        "directional_light": directional,
        "hemisphere_light": hemisphere,
        "flip_green": config.flip_green,
        "output_resolution": config.resolution,
        "output_bits": config.output_bits,
        "output_gamma": config.output_gamma,
        "bit_depth": config.output_bits,
        "gamma": config.output_gamma,
        "filenames": config.output_names,
        "resolution": {
            "width": config.resolution,
            "height": config.resolution,
        },
        "output": {
            "directory": str(output_dir.resolve()),
            "width": config.resolution,
            "height": config.resolution,
            "bit_depth": config.output_bits,
            "gamma": config.output_gamma,
            "encoding": "linear_clamped_pow_1_over_gamma",
            "images": config.output_names,
        },
        "images": images_metadata,
        "inputs": {
            name: {
                "file": INPUT_NAMES[name],
                "width": image.width,
                "height": image.height,
                "bit_depth": image.bit_depth,
            }
            for name, image in images.items()
        },
        "model": {
            "direct": "Cook-Torrance GGX/Smith/Schlick",
            "roughness_mapping": "alpha = max(roughness^2, alpha_epsilon)",
            "alpha_epsilon": config.alpha_epsilon,
            "dielectric_f0": 0.04,
            "energy_safety_cap": "direct RGB <= intensity*color*NdotL",
            "environment": "hemisphere gradient analytic approximation",
        },
        "warnings": {"zero_length_normals": int(zero_normal_count)},
    }


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        temp_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    except OSError as exc:
        raise io_error(f"Cannot write metadata file {path}: {exc}") from exc


def run_bake(
    config_path: str | Path,
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    stderr: TextIO | None = None,
) -> dict[str, Any]:
    stderr = stderr or sys.stderr
    config = load_config(config_path)
    resolved_input = _resolve_directory(input_dir, config.input_dir, "input_dir")
    resolved_output = _resolve_directory(output_dir, config.output_dir, "output_dir")

    images = _load_inputs(resolved_input)
    albedo, roughness, metallic, normal, zero_count = _resample_material(
        images, config
    )
    if zero_count:
        print(
            "WARNING: zero-length normal(s) detected after decoding/resampling; "
            f"replaced {zero_count} texel(s) with (0, 0, 1)",
            file=stderr,
        )

    result = evaluate_material(
        albedo,
        roughness,
        metallic,
        normal,
        config.view_direction,
        config.directional,
        config.hemisphere,
        alpha_epsilon=config.alpha_epsilon,
    )

    try:
        resolved_output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise io_error(f"Cannot create output directory {resolved_output}: {exc}") from exc

    images_to_write = {
        "irradiance": result.irradiance,
        "specular": result.specular,
        "final_color": result.final_color,
    }
    output_files: dict[str, str] = {}
    for key, image in images_to_write.items():
        if not np.all(np.isfinite(image)):
            raise ToolError("E_INTERNAL", f"Non-finite value in {key} image")
        filename = config.output_names[key]
        write_rgb_png(
            resolved_output / filename,
            image,
            config.output_bits,
            config.output_gamma,
        )
        output_files[key] = filename

    metadata = _metadata(config, images, zero_count, resolved_output)
    metadata_path = resolved_output / "metadata.json"
    _write_metadata(metadata_path, metadata)

    return {
        "status": "ok",
        "output_dir": str(resolved_output.resolve()),
        "metadata": str(metadata_path.resolve()),
        "resolution": config.resolution,
        "bit_depth": config.output_bits,
        "gamma": config.output_gamma,
        "files": output_files,
        "zero_length_normals": zero_count,
    }

