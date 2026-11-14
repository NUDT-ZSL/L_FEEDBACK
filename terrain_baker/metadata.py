"""Loadable JSON manifest assembly."""

from __future__ import annotations

import json
import os

from .checksums import overall_sha256, sha256_file


def build_manifest(
    *,
    spec_dict: dict,
    lod_infos: list,
    normal_path: str,
    heightfield_path: str,
    tile_stats: list,
    memory_snapshot: dict,
    normal_quality: dict,
    consistency: list[dict],
    preview: dict,
    output_files: list[str],
    target_dir: str,
    tile_size: int,
    memory_limit_bytes: int,
) -> dict:
    """Assemble the manifest and compute per-file + overall SHA-256 hashes."""
    per_file: dict[str, str] = {}
    for relative in output_files:
        absolute = os.path.join(target_dir, relative)
        per_file[relative] = sha256_file(absolute)

    manifest = {
        "schema_version": 1,
        "inputs": spec_dict,
        "tile_size": int(tile_size),
        "memory": {
            "limit_bytes": int(memory_limit_bytes),
            **memory_snapshot,
        },
        "heightfield": {
            "path": heightfield_path,
            "dtype": "float32",
            "endianness": "little",
        },
        "normal_map": {
            "path": normal_path,
            "format": "png",
            "color_space": "sRGB",
            "encoding": "rgb8",
            "tangent_space": {"x": "u", "y": "v_down", "z": "viewer"},
            "quality": normal_quality,
        },
        "lod_levels": [info.to_dict() for info in lod_infos],
        "consistency_checks": consistency,
        "tile_stats": [stat.to_dict() for stat in tile_stats],
        "preview": preview,
        "files": output_files,
        "sha256": {
            "per_file": per_file,
            "overall": overall_sha256(per_file),
        },
    }
    return manifest


def write_manifest(path: str, manifest: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
