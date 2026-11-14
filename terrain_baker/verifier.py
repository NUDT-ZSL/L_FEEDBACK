"""Independent verification of a baked output set.

Re-reads every binary buffer using only the offsets/counts in the manifest
and recomputes SHA-256 hashes, so a render pipeline can trust the metadata
without parsing the geometry itself.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .checksums import overall_sha256, sha256_file
from .lod import INDEX_DTYPE, VERTEX_DTYPE


def verify_output(target_dir: str, manifest_path: str | None = None) -> dict:
    manifest_path = manifest_path or os.path.join(target_dir, "metadata.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    errors: list[str] = []
    per_file: dict[str, str] = {}
    for relative in manifest["files"]:
        absolute = os.path.join(target_dir, relative)
        if not os.path.exists(absolute):
            errors.append(f"missing output file: {relative}")
            continue
        per_file[relative] = sha256_file(absolute)

    for relative, expected in manifest["sha256"]["per_file"].items():
        if per_file.get(relative) != expected:
            errors.append(f"sha256 mismatch for {relative}")
    if overall_sha256(per_file) != manifest["sha256"]["overall"]:
        errors.append("overall sha256 mismatch")

    for level in manifest["lod_levels"]:
        binary = os.path.join(target_dir, level["binary_path"])
        if not os.path.exists(binary):
            continue
        size = os.path.getsize(binary)
        expected_size = (
            level["vertex_offset_bytes"]
            + level["vertex_count"] * level["vertex_stride_bytes"]
            + level["index_count"] * INDEX_DTYPE.itemsize
        )
        if size != expected_size:
            errors.append(
                f"lod{level['level']}: file size {size} != expected {expected_size}"
            )
        if level["index_offset_bytes"] != (
            level["vertex_offset_bytes"]
            + level["vertex_count"] * level["vertex_stride_bytes"]
        ):
            errors.append(f"lod{level['level']}: index offset mismatch")
        vertices = np.memmap(
            binary,
            dtype=VERTEX_DTYPE,
            mode="r",
            offset=level["vertex_offset_bytes"],
            shape=level["vertex_count"],
        )
        indices = np.memmap(
            binary,
            dtype=INDEX_DTYPE,
            mode="r",
            offset=level["index_offset_bytes"],
            shape=level["index_count"],
        )
        if int(indices.max(initial=0)) >= level["vertex_count"]:
            errors.append(f"lod{level['level']}: index out of vertex range")
        uv = np.asarray(vertices["uv"])
        if uv.min() < 0.0 or uv.max() > 1.0:
            errors.append(f"lod{level['level']}: UV outside [0, 1]")
        if level["triangle_count"] * 3 != level["index_count"]:
            errors.append(f"lod{level['level']}: triangle/index count mismatch")

    return {
        "ok": not errors,
        "errors": errors,
        "files_checked": len(per_file),
        "lod_checked": len(manifest["lod_levels"]),
    }
