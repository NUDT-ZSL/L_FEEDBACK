"""Structured error codes for the terrain baker."""

from __future__ import annotations


class BakeError(Exception):
    """An expected, user-facing baking failure."""

    def __init__(self, code: str, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code

    def to_json_dict(self) -> dict:
        return {"error": self.code, "message": self.message}


def scene_missing(path: str) -> BakeError:
    return BakeError("E_SCENE_MISSING", f"scene file not found: {path}", 2)


def scene_invalid(detail: str) -> BakeError:
    return BakeError("E_SCENE_INVALID", f"invalid scene: {detail}", 3)


def unknown_material(material_id: str, layer_index: int) -> BakeError:
    return BakeError(
        "E_UNKNOWN_MATERIAL",
        f"layer {layer_index} references unregistered material id "
        f"{material_id!r}",
        4,
    )


def resolution_limit(width: int, height: int, max_pixels: int) -> BakeError:
    return BakeError(
        "E_RESOLUTION_LIMIT",
        f"resolution {width}x{height} exceeds memory budget "
        f"({width * height} > {max_pixels} pixels)",
        5,
    )


def target_exists(path: str) -> BakeError:
    return BakeError(
        "E_TARGET_EXISTS",
        f"output directory {path!r} exists and was not created by this tool",
        6,
    )


def path_unsafe(detail: str) -> BakeError:
    return BakeError("E_PATH_UNSAFE", f"unsafe output path: {detail}", 7)


def cache_invalid(detail: str) -> BakeError:
    return BakeError("E_CACHE_INVALID", f"cache validation failed: {detail}", 8)
