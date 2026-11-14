"""MRT G-Buffer with format-failure injection and a graceful fallback chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import OpenGL.GL as gl
from OpenGL import error as gl_error
from OpenGL.GL import GL_INVALID_ENUM

from .gl_helpers import FBO_STATUS_NAMES, drain_gl_errors


# Attachment name -> (default GL internal format, texture readback format/type).
# Slot index in this dict is the MRT color-attachment index and must match the
# geometry shader layout(location=...) outputs one-for-one.
ATTACHMENT_SPECS: list[dict[str, Any]] = [
    {
        "slot": 0,
        "name": "world_position",
        "default_format": "RGBA32F",
        "internalformat": gl.GL_RGBA32F,
        "pixel_format": gl.GL_RGBA,
        "pixel_type": gl.GL_FLOAT,
        "components": 4,
    },
    {
        "slot": 1,
        "name": "world_normal",
        "default_format": "RGB16F",
        "internalformat": gl.GL_RGB16F,
        "pixel_format": gl.GL_RGB,
        "pixel_type": gl.GL_FLOAT,
        "components": 3,
    },
    {
        "slot": 2,
        "name": "albedo_material",
        "default_format": "RGBA8",
        "internalformat": gl.GL_RGBA8,
        "pixel_format": gl.GL_RGBA,
        "pixel_type": gl.GL_UNSIGNED_BYTE,
        "components": 4,
    },
]

# All formats the fallback chain is allowed to request.
SUPPORTED_FORMAT_TOKENS: dict[str, int] = {
    "RGBA32F": gl.GL_RGBA32F,
    "RGB32F": gl.GL_RGB32F,
    "RGBA16F": gl.GL_RGBA16F,
    "RGB16F": gl.GL_RGB16F,
    "RGBA8": gl.GL_RGBA8,
}

# A real driver would reject truly unsupported internalformats; this synthetic
# token is mapped to a deliberately invalid enum so --force-format can trigger
# the same failure branch deterministically on every machine (incl. software GL).
SYNTHETIC_UNSUPPORTED = "UNSUPPORTED"
INVALID_INTERNALFORMAT = 0xFFFFFFFF


def resolve_format_token(token: str) -> int:
    token = token.strip().upper()
    if token == SYNTHETIC_UNSUPPORTED:
        return INVALID_INTERNALFORMAT
    if token not in SUPPORTED_FORMAT_TOKENS:
        # Unknown names are treated as a guaranteed failure too.
        return INVALID_INTERNALFORMAT
    return SUPPORTED_FORMAT_TOKENS[token]


@dataclass
class GBuffer:
    width: int
    height: int
    requested_formats: dict[str, str]
    fbo: int = 0
    depth_rbo: int = 0
    textures: dict[str, int] = field(default_factory=dict)
    active_formats: dict[str, str] = field(default_factory=dict)

    @property
    def attachment_names(self) -> list[str]:
        return [spec["name"] for spec in ATTACHMENT_SPECS]

    def _create_color_texture(self, spec: dict[str, Any], token: str) -> tuple[int, list[dict]]:
        drain_gl_errors()
        internal = resolve_format_token(token)
        tex = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        try:
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, internal,
                self.width, self.height, 0,
                spec["pixel_format"], spec["pixel_type"], None,
            )
            errors = drain_gl_errors()
        except gl_error.GLError as exc:
            # PyOpenGL may raise during argument conversion before we can poll;
            # still report the authoritative queued GL error via glGetError.
            errors = drain_gl_errors()
            if not errors:
                code = int(getattr(exc, "err", 0)) or GL_INVALID_ENUM
                from .gl_helpers import GL_ERROR_NAMES
                errors = [{"code": code,
                           "name": GL_ERROR_NAMES.get(code, "GL_UNKNOWN")}]
        return tex, errors

    def build(self) -> list[dict[str, Any]]:
        """Build the FBO, falling back per attachment when a format is invalid.

        Returns a list of structured warning records (empty on a clean build).
        Raises RuntimeError only when even the safe default format fails.
        """
        warnings: list[dict[str, Any]] = []

        fbo = int(gl.glGenFramebuffers(1))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        created_textures: dict[str, int] = {}
        chosen_formats: dict[str, str] = {}

        for spec in ATTACHMENT_SPECS:
            name = spec["name"]
            requested = self.requested_formats[name]
            tex, errors = self._create_color_texture(spec, requested)
            used_token = requested

            if errors:
                gl.glDeleteTextures(1, [tex])
                fallback_token = spec["default_format"]
                warnings.append({
                    "code": "RENDER_FBO_INCOMPLETE",
                    "attachment": name,
                    "slot": spec["slot"],
                    "requested_format": requested,
                    "fallback_format": fallback_token,
                    "stage": "glTexImage2D",
                    "gl_errors": errors,
                    "message": (
                        f"Requested format '{requested}' for attachment '{name}' "
                        f"was rejected; falling back to '{fallback_token}'."
                    ),
                })
                tex, second_errors = self._create_color_texture(spec, fallback_token)
                if second_errors:
                    gl.glDeleteTextures(1, [tex])
                    raise RuntimeError(
                        f"Default format {fallback_token} for {name} also failed: "
                        f"{second_errors}"
                    )
                used_token = fallback_token

            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER,
                gl.GL_COLOR_ATTACHMENT0 + spec["slot"],
                gl.GL_TEXTURE_2D, tex, 0,
            )
            created_textures[name] = tex
            chosen_formats[name] = used_token

        draw_buffers = [
            gl.GL_COLOR_ATTACHMENT0 + spec["slot"] for spec in ATTACHMENT_SPECS
        ]
        gl.glDrawBuffers(len(draw_buffers), draw_buffers)

        depth_rbo = int(gl.glGenRenderbuffers(1))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rbo)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, self.width, self.height
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rbo
        )

        drain_gl_errors()
        status = int(gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER))
        status_errors = drain_gl_errors()
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            incomplete = {
                "code": "RENDER_FBO_INCOMPLETE",
                "attachment": "framebuffer",
                "requested_formats": dict(self.requested_formats),
                "status": status,
                "status_name": FBO_STATUS_NAMES.get(status, "GL_UNKNOWN"),
                "gl_errors": status_errors,
            }
            self._delete_objects(fbo, depth_rbo, list(created_textures.values()))
            # Retry once using every safe default format.
            retry = GBuffer(
                self.width,
                self.height,
                {spec["name"]: spec["default_format"] for spec in ATTACHMENT_SPECS},
            )
            retry_warnings = retry.build()
            warnings.append({
                **incomplete,
                "fallback_format": "all-defaults",
                "message": "Framebuffer incomplete; rebuilt with default formats.",
            })
            warnings.extend(retry_warnings)
            self._adopt(retry)
            return warnings

        self.fbo = fbo
        self.depth_rbo = depth_rbo
        self.textures = created_textures
        self.active_formats = chosen_formats
        return warnings

    def _adopt(self, other: "GBuffer") -> None:
        self.fbo = other.fbo
        self.depth_rbo = other.depth_rbo
        self.textures = other.textures
        self.active_formats = other.active_formats

    @staticmethod
    def _delete_objects(fbo: int, depth_rbo: int, textures: list[int]) -> None:
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        if textures:
            gl.glDeleteTextures(len(textures), textures)
        gl.glDeleteRenderbuffers(1, [depth_rbo])
        gl.glDeleteFramebuffers(1, [fbo])

    def destroy(self) -> None:
        if self.fbo:
            self._delete_objects(
                self.fbo, self.depth_rbo, list(self.textures.values())
            )
        self.fbo = 0
        self.depth_rbo = 0
        self.textures = {}
        self.active_formats = {}

    def bind(self) -> None:
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, self.width, self.height)
        draw_buffers = [
            gl.GL_COLOR_ATTACHMENT0 + spec["slot"] for spec in ATTACHMENT_SPECS
        ]
        gl.glDrawBuffers(len(draw_buffers), draw_buffers)

    def bind_for_position_readback(self) -> None:
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.fbo)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)

    def read_position_pixels(self, count: int = 10) -> np.ndarray:
        """Real GPU readback of `count` position-attachment pixels.

        Reads a horizontal strip through the screen center rather than a corner
        so the sample intersects moving geometry; it is still a contiguous
        block of exactly `count` pixels fetched from the GPU each frame.
        """
        self.bind_for_position_readback()
        buf = np.zeros((count, 4), dtype=np.float32)
        x0 = max((self.width - count) // 2, 0)
        y0 = self.height // 2
        gl.glReadPixels(x0, y0, count, 1, gl.GL_RGBA, gl.GL_FLOAT, buf)
        return buf

    def bind_color_textures(self) -> None:
        for spec in ATTACHMENT_SPECS:
            gl.glActiveTexture(gl.GL_TEXTURE0 + spec["slot"])
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[spec["name"]])


def default_requested_formats() -> dict[str, str]:
    return {spec["name"]: spec["default_format"] for spec in ATTACHMENT_SPECS}
