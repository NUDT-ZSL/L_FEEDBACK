"""Headless EGL OpenGL 3.3 verification path for Linux."""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, byref, c_int, c_uint, c_void_p

from .errors import gl_unavailable
from .gl_renderer import GLRenderResult, GLRenderer
from .runtime import BakedAsset


EGL_DEFAULT_DISPLAY = 0
EGL_SURFACE_TYPE = 0x3033
EGL_PBUFFER_BIT = 0x0001
EGL_RENDERABLE_TYPE = 0x3040
EGL_OPENGL_BIT = 0x0008
EGL_RED_SIZE = 0x3024
EGL_GREEN_SIZE = 0x3023
EGL_BLUE_SIZE = 0x3022
EGL_DEPTH_SIZE = 0x3025
EGL_NONE = 0x3038
EGL_OPENGL_API = 0x30A2
EGL_CONTEXT_MAJOR_VERSION = 0x3098
EGL_CONTEXT_MINOR_VERSION = 0x30FB
EGL_CONTEXT_OPENGL_PROFILE_MASK = 0x30FD
EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x00000001
EGL_WIDTH = 0x3057
EGL_HEIGHT = 0x3056


def _load_egl():
    try:
        egl = ctypes.CDLL("libEGL.so.1")
    except OSError as exc:
        raise gl_unavailable(f"EGL runtime is unavailable: {exc}") from exc

    def bind(name, args, restype):
        func = getattr(egl, name)
        func.argtypes = args
        func.restype = restype
        return func

    return {
        "get_display": bind("eglGetDisplay", [c_void_p], c_void_p),
        "initialize": bind(
            "eglInitialize", [c_void_p, POINTER(c_int), POINTER(c_int)], c_uint
        ),
        "choose_config": bind(
            "eglChooseConfig",
            [c_void_p, POINTER(c_int), POINTER(c_void_p), c_int, POINTER(c_int)],
            c_uint,
        ),
        "bind_api": bind("eglBindAPI", [c_uint], c_uint),
        "create_context": bind(
            "eglCreateContext", [c_void_p, c_void_p, c_void_p, POINTER(c_int)], c_void_p
        ),
        "create_pbuffer": bind(
            "eglCreatePbufferSurface", [c_void_p, c_void_p, POINTER(c_int)], c_void_p
        ),
        "make_current": bind(
            "eglMakeCurrent", [c_void_p, c_void_p, c_void_p, c_void_p], c_uint
        ),
        "destroy_context": bind("eglDestroyContext", [c_void_p, c_void_p], c_uint),
        "destroy_surface": bind("eglDestroySurface", [c_void_p, c_void_p], c_uint),
        "terminate": bind("eglTerminate", [c_void_p], c_uint),
        "get_error": bind("eglGetError", [], c_int),
    }


def render_egl_frame(
    asset: BakedAsset,
    time: float,
    width: int,
    height: int,
) -> GLRenderResult:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    egl = _load_egl()
    display = egl["get_display"](None)
    if not display:
        raise gl_unavailable("eglGetDisplay returned no display")
    major, minor = c_int(), c_int()
    if not egl["initialize"](display, byref(major), byref(minor)):
        raise gl_unavailable("eglInitialize failed")
    config_attrs = (c_int * 19)(
        EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_DEPTH_SIZE,
        24,
        EGL_NONE,
    )
    configs = (c_void_p * 8)()
    config_count = c_int()
    if not egl["choose_config"](
        display, config_attrs, configs, 8, byref(config_count)
    ) or config_count.value == 0:
        egl["terminate"](display)
        raise gl_unavailable("no EGL OpenGL pbuffer configuration is available")
    if not egl["bind_api"](EGL_OPENGL_API):
        egl["terminate"](display)
        raise gl_unavailable("eglBindAPI(EGL_OPENGL_API) failed")
    context_attrs = (c_int * 7)(
        EGL_CONTEXT_MAJOR_VERSION,
        3,
        EGL_CONTEXT_MINOR_VERSION,
        3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK,
        EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE,
    )
    context = egl["create_context"](display, configs[0], None, context_attrs)
    if not context:
        egl["terminate"](display)
        raise gl_unavailable("could not create an EGL OpenGL 3.3 core context")
    surface_attrs = (c_int * 5)(EGL_WIDTH, int(width), EGL_HEIGHT, int(height), EGL_NONE)
    surface = egl["create_pbuffer"](display, configs[0], surface_attrs)
    if not surface:
        egl["destroy_context"](display, context)
        egl["terminate"](display)
        raise gl_unavailable("could not create an EGL pbuffer surface")
    if not egl["make_current"](display, surface, surface, context):
        egl["destroy_surface"](display, surface)
        egl["destroy_context"](display, context)
        egl["terminate"](display)
        raise gl_unavailable("eglMakeCurrent failed")
    try:
        renderer = GLRenderer(asset, width, height)
        return renderer.render(time)
    finally:
        egl["make_current"](display, None, None, None)
        egl["destroy_surface"](display, surface)
        egl["destroy_context"](display, context)
        egl["terminate"](display)

