from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from typing import Any

from .errors import ContextError


class OffscreenGLContext:
    """A minimal hidden OpenGL 3.3 core context.

    Windows uses a hidden HWND plus WGL. Linux uses EGL surfaceless (or a
    1x1 pbuffer fallback). No window manager or external GLFW dependency is
    required.
    """

    def __init__(self) -> None:
        self._backend: Any = None
        self._active = False

    def __enter__(self) -> OffscreenGLContext:
        self.initialize()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def initialize(self) -> None:
        if self._active:
            return
        try:
            if sys.platform == "win32":
                self._backend = _WGLBackend()
            elif sys.platform.startswith("linux"):
                self._backend = _EGLBackend()
            else:
                raise ContextError("E_CONTEXT_PLATFORM", f"unsupported platform {sys.platform}")
            self._backend.initialize()
            self._active = True
        except Exception:
            if self._backend is not None:
                self._backend.close()
                self._backend = None
            self._active = False
            raise

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
        self._backend = None
        self._active = False


class _EGLBackend:
    EGL_NO_DISPLAY = ctypes.c_void_p(0)
    EGL_NO_CONTEXT = ctypes.c_void_p(0)
    EGL_NO_SURFACE = ctypes.c_void_p(0)
    EGL_DEFAULT_DISPLAY = ctypes.c_void_p(0)
    EGL_OPENGL_API = 0x30A2
    EGL_OPENGL_BIT = 0x0008
    EGL_SURFACE_TYPE = 0x3033
    EGL_PBUFFER_BIT = 0x0001
    EGL_RENDERABLE_TYPE = 0x3040
    EGL_RED_SIZE = 0x3024
    EGL_GREEN_SIZE = 0x3023
    EGL_BLUE_SIZE = 0x3022
    EGL_ALPHA_SIZE = 0x3021
    EGL_NONE = 0x3038
    EGL_PLATFORM_SURFACELESS_MESA = 0x31DD
    EGL_CONTEXT_MAJOR_VERSION = 0x3098
    EGL_CONTEXT_MINOR_VERSION = 0x30FB
    EGL_CONTEXT_OPENGL_PROFILE_MASK = 0x30FD
    EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x00000001
    EGL_WIDTH = 0x3057
    EGL_HEIGHT = 0x3056

    ERROR_NAMES = {
        0x3000: "EGL_SUCCESS",
        0x3001: "EGL_NOT_INITIALIZED",
        0x3002: "EGL_BAD_ACCESS",
        0x3003: "EGL_BAD_ALLOC",
        0x3004: "EGL_BAD_ATTRIBUTE",
        0x3005: "EGL_BAD_CONFIG",
        0x3006: "EGL_BAD_CONTEXT",
        0x3007: "EGL_BAD_CURRENT_SURFACE",
        0x3008: "EGL_BAD_DISPLAY",
        0x3009: "EGL_BAD_MATCH",
        0x300A: "EGL_BAD_NATIVE_PIXMAP",
        0x300B: "EGL_BAD_NATIVE_WINDOW",
        0x300C: "EGL_BAD_PARAMETER",
        0x300D: "EGL_BAD_SURFACE",
        0x300E: "EGL_CONTEXT_LOST",
    }

    def __init__(self) -> None:
        self.lib: Any = None
        self.display = self.EGL_NO_DISPLAY
        self.context = self.EGL_NO_CONTEXT
        self.surface = self.EGL_NO_SURFACE
        self.initialized = False

    def _load(self) -> Any:
        library_name = ctypes.util.find_library("EGL")
        names = [name for name in (library_name, "libEGL.so.1", "libEGL.so") if name]
        for name in names:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        raise ContextError("E_CONTEXT_LOAD", "cannot load libEGL")

    def _check(self, result: Any, operation: str) -> Any:
        if not result:
            error = self.lib.eglGetError()
            name = self.ERROR_NAMES.get(error, hex(error))
            raise ContextError("E_CONTEXT_EGL", f"{operation} failed", name)
        return result

    def initialize(self) -> None:
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        self.lib = self._load()
        lib = self.lib
        lib.eglGetError.restype = ctypes.c_uint
        lib.eglGetDisplay.argtypes = [ctypes.c_void_p]
        lib.eglGetDisplay.restype = ctypes.c_void_p
        lib.eglGetPlatformDisplay.argtypes = [
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.eglGetPlatformDisplay.restype = ctypes.c_void_p
        lib.eglInitialize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.eglInitialize.restype = ctypes.c_uint
        lib.eglBindAPI.argtypes = [ctypes.c_uint]
        lib.eglBindAPI.restype = ctypes.c_uint
        lib.eglChooseConfig.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.eglChooseConfig.restype = ctypes.c_uint
        lib.eglCreateContext.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.eglCreateContext.restype = ctypes.c_void_p
        lib.eglMakeCurrent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.eglMakeCurrent.restype = ctypes.c_uint
        lib.eglCreatePbufferSurface.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.eglCreatePbufferSurface.restype = ctypes.c_void_p
        lib.eglDestroySurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.eglDestroyContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.eglTerminate.argtypes = [ctypes.c_void_p]

        display_attrs = (ctypes.c_int * 1)(self.EGL_NONE)
        try:
            self.display = lib.eglGetPlatformDisplay(
                self.EGL_PLATFORM_SURFACELESS_MESA,
                ctypes.c_void_p(0),
                display_attrs,
            )
        except AttributeError:
            self.display = self.EGL_NO_DISPLAY
        if not self.display:
            self.display = lib.eglGetDisplay(self.EGL_DEFAULT_DISPLAY)
        if not self.display:
            raise ContextError("E_CONTEXT_EGL", "could not obtain an EGL display")
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._check(lib.eglInitialize(self.display, ctypes.byref(major), ctypes.byref(minor)), "eglInitialize")
        self.initialized = True
        self._check(lib.eglBindAPI(self.EGL_OPENGL_API), "eglBindAPI(OpenGL)")

        config_attributes = (ctypes.c_int * 13)(
            self.EGL_SURFACE_TYPE,
            self.EGL_PBUFFER_BIT,
            self.EGL_RENDERABLE_TYPE,
            self.EGL_OPENGL_BIT,
            self.EGL_RED_SIZE,
            8,
            self.EGL_GREEN_SIZE,
            8,
            self.EGL_BLUE_SIZE,
            8,
            self.EGL_ALPHA_SIZE,
            8,
            self.EGL_NONE,
        )
        configs = (ctypes.c_void_p * 64)()
        count = ctypes.c_int(0)
        self._check(
            lib.eglChooseConfig(
                self.display, config_attributes, configs, 64, ctypes.byref(count)
            ),
            "eglChooseConfig",
        )
        if count.value == 0:
            raise ContextError("E_CONTEXT_EGL", "no matching EGL framebuffer config")
        context_attributes = (ctypes.c_int * 7)(
            self.EGL_CONTEXT_MAJOR_VERSION,
            3,
            self.EGL_CONTEXT_MINOR_VERSION,
            3,
            self.EGL_CONTEXT_OPENGL_PROFILE_MASK,
            self.EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            self.EGL_NONE,
        )
        self.context = lib.eglCreateContext(
            self.display, configs[0], self.EGL_NO_CONTEXT, context_attributes
        )
        if not self.context:
            raise ContextError(
                "E_CONTEXT_EGL",
                "eglCreateContext failed",
                self.ERROR_NAMES.get(lib.eglGetError(), "unknown"),
            )
        if not lib.eglMakeCurrent(
            self.display, self.EGL_NO_SURFACE, self.EGL_NO_SURFACE, self.context
        ):
            pbuffer_attributes = (ctypes.c_int * 5)(
                self.EGL_WIDTH, 1, self.EGL_HEIGHT, 1, self.EGL_NONE
            )
            self.surface = lib.eglCreatePbufferSurface(
                self.display, configs[0], pbuffer_attributes
            )
            if not self.surface or not lib.eglMakeCurrent(
                self.display, self.surface, self.surface, self.context
            ):
                raise ContextError(
                    "E_CONTEXT_EGL",
                    "eglMakeCurrent failed",
                    self.ERROR_NAMES.get(lib.eglGetError(), "unknown"),
                )

    def close(self) -> None:
        if self.lib is None:
            return
        if self.display:
            self.lib.eglMakeCurrent(
                self.display, self.EGL_NO_SURFACE, self.EGL_NO_SURFACE, self.EGL_NO_CONTEXT
            )
            if self.context:
                self.lib.eglDestroyContext(self.display, self.context)
            if self.surface:
                self.lib.eglDestroySurface(self.display, self.surface)
            self.lib.eglTerminate(self.display)
        self.context = self.EGL_NO_CONTEXT
        self.surface = self.EGL_NO_SURFACE
        self.display = self.EGL_NO_DISPLAY
        self.initialized = False


if sys.platform == "win32":
    from ctypes import wintypes

    class _PIXELFORMATDESCRIPTOR(ctypes.Structure):
        _fields_ = [
            ("nSize", wintypes.WORD),
            ("nVersion", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("iPixelType", ctypes.c_ubyte),
            ("cColorBits", ctypes.c_ubyte),
            ("cRedBits", ctypes.c_ubyte),
            ("cRedShift", ctypes.c_ubyte),
            ("cGreenBits", ctypes.c_ubyte),
            ("cGreenShift", ctypes.c_ubyte),
            ("cBlueBits", ctypes.c_ubyte),
            ("cBlueShift", ctypes.c_ubyte),
            ("cAlphaBits", ctypes.c_ubyte),
            ("cAlphaShift", ctypes.c_ubyte),
            ("cAccumBits", ctypes.c_ubyte),
            ("cAccumRedBits", ctypes.c_ubyte),
            ("cAccumGreenBits", ctypes.c_ubyte),
            ("cAccumBlueBits", ctypes.c_ubyte),
            ("cAccumAlphaBits", ctypes.c_ubyte),
            ("cDepthBits", ctypes.c_ubyte),
            ("cStencilBits", ctypes.c_ubyte),
            ("cAuxBuffers", ctypes.c_ubyte),
            ("iLayerType", ctypes.c_ubyte),
            ("bReserved", ctypes.c_ubyte),
            ("dwLayerMask", wintypes.DWORD),
            ("dwVisibleMask", wintypes.DWORD),
            ("dwDamageMask", wintypes.DWORD),
        ]

    class _WNDCLASSW(ctypes.Structure):
        _fields_ = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", ctypes.c_void_p),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", wintypes.HANDLE),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]

    class _WGLBackend:
        PFD_DRAW_TO_WINDOW = 0x00000004
        PFD_SUPPORT_OPENGL = 0x00000020
        PFD_DOUBLEBUFFER = 0x00000001
        PFD_TYPE_RGBA = 0
        PFD_MAIN_PLANE = 0
        WGL_CONTEXT_MAJOR_VERSION_ARB = 0x2091
        WGL_CONTEXT_MINOR_VERSION_ARB = 0x2092
        WGL_CONTEXT_PROFILE_MASK_ARB = 0x2094
        WGL_CONTEXT_CORE_PROFILE_BIT_ARB = 0x00000001

        def __init__(self) -> None:
            self.user32: Any = None
            self.gdi32: Any = None
            self.opengl32: Any = None
            self.wndproc: Any = None
            self.window = ctypes.c_void_p(0)
            self.dc = ctypes.c_void_p(0)
            self.context = ctypes.c_void_p(0)
            self.legacy_context = ctypes.c_void_p(0)
            self.class_name: str | None = None
            self.class_registered = False

        def initialize(self) -> None:
            self.user32 = ctypes.WinDLL("user32", use_last_error=True)
            self.gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
            self.opengl32 = ctypes.WinDLL("opengl32", use_last_error=True)
            self.user32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
            self.user32.GetModuleHandleW.restype = wintypes.HMODULE
            self.user32.RegisterClassW.argtypes = [ctypes.POINTER(_WNDCLASSW)]
            self.user32.RegisterClassW.restype = wintypes.ATOM
            self.user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
            self.user32.UnregisterClassW.restype = wintypes.BOOL
            self.user32.CreateWindowExW.argtypes = [
                wintypes.DWORD,
                wintypes.LPCWSTR,
                wintypes.LPCWSTR,
                wintypes.DWORD,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.HWND,
                wintypes.HMENU,
                wintypes.HINSTANCE,
                ctypes.c_void_p,
            ]
            self.user32.CreateWindowExW.restype = wintypes.HWND
            self.user32.GetDC.argtypes = [wintypes.HWND]
            self.user32.GetDC.restype = wintypes.HDC
            self.user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
            self.user32.ReleaseDC.restype = ctypes.c_int
            self.user32.DestroyWindow.argtypes = [wintypes.HWND]
            self.user32.DestroyWindow.restype = wintypes.BOOL
            self.user32.DefWindowProcW.argtypes = [
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]
            self.user32.DefWindowProcW.restype = ctypes.c_ssize_t
            self.gdi32.ChoosePixelFormat.argtypes = [
                wintypes.HDC,
                ctypes.POINTER(_PIXELFORMATDESCRIPTOR),
            ]
            self.gdi32.ChoosePixelFormat.restype = ctypes.c_int
            self.gdi32.SetPixelFormat.argtypes = [
                wintypes.HDC,
                ctypes.c_int,
                ctypes.POINTER(_PIXELFORMATDESCRIPTOR),
            ]
            self.gdi32.SetPixelFormat.restype = wintypes.BOOL
            self.opengl32.wglCreateContext.argtypes = [wintypes.HDC]
            self.opengl32.wglCreateContext.restype = ctypes.c_void_p
            self.opengl32.wglMakeCurrent.argtypes = [wintypes.HDC, ctypes.c_void_p]
            self.opengl32.wglMakeCurrent.restype = wintypes.BOOL
            self.opengl32.wglDeleteContext.argtypes = [ctypes.c_void_p]
            self.opengl32.wglDeleteContext.restype = wintypes.BOOL
            self.class_name = f"CharacterPreview_{os.getpid()}_{id(self)}"
            self.wndproc = ctypes.WINFUNCTYPE(
                ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p
            )(self._window_proc)
            hinstance = self.user32.GetModuleHandleW(None)
            window_class = _WNDCLASSW(
                0,
                ctypes.cast(self.wndproc, ctypes.c_void_p),
                0,
                0,
                hinstance,
                None,
                None,
                None,
                None,
                self.class_name,
            )
            if not self.user32.RegisterClassW(ctypes.byref(window_class)):
                raise ContextError(
                    "E_CONTEXT_WGL",
                    "RegisterClassW failed",
                    str(ctypes.get_last_error()),
                )
            self.class_registered = True
            self.window = self.user32.CreateWindowExW(
                0,
                self.class_name,
                "CharacterPreviewOffscreen",
                0x80000000,
                0,
                0,
                1,
                1,
                None,
                None,
                hinstance,
                None,
            )
            if not self.window:
                raise ContextError(
                    "E_CONTEXT_WGL", "CreateWindowExW failed", str(ctypes.get_last_error())
                )
            self.dc = self.user32.GetDC(self.window)
            if not self.dc:
                raise ContextError("E_CONTEXT_WGL", "GetDC failed")
            pfd = _PIXELFORMATDESCRIPTOR()
            pfd.nSize = ctypes.sizeof(_PIXELFORMATDESCRIPTOR)
            pfd.nVersion = 1
            pfd.dwFlags = (
                self.PFD_DRAW_TO_WINDOW | self.PFD_SUPPORT_OPENGL | self.PFD_DOUBLEBUFFER
            )
            pfd.iPixelType = self.PFD_TYPE_RGBA
            pfd.cColorBits = 32
            pfd.cAlphaBits = 8
            pfd.cDepthBits = 24
            pfd.iLayerType = self.PFD_MAIN_PLANE
            pixel_format = self.gdi32.ChoosePixelFormat(self.dc, ctypes.byref(pfd))
            if not pixel_format or not self.gdi32.SetPixelFormat(
                self.dc, pixel_format, ctypes.byref(pfd)
            ):
                raise ContextError(
                    "E_CONTEXT_WGL", "could not set an OpenGL pixel format"
                )
            self.legacy_context = self.opengl32.wglCreateContext(self.dc)
            if not self.legacy_context or not self.opengl32.wglMakeCurrent(
                self.dc, self.legacy_context
            ):
                raise ContextError("E_CONTEXT_WGL", "could not create a bootstrap WGL context")
            self.opengl32.wglGetProcAddress.restype = ctypes.c_void_p
            self.opengl32.wglGetProcAddress.argtypes = [ctypes.c_char_p]
            address = self.opengl32.wglGetProcAddress(b"wglCreateContextAttribsARB")
            if address:
                creator_type = ctypes.WINFUNCTYPE(
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_int),
                )
                creator = creator_type(address)
                attributes = (ctypes.c_int * 7)(
                    self.WGL_CONTEXT_MAJOR_VERSION_ARB,
                    3,
                    self.WGL_CONTEXT_MINOR_VERSION_ARB,
                    3,
                    self.WGL_CONTEXT_PROFILE_MASK_ARB,
                    self.WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
                    0,
                )
                self.context = creator(self.dc, None, attributes)
            if not self.context:
                raise ContextError(
                    "E_CONTEXT_WGL",
                    "wglCreateContextAttribsARB failed; OpenGL 3.3 is required",
                )
            if not self.opengl32.wglMakeCurrent(self.dc, self.context):
                raise ContextError("E_CONTEXT_WGL", "wglMakeCurrent failed")
            if self.legacy_context:
                self.opengl32.wglDeleteContext(self.legacy_context)
                self.legacy_context = ctypes.c_void_p(0)

        def _window_proc(self, hwnd: Any, message: int, wparam: Any, lparam: Any) -> int:
            if message == 0x0010:  # WM_CLOSE
                return 0
            return self.user32.DefWindowProcW(hwnd, message, wparam, lparam)

        def close(self) -> None:
            if self.opengl32 is not None:
                self.opengl32.wglMakeCurrent(None, None)
                if self.context:
                    self.opengl32.wglDeleteContext(self.context)
                if self.legacy_context:
                    self.opengl32.wglDeleteContext(self.legacy_context)
            self.context = ctypes.c_void_p(0)
            self.legacy_context = ctypes.c_void_p(0)
            if self.user32 is not None and self.window:
                if self.dc:
                    self.user32.ReleaseDC(self.window, self.dc)
                self.user32.DestroyWindow(self.window)
            if self.user32 is not None and self.class_name and self.class_registered:
                self.user32.UnregisterClassW(
                    self.class_name, self.user32.GetModuleHandleW(None)
                )
                self.class_registered = False
            self.window = ctypes.c_void_p(0)
            self.dc = ctypes.c_void_p(0)

