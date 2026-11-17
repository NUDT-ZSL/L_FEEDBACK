from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Any


class GlfwWindowBackend:
    name = "glfw"

    def __init__(self, width: int, height: int, hidden: bool):
        import glfw  # type: ignore[import-not-found]

        self.glfw = glfw
        self.window = None
        if not glfw.init():
            raise RuntimeError("glfw.init() returned false")
        try:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE if hidden else glfw.TRUE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            self.window = glfw.create_window(
                width,
                height,
                "Deterministic Deferred Renderer",
                None,
                None,
            )
            if self.window is None:
                raise RuntimeError("glfw.create_window() returned null")
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)
        except Exception:
            if self.window is not None:
                glfw.destroy_window(self.window)
            glfw.terminate()
            raise

    def framebuffer_size(self) -> tuple[int, int]:
        return tuple(self.glfw.get_framebuffer_size(self.window))

    def poll_events(self) -> None:
        self.glfw.poll_events()

    def should_close(self) -> bool:
        return bool(self.glfw.window_should_close(self.window))

    def escape_pressed(self) -> bool:
        return (
            self.glfw.get_key(self.window, self.glfw.KEY_ESCAPE)
            == self.glfw.PRESS
        )

    def swap_buffers(self) -> None:
        self.glfw.swap_buffers(self.window)

    def close(self) -> None:
        if self.window is not None:
            self.glfw.destroy_window(self.window)
            self.window = None
        self.glfw.terminate()


class Win32WindowBackend:
    """Minimal Win32/WGL window using only ctypes.

    This is the fallback when the optional glfw package is not installed. The
    window owns a legacy WGL context; ModernGL attaches to the already-current
    context and creates its OpenGL 3.3+ resources from it.
    """

    name = "win32-wgl"
    WS_OVERLAPPEDWINDOW = 0x00CF0000
    CW_USEDEFAULT = 0x80000000
    SW_SHOW = 5
    PM_REMOVE = 0x0001
    WM_DESTROY = 0x0002
    WM_CLOSE = 0x0010
    WM_ERASEBKGND = 0x0014
    WM_KEYDOWN = 0x0100
    VK_ESCAPE = 0x1B
    PFD_DRAW_TO_WINDOW = 0x00000004
    PFD_SUPPORT_OPENGL = 0x00000020
    PFD_DOUBLEBUFFER = 0x00000001
    PFD_TYPE_RGBA = 0
    PFD_MAIN_PLANE = 0

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", wintypes.LPVOID),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", wintypes.HANDLE),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]

    class PIXELFORMATDESCRIPTOR(ctypes.Structure):
        _fields_ = [
            ("nSize", wintypes.WORD),
            ("nVersion", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("iPixelType", wintypes.BYTE),
            ("cColorBits", wintypes.BYTE),
            ("cRedBits", wintypes.BYTE),
            ("cRedShift", wintypes.BYTE),
            ("cGreenBits", wintypes.BYTE),
            ("cGreenShift", wintypes.BYTE),
            ("cBlueBits", wintypes.BYTE),
            ("cBlueShift", wintypes.BYTE),
            ("cAlphaBits", wintypes.BYTE),
            ("cAlphaShift", wintypes.BYTE),
            ("cAccumBits", wintypes.BYTE),
            ("cAccumRedBits", wintypes.BYTE),
            ("cAccumGreenBits", wintypes.BYTE),
            ("cAccumBlueBits", wintypes.BYTE),
            ("cAccumAlphaBits", wintypes.BYTE),
            ("cDepthBits", wintypes.BYTE),
            ("cStencilBits", wintypes.BYTE),
            ("cAuxBuffers", wintypes.BYTE),
            ("iLayerType", wintypes.BYTE),
            ("bReserved", wintypes.BYTE),
            ("dwLayerMask", wintypes.DWORD),
            ("dwVisibleMask", wintypes.DWORD),
            ("dwDamageMask", wintypes.DWORD),
        ]

    class MSG(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("message", wintypes.UINT),
            ("wParam", wintypes.WPARAM),
            ("lParam", wintypes.LPARAM),
            ("time", wintypes.DWORD),
            ("pt", wintypes.POINT),
        ]

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    def __init__(self, width: int, height: int, hidden: bool):
        if not hasattr(ctypes, "WinDLL"):
            raise RuntimeError("Win32 window backend is only available on Windows")
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.opengl32 = ctypes.WinDLL("opengl32", use_last_error=True)
        self._configure_signatures()

        self.hinstance = self.kernel32.GetModuleHandleW(None)
        self.class_name = f"DeferredRendererWindow_{id(self)}"
        self._closed = False
        self._escape = False
        self.hwnd = None
        self.hdc = None
        self.hglrc = None

        wndproc_type = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t,
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        self._wndproc = wndproc_type(self._window_proc)
        window_class = self.WNDCLASSW()
        window_class.style = 0
        window_class.lpfnWndProc = ctypes.cast(self._wndproc, wintypes.LPVOID)
        window_class.hInstance = self.hinstance
        window_class.hCursor = self.user32.LoadCursorW(
            None, ctypes.c_void_p(32512)
        )
        window_class.lpszClassName = self.class_name
        if not self.user32.RegisterClassW(ctypes.byref(window_class)):
            raise RuntimeError(
                f"RegisterClassW failed: {ctypes.get_last_error()}"
            )

        style = self.WS_OVERLAPPEDWINDOW
        rect = self.RECT(0, 0, width, height)
        self.user32.AdjustWindowRect(ctypes.byref(rect), style, False)
        outer_width = rect.right - rect.left
        outer_height = rect.bottom - rect.top
        self.hwnd = self.user32.CreateWindowExW(
            0,
            self.class_name,
            "Deterministic Deferred Renderer",
            style,
            self.CW_USEDEFAULT,
            self.CW_USEDEFAULT,
            outer_width,
            outer_height,
            None,
            None,
            self.hinstance,
            None,
        )
        if not self.hwnd:
            self.user32.UnregisterClassW(self.class_name, self.hinstance)
            raise RuntimeError(
                f"CreateWindowExW failed: {ctypes.get_last_error()}"
            )
        if not hidden:
            self.user32.ShowWindow(self.hwnd, self.SW_SHOW)
            self.user32.UpdateWindow(self.hwnd)

        self.hdc = self.user32.GetDC(self.hwnd)
        if not self.hdc:
            self.close()
            raise RuntimeError("GetDC failed")
        pixel_format = self.PIXELFORMATDESCRIPTOR()
        pixel_format.nSize = ctypes.sizeof(self.PIXELFORMATDESCRIPTOR)
        pixel_format.nVersion = 1
        pixel_format.dwFlags = (
            self.PFD_DRAW_TO_WINDOW
            | self.PFD_SUPPORT_OPENGL
            | self.PFD_DOUBLEBUFFER
        )
        pixel_format.iPixelType = self.PFD_TYPE_RGBA
        pixel_format.cColorBits = 32
        pixel_format.cAlphaBits = 8
        pixel_format.cDepthBits = 24
        pixel_format.cStencilBits = 8
        pixel_format.iLayerType = self.PFD_MAIN_PLANE
        format_index = self.gdi32.ChoosePixelFormat(
            self.hdc, ctypes.byref(pixel_format)
        )
        if format_index == 0 or not self.gdi32.SetPixelFormat(
            self.hdc, format_index, ctypes.byref(pixel_format)
        ):
            error = ctypes.get_last_error()
            self.close()
            raise RuntimeError(f"pixel format selection failed: {error}")
        self.hglrc = self.opengl32.wglCreateContext(self.hdc)
        if not self.hglrc:
            error = ctypes.get_last_error()
            self.close()
            raise RuntimeError(f"wglCreateContext failed: {error}")
        if not self.opengl32.wglMakeCurrent(self.hdc, self.hglrc):
            error = ctypes.get_last_error()
            self.close()
            raise RuntimeError(f"wglMakeCurrent failed: {error}")

    def _configure_signatures(self) -> None:
        self.kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        self.kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE
        self.user32.DefWindowProcW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self.user32.DefWindowProcW.restype = ctypes.c_ssize_t
        self.user32.RegisterClassW.argtypes = [ctypes.POINTER(self.WNDCLASSW)]
        self.user32.RegisterClassW.restype = wintypes.ATOM
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
            wintypes.LPVOID,
        ]
        self.user32.CreateWindowExW.restype = wintypes.HWND
        self.user32.GetDC.argtypes = [wintypes.HWND]
        self.user32.GetDC.restype = wintypes.HDC
        self.user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        self.user32.ReleaseDC.restype = ctypes.c_int
        self.user32.LoadCursorW.argtypes = [
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        self.user32.LoadCursorW.restype = wintypes.HANDLE
        self.user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self.user32.ShowWindow.restype = wintypes.BOOL
        self.user32.UpdateWindow.argtypes = [wintypes.HWND]
        self.user32.UpdateWindow.restype = wintypes.BOOL
        self.user32.DestroyWindow.argtypes = [wintypes.HWND]
        self.user32.DestroyWindow.restype = wintypes.BOOL
        self.user32.UnregisterClassW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.HINSTANCE,
        ]
        self.user32.UnregisterClassW.restype = wintypes.BOOL
        self.user32.AdjustWindowRect.argtypes = [
            ctypes.POINTER(self.RECT),
            wintypes.DWORD,
            wintypes.BOOL,
        ]
        self.user32.AdjustWindowRect.restype = wintypes.BOOL
        self.user32.GetClientRect.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(self.RECT),
        ]
        self.user32.GetClientRect.restype = wintypes.BOOL
        self.user32.PeekMessageW.argtypes = [
            ctypes.POINTER(self.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
        ]
        self.user32.PeekMessageW.restype = wintypes.BOOL
        self.user32.TranslateMessage.argtypes = [ctypes.POINTER(self.MSG)]
        self.user32.TranslateMessage.restype = wintypes.BOOL
        self.user32.DispatchMessageW.argtypes = [ctypes.POINTER(self.MSG)]
        self.user32.DispatchMessageW.restype = ctypes.c_ssize_t
        self.gdi32.ChoosePixelFormat.argtypes = [
            wintypes.HDC,
            ctypes.POINTER(self.PIXELFORMATDESCRIPTOR),
        ]
        self.gdi32.ChoosePixelFormat.restype = ctypes.c_int
        self.gdi32.SetPixelFormat.argtypes = [
            wintypes.HDC,
            ctypes.c_int,
            ctypes.POINTER(self.PIXELFORMATDESCRIPTOR),
        ]
        self.gdi32.SetPixelFormat.restype = wintypes.BOOL
        self.gdi32.SwapBuffers.argtypes = [wintypes.HDC]
        self.gdi32.SwapBuffers.restype = wintypes.BOOL
        self.opengl32.wglCreateContext.argtypes = [wintypes.HDC]
        self.opengl32.wglCreateContext.restype = wintypes.HANDLE
        self.opengl32.wglMakeCurrent.argtypes = [
            wintypes.HDC,
            wintypes.HANDLE,
        ]
        self.opengl32.wglMakeCurrent.restype = wintypes.BOOL
        self.opengl32.wglDeleteContext.argtypes = [wintypes.HANDLE]
        self.opengl32.wglDeleteContext.restype = wintypes.BOOL

    def _window_proc(
        self,
        hwnd: int,
        message: int,
        wparam: int,
        lparam: int,
    ) -> int:
        if message == self.WM_CLOSE:
            self._closed = True
            return 0
        if message == self.WM_DESTROY:
            self._closed = True
            return 0
        if message == self.WM_ERASEBKGND:
            return 1
        if message == self.WM_KEYDOWN and wparam == self.VK_ESCAPE:
            self._escape = True
        return int(
            self.user32.DefWindowProcW(hwnd, message, wparam, lparam)
        )

    def framebuffer_size(self) -> tuple[int, int]:
        rect = self.RECT()
        if not self.user32.GetClientRect(self.hwnd, ctypes.byref(rect)):
            raise RuntimeError("GetClientRect failed")
        return rect.right - rect.left, rect.bottom - rect.top

    def poll_events(self) -> None:
        message = self.MSG()
        while self.user32.PeekMessageW(
            ctypes.byref(message),
            None,
            0,
            0,
            self.PM_REMOVE,
        ):
            if message.message == 0x0012:  # WM_QUIT
                self._closed = True
                break
            self.user32.TranslateMessage(ctypes.byref(message))
            self.user32.DispatchMessageW(ctypes.byref(message))

    def should_close(self) -> bool:
        return self._closed

    def escape_pressed(self) -> bool:
        return self._escape

    def swap_buffers(self) -> None:
        if not self.gdi32.SwapBuffers(self.hdc):
            raise RuntimeError("SwapBuffers failed")

    def close(self) -> None:
        if self.hglrc:
            self.opengl32.wglMakeCurrent(None, None)
            self.opengl32.wglDeleteContext(self.hglrc)
            self.hglrc = None
        if self.hdc and self.hwnd:
            self.user32.ReleaseDC(self.hwnd, self.hdc)
            self.hdc = None
        if self.hwnd:
            self.user32.DestroyWindow(self.hwnd)
            self.hwnd = None
        if self.hinstance:
            try:
                self.user32.UnregisterClassW(self.class_name, self.hinstance)
            except Exception:
                pass


def create_window_backend(
    width: int,
    height: int,
    hidden: bool,
) -> Any:
    errors: list[str] = []
    try:
        return GlfwWindowBackend(width, height, hidden)
    except Exception as exc:
        errors.append(f"glfw: {type(exc).__name__}: {exc}")
        if isinstance(exc, ModuleNotFoundError):
            errors[-1] = "glfw: module not installed"
    import sys

    if sys.platform == "win32":
        try:
            return Win32WindowBackend(width, height, hidden)
        except Exception as exc:
            errors.append(f"win32-wgl: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))

