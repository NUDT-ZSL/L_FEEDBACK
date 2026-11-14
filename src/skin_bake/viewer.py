"""Win32/WGL interactive viewer and local offscreen OpenGL checks."""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .errors import gl_runtime, gl_unavailable
from .runtime import BakedAsset, load_baked_asset
from .software import write_png


def _make_windows_api():
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    opengl32 = ctypes.WinDLL("opengl32", use_last_error=True)

    def proto(lib, name, restype, argtypes):
        func = getattr(lib, name)
        func.restype = restype
        func.argtypes = argtypes
        return func

    WNDPROC = ctypes.WINFUNCTYPE(
        ctypes.c_ssize_t,
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    )

    class WNDCLASSEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.UINT),
            ("style", wintypes.UINT),
            ("lpfnWndProc", WNDPROC),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", wintypes.HANDLE),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
            ("hIconSm", wintypes.HICON),
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

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
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

    api = {
        "user32": user32,
        "gdi32": gdi32,
        "opengl32": opengl32,
        "WNDPROC": WNDPROC,
        "WNDCLASSEXW": WNDCLASSEXW,
        "PIXELFORMATDESCRIPTOR": PIXELFORMATDESCRIPTOR,
        "RECT": RECT,
        "MSG": MSG,
    }
    proto(user32, "GetModuleHandleW", wintypes.HMODULE, [wintypes.LPCWSTR])
    proto(
        user32,
        "RegisterClassExW",
        wintypes.ATOM,
        [ctypes.POINTER(WNDCLASSEXW)],
    )
    proto(user32, "UnregisterClassW", wintypes.BOOL, [wintypes.LPCWSTR, wintypes.HINSTANCE])
    proto(
        user32,
        "CreateWindowExW",
        wintypes.HWND,
        [
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
        ],
    )
    proto(user32, "DefWindowProcW", ctypes.c_ssize_t, [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM])
    proto(user32, "DestroyWindow", wintypes.BOOL, [wintypes.HWND])
    proto(user32, "ShowWindow", wintypes.BOOL, [wintypes.HWND, ctypes.c_int])
    proto(user32, "UpdateWindow", wintypes.BOOL, [wintypes.HWND])
    proto(user32, "GetClientRect", wintypes.BOOL, [wintypes.HWND, ctypes.POINTER(RECT)])
    proto(user32, "GetDC", wintypes.HDC, [wintypes.HWND])
    proto(user32, "ReleaseDC", ctypes.c_int, [wintypes.HWND, wintypes.HDC])
    proto(user32, "PostQuitMessage", None, [ctypes.c_int])
    proto(user32, "PeekMessageW", wintypes.BOOL, [ctypes.POINTER(MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT])
    proto(user32, "TranslateMessage", wintypes.BOOL, [ctypes.POINTER(MSG)])
    proto(user32, "DispatchMessageW", ctypes.c_ssize_t, [ctypes.POINTER(MSG)])
    proto(user32, "SetWindowTextW", wintypes.BOOL, [wintypes.HWND, wintypes.LPCWSTR])
    proto(user32, "LoadCursorW", wintypes.HANDLE, [wintypes.HINSTANCE, ctypes.c_void_p])
    proto(gdi32, "ChoosePixelFormat", ctypes.c_int, [wintypes.HDC, ctypes.POINTER(PIXELFORMATDESCRIPTOR)])
    proto(gdi32, "SetPixelFormat", wintypes.BOOL, [wintypes.HDC, ctypes.c_int, ctypes.POINTER(PIXELFORMATDESCRIPTOR)])
    proto(gdi32, "SwapBuffers", wintypes.BOOL, [wintypes.HDC])
    proto(opengl32, "wglCreateContext", ctypes.c_void_p, [wintypes.HDC])
    proto(opengl32, "wglDeleteContext", wintypes.BOOL, [ctypes.c_void_p])
    proto(opengl32, "wglMakeCurrent", wintypes.BOOL, [wintypes.HDC, ctypes.c_void_p])
    proto(opengl32, "wglGetProcAddress", ctypes.c_void_p, [ctypes.c_char_p])
    return api


class WGLWindow:
    def __init__(self, width: int, height: int, title: str, *, visible: bool = True):
        self.api = _make_windows_api()
        self.user32 = self.api["user32"]
        self.gdi32 = self.api["gdi32"]
        self.opengl32 = self.api["opengl32"]
        self.keys: list[int] = []
        self.running = True
        self.hinstance = self.user32.GetModuleHandleW(None)
        self.class_name = f"SkinBakeViewer_{os.getpid()}_{id(self)}"
        self._wndproc = self.api["WNDPROC"](self._on_message)
        wc = self.api["WNDCLASSEXW"]()
        wc.cbSize = ctypes.sizeof(self.api["WNDCLASSEXW"])
        wc.style = 0x0002 | 0x0001 | 0x0020
        wc.lpfnWndProc = self._wndproc
        wc.hInstance = self.hinstance
        wc.hCursor = self.user32.LoadCursorW(None, ctypes.c_void_p(32512))
        wc.lpszClassName = self.class_name
        if not self.user32.RegisterClassExW(ctypes.byref(wc)):
            raise gl_runtime(f"RegisterClassExW failed: {ctypes.get_last_error()}")
        style = 0x00CF0000 | 0x10000000 if visible else 0x80000000
        self.hwnd = self.user32.CreateWindowExW(
            0,
            self.class_name,
            title,
            style,
            80,
            80,
            int(width),
            int(height),
            None,
            None,
            self.hinstance,
            None,
        )
        if not self.hwnd:
            self.user32.UnregisterClassW(self.class_name, self.hinstance)
            raise gl_runtime(f"CreateWindowExW failed: {ctypes.get_last_error()}")
        self.hdc = self.user32.GetDC(self.hwnd)
        if not self.hdc:
            raise gl_runtime("GetDC failed")
        pfd = self.api["PIXELFORMATDESCRIPTOR"]()
        pfd.nSize = ctypes.sizeof(self.api["PIXELFORMATDESCRIPTOR"])
        pfd.nVersion = 1
        pfd.dwFlags = 0x00000004 | 0x00000020 | 0x00000001
        pfd.iPixelType = 0
        pfd.cColorBits = 32
        pfd.cDepthBits = 24
        pfd.cStencilBits = 8
        pixel_format = self.gdi32.ChoosePixelFormat(self.hdc, ctypes.byref(pfd))
        if pixel_format == 0 or not self.gdi32.SetPixelFormat(
            self.hdc, pixel_format, ctypes.byref(pfd)
        ):
            raise gl_runtime(f"could not set an OpenGL pixel format: {ctypes.get_last_error()}")
        self.gl_context = self._create_core_context()
        if not self.opengl32.wglMakeCurrent(self.hdc, self.gl_context):
            raise gl_runtime("wglMakeCurrent failed")
        self._set_swap_interval(1)
        if visible:
            self.user32.ShowWindow(self.hwnd, 5)
            self.user32.UpdateWindow(self.hwnd)

    def _create_core_context(self) -> int:
        dummy = self.opengl32.wglCreateContext(self.hdc)
        if not dummy:
            raise gl_runtime("wglCreateContext failed")
        if not self.opengl32.wglMakeCurrent(self.hdc, dummy):
            raise gl_runtime("could not make the bootstrap WGL context current")
        address = self.opengl32.wglGetProcAddress(b"wglCreateContextAttribsARB")
        if not address:
            self.opengl32.wglDeleteContext(dummy)
            raise gl_unavailable("driver does not expose WGL_ARB_create_context")
        create = ctypes.WINFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p,
        )(address)
        attributes = (ctypes.c_int * 7)(0x2091, 3, 0x2092, 3, 0x9126, 0x00000001, 0)
        context = create(self.hdc, attributes, None)
        if not context:
            self.opengl32.wglDeleteContext(dummy)
            raise gl_unavailable("driver could not create an OpenGL 3.3 core context")
        self.opengl32.wglMakeCurrent(self.hdc, context)
        self.opengl32.wglDeleteContext(dummy)
        return context

    def _set_swap_interval(self, interval: int) -> None:
        address = self.opengl32.wglGetProcAddress(b"wglSwapIntervalEXT")
        if address:
            func = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int)(address)
            func(interval)

    def _on_message(self, hwnd, message, wparam, lparam):
        if message == 0x0100:
            self.keys.append(int(wparam))
            return 0
        if message == 0x0010:
            self.user32.DestroyWindow(hwnd)
            return 0
        if message == 0x0002:
            self.running = False
            self.user32.PostQuitMessage(0)
            return 0
        return self.user32.DefWindowProcW(hwnd, message, wparam, lparam)

    def pump(self) -> None:
        msg = self.api["MSG"]()
        while self.user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
            if msg.message == 0x0012:
                self.running = False
                break
            self.user32.TranslateMessage(ctypes.byref(msg))
            self.user32.DispatchMessageW(ctypes.byref(msg))

    def client_size(self) -> tuple[int, int]:
        rect = self.api["RECT"]()
        if not self.user32.GetClientRect(self.hwnd, ctypes.byref(rect)):
            return (1, 1)
        return (max(1, rect.right - rect.left), max(1, rect.bottom - rect.top))

    def swap(self) -> None:
        self.gdi32.SwapBuffers(self.hdc)

    def set_title(self, title: str) -> None:
        self.user32.SetWindowTextW(self.hwnd, title)

    def close(self) -> None:
        if getattr(self, "gl_context", None):
            self.opengl32.wglMakeCurrent(None, None)
            self.opengl32.wglDeleteContext(self.gl_context)
            self.gl_context = None
        if getattr(self, "hdc", None):
            self.user32.ReleaseDC(self.hwnd, self.hdc)
            self.hdc = None
        if getattr(self, "hwnd", None):
            self.user32.DestroyWindow(self.hwnd)
            self.hwnd = None
        self.user32.UnregisterClassW(self.class_name, self.hinstance)


def _run_windows_viewer(
    asset: BakedAsset,
    start_frame: int,
    width: int,
    height: int,
) -> int:
    from .gl_renderer import GLRenderer

    window = WGLWindow(width, height, "Skin Bake Viewer")
    try:
        renderer = GLRenderer(asset, width, height)
        current_time = start_frame / asset.scene.fps
        playing = True
        previous = time.perf_counter()
        last_title_update = 0.0
        while window.running:
            window.pump()
            if not window.running:
                break
            window_width, window_height = window.client_size()
            if window_width != renderer.width or window_height != renderer.height:
                renderer.width = window_width
                renderer.height = window_height
            keys, window.keys = window.keys, []
            for key in keys:
                if key == 0x1B:
                    window.user32.DestroyWindow(window.hwnd)
                    window.running = False
                    break
                if key == 0x20:
                    playing = not playing
                elif key in (0x25, 0x27):
                    playing = False
                    frame = int(round(current_time * asset.scene.fps))
                    delta = -1 if key == 0x25 else 1
                    if asset.scene.loop_mode == "loop":
                        frame = (frame + delta) % asset.frame_count
                    else:
                        frame = min(asset.frame_count - 1, max(0, frame + delta))
                    current_time = frame / asset.scene.fps
                elif key == 0x52:
                    playing = True
                    current_time = 0.0
            now = time.perf_counter()
            delta_time = min(0.1, now - previous)
            previous = now
            if playing:
                current_time += delta_time
                if current_time > asset.scene.duration:
                    if asset.scene.loop_mode == "loop":
                        current_time %= asset.scene.duration
                    else:
                        current_time = asset.scene.duration
                        playing = False
            renderer.render(current_time)
            window.swap()
            if now - last_title_update > 0.1:
                frame = int(round(current_time * asset.scene.fps))
                if asset.scene.loop_mode == "loop":
                    frame %= asset.frame_count
                else:
                    frame = min(asset.frame_count - 1, frame)
                state = "PLAY" if playing else "PAUSE"
                window.set_title(
                    f"Skin Bake Viewer | {state} | frame {frame}/{asset.frame_count - 1} "
                    f"| {current_time:.3f}s | arrows step, space play, R reset, Esc quit"
                )
                last_title_update = now
            time.sleep(1.0 / 144.0)
        return 0
    finally:
        window.close()


def _run_windows_offscreen(
    asset: BakedAsset,
    start_frame: int,
    width: int,
    height: int,
    output: Path,
) -> int:
    from .gl_renderer import GLRenderer

    window = WGLWindow(width, height, "Skin Bake Offscreen", visible=False)
    try:
        renderer = GLRenderer(asset, width, height)
        current_time = start_frame / asset.scene.fps
        result = renderer.render(current_time)
        write_png(output, result.image)
        summary = {
            "status": "ok",
            "backend": "wgl-offscreen-opengl",
            "glVersion": result.version,
            "glRenderer": result.renderer,
            "frame": start_frame,
            "sampleIndex": result.sample_index,
            "preview": str(output),
            "nonBackgroundPixels": int(
                np.count_nonzero(
                    np.any(result.image > np.array([18, 22, 28], dtype=np.uint8), axis=2)
                )
            ),
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        window.close()


def _run_egl_offscreen(
    asset: BakedAsset,
    start_frame: int,
    width: int,
    height: int,
    output: Path,
) -> int:
    from .egl_headless import render_egl_frame

    current_time = start_frame / asset.scene.fps
    result = render_egl_frame(asset, current_time, width, height)
    write_png(output, result.image)
    summary = {
        "status": "ok",
        "backend": "egl-offscreen-opengl",
        "glVersion": result.version,
        "glRenderer": result.renderer,
        "frame": start_frame,
        "sampleIndex": result.sample_index,
        "preview": str(output),
        "nonBackgroundPixels": int(
            np.count_nonzero(
                np.any(result.image > np.array([18, 22, 28], dtype=np.uint8), axis=2)
            )
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def run_viewer(
    scene_path: str | Path,
    bake_dir: str | Path,
    *,
    start_frame: int = 0,
    headless_output: Path | None = None,
    width: int = 900,
    height: int = 640,
) -> int:
    asset = load_baked_asset(scene_path, bake_dir)
    if start_frame < 0 or start_frame >= asset.frame_count:
        raise gl_runtime(
            f"start frame must be in [0, {asset.frame_count - 1}], got {start_frame}"
        )
    width = max(64, int(width))
    height = max(64, int(height))
    if headless_output is not None:
        output = Path(headless_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            return _run_windows_offscreen(asset, start_frame, width, height, output)
        if sys.platform.startswith("linux"):
            return _run_egl_offscreen(asset, start_frame, width, height, output)
        raise gl_unavailable("offscreen OpenGL check is supported on Windows and Linux")
    if os.name != "nt":
        raise gl_unavailable(
            "the interactive context is Win32/WGL; use --headless-check for local EGL verification"
        )
    return _run_windows_viewer(asset, start_frame, width, height)

