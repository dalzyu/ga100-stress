"""Headless GL verification: EGL surfaceless context, FBO render, CUDA interop."""

from __future__ import annotations

import ctypes
import glob
import pathlib

import numpy as np
import torch

from .guard import require_bare_gpu
from .report import Result

# EGL constants
_EGL_PLATFORM_DEVICE_EXT = 0x313F
_EGL_RENDERABLE_TYPE = 0x3040
_EGL_OPENGL_BIT = 0x0008
_EGL_SURFACE_TYPE = 0x3033
_EGL_PBUFFER_BIT = 0x0001
_EGL_NONE = 0x3038
_EGL_VENDOR = 0x3053
_EGL_CONTEXT_MAJOR_VERSION = 0x3098
_EGL_CONTEXT_MINOR_VERSION = 0x30FB
_EGL_OPENGL_API = 0x30A2
_EGL_WIDTH = 0x3057
_EGL_HEIGHT = 0x3056

# GL constants
_GL_RENDERER = 0x1F01
_GL_COMPILE_STATUS = 0x8B81
_GL_LINK_STATUS = 0x8B82
_GL_FRAMEBUFFER = 0x8D40
_GL_FRAMEBUFFER_COMPLETE = 0x8CD5
_GL_COLOR_ATTACHMENT0 = 0x8CE0
_GL_TEXTURE_2D = 0x0DE1
_GL_RGBA8 = 0x8058
_GL_RGBA = 0x1908
_GL_UNSIGNED_BYTE = 0x1401
_GL_ARRAY_BUFFER = 0x8892
_GL_STATIC_DRAW = 0x88E4
_GL_TRIANGLES = 0x0004

_W = _H = 256


class _GlError(RuntimeError):
    pass


def _load_cudart() -> ctypes.CDLL:
    try:
        return ctypes.CDLL("libcudart.so.12")
    except OSError:
        root = pathlib.Path(torch.__file__).parent
        for pat in (
            root / "lib" / "libcudart.so*",
            root.parent / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so*",
        ):
            for p in sorted(glob.glob(str(pat))):
                return ctypes.CDLL(p)
    raise _GlError("libcudart.so not found")


def _bind(decls: dict[str, tuple], lib: ctypes.CDLL) -> None:
    for name, (res, args) in decls.items():
        fn = getattr(lib, name)
        fn.restype = res
        fn.argtypes = args


def _nvidia_display(egl) -> int:
    n = ctypes.c_int(0)
    if egl.eglQueryDevicesEXT(0, None, ctypes.byref(n)) == 0 or n.value == 0:
        raise _GlError("eglQueryDevicesEXT found no EGL devices")
    devices = (ctypes.c_void_p * n.value)()
    if egl.eglQueryDevicesEXT(n.value, devices, ctypes.byref(n)) == 0:
        raise _GlError("eglQueryDevicesEXT failed")
    for dev in devices:
        dpy = egl.eglGetPlatformDisplayEXT(_EGL_PLATFORM_DEVICE_EXT, dev, None)
        if not dpy:
            continue
        if egl.eglInitialize(dpy, None, None) == 0:
            continue
        vendor = egl.eglQueryString(dpy, _EGL_VENDOR)
        if vendor and b"NVIDIA" in vendor:
            return dpy
        egl.eglTerminate(dpy)
    raise _GlError("no NVIDIA EGL display found")


def _make_context(egl, dpy: int):
    egl.eglBindAPI(_EGL_OPENGL_API)
    cfg = ctypes.c_void_p()
    count = ctypes.c_int(0)
    for surface_type in (0, _EGL_PBUFFER_BIT):  # surfaceless first, pbuffer fallback
        attribs = (ctypes.c_int * 5)(
            _EGL_RENDERABLE_TYPE, _EGL_OPENGL_BIT,
            _EGL_SURFACE_TYPE, surface_type,
            _EGL_NONE,
        )
        if egl.eglChooseConfig(dpy, attribs, ctypes.byref(cfg), 1, ctypes.byref(count)) == 0 or count.value < 1:
            continue
        cattr = (ctypes.c_int * 7)(
            _EGL_CONTEXT_MAJOR_VERSION, 3, _EGL_CONTEXT_MINOR_VERSION, 3, _EGL_NONE, 0, 0
        )
        ctx = egl.eglCreateContext(dpy, cfg, None, cattr)
        if not ctx:
            continue
        surf = ctypes.c_void_p(0)
        if surface_type == _EGL_PBUFFER_BIT:
            pattr = (ctypes.c_int * 5)(_EGL_WIDTH, 16, _EGL_HEIGHT, 16, _EGL_NONE)
            surf = egl.eglCreatePbufferSurface(dpy, cfg, pattr)
            if not surf:
                egl.eglDestroyContext(dpy, ctx)
                continue
        if egl.eglMakeCurrent(dpy, surf, surf, ctx) == 0:
            egl.eglDestroyContext(dpy, ctx)
            continue
        return ctx, surf
    raise _GlError("could not create an EGL GL context")


_VS = b"""
#version 330 core
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_FS = b"""
#version 330 core
out vec4 fragColor;
void main() {
    float r = mod(floor(gl_FragCoord.x), 256.0) / 255.0;
    float g = mod(floor(gl_FragCoord.y), 256.0) / 255.0;
    float b = mod(floor(gl_FragCoord.x + gl_FragCoord.y), 256.0) / 255.0;
    fragColor = vec4(r, g, b, 1.0);
}
"""


def _shader(gl, kind: int, src: bytes) -> int:
    sh = gl.glCreateShader(kind)
    buf = ctypes.c_char_p(src)
    ln = ctypes.c_int(len(src))
    gl.glShaderSource(sh, 1, ctypes.byref(buf), ctypes.byref(ln))
    gl.glCompileShader(sh)
    ok = ctypes.c_int(0)
    gl.glGetShaderiv(sh, _GL_COMPILE_STATUS, ctypes.byref(ok))
    if not ok.value:
        log = ctypes.create_string_buffer(1024)
        gl.glGetShaderInfoLog(sh, 1024, None, log)
        raise _GlError(f"shader compile failed: {log.value.decode(errors='replace')}")
    return sh


def _expected_pixels() -> np.ndarray:
    xs = np.arange(_W, dtype=np.float32) + np.float32(0.5)
    ys = np.arange(_H, dtype=np.float32) + np.float32(0.5)
    fx, fy = np.meshgrid(xs, ys)
    r = (np.floor(fx) % 256.0).astype(np.float32) / np.float32(255.0)
    g = (np.floor(fy) % 256.0).astype(np.float32) / np.float32(255.0)
    b = (np.floor(fx + fy) % 256.0).astype(np.float32) / np.float32(255.0)
    px = np.stack([r, g, b, np.ones_like(r)], axis=-1)
    return np.round(px * np.float32(255.0)).astype(np.uint8)


def run(device: int = 0) -> Result:
    require_bare_gpu(device)
    torch.cuda.set_device(device)
    torch.zeros(1, device=device)
    metrics: dict = {}
    notes: list[str] = []

    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglQueryDevicesEXT.restype = ctypes.c_int
    egl.eglQueryDevicesEXT.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)]
    egl.eglGetPlatformDisplayEXT.restype = ctypes.c_void_p
    egl.eglGetPlatformDisplayEXT.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    egl.eglInitialize.restype = ctypes.c_uint
    egl.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    egl.eglQueryString.restype = ctypes.c_char_p
    egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
    egl.eglChooseConfig.restype = ctypes.c_uint
    egl.eglChooseConfig.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    egl.eglBindAPI.restype = ctypes.c_uint
    egl.eglBindAPI.argtypes = [ctypes.c_uint]
    egl.eglCreateContext.restype = ctypes.c_void_p
    egl.eglCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    egl.eglCreatePbufferSurface.restype = ctypes.c_void_p
    egl.eglCreatePbufferSurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    egl.eglMakeCurrent.restype = ctypes.c_uint
    egl.eglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    egl.eglDestroyContext.restype = ctypes.c_uint
    egl.eglDestroyContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    egl.eglDestroySurface.restype = ctypes.c_uint
    egl.eglDestroySurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    egl.eglTerminate.restype = ctypes.c_uint
    egl.eglTerminate.argtypes = [ctypes.c_void_p]

    gl = ctypes.CDLL("libGL.so.1")
    _bind(
        {
            "glGetString": (ctypes.c_char_p, [ctypes.c_uint]),
            "glGetError": (ctypes.c_uint, []),
            "glCreateShader": (ctypes.c_uint, [ctypes.c_uint]),
            "glShaderSource": (None, [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int)]),
            "glCompileShader": (None, [ctypes.c_uint]),
            "glGetShaderiv": (None, [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]),
            "glGetShaderInfoLog": (None, [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p]),
            "glCreateProgram": (ctypes.c_uint, []),
            "glAttachShader": (None, [ctypes.c_uint, ctypes.c_uint]),
            "glLinkProgram": (None, [ctypes.c_uint]),
            "glGetProgramiv": (None, [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]),
            "glGetProgramInfoLog": (None, [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p]),
            "glDeleteShader": (None, [ctypes.c_uint]),
            "glGenVertexArrays": (None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]),
            "glBindVertexArray": (None, [ctypes.c_uint]),
            "glGenFramebuffers": (None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]),
            "glBindFramebuffer": (None, [ctypes.c_uint, ctypes.c_uint]),
            "glGenTextures": (None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]),
            "glBindTexture": (None, [ctypes.c_uint, ctypes.c_uint]),
            "glTexImage2D": (None, [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]),
            "glFramebufferTexture2D": (None, [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_int]),
            "glCheckFramebufferStatus": (ctypes.c_uint, [ctypes.c_uint]),
            "glViewport": (None, [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]),
            "glUseProgram": (None, [ctypes.c_uint]),
            "glDrawArrays": (None, [ctypes.c_uint, ctypes.c_int, ctypes.c_int]),
            "glReadPixels": (None, [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]),
            "glDeleteTextures": (None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]),
            "glDeleteFramebuffers": (None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]),
        },
        gl,
    )

    dpy = _nvidia_display(egl)
    ctx = surf = None
    try:
        ctx, surf = _make_context(egl, dpy)
        renderer = gl.glGetString(_GL_RENDERER)
        metrics["renderer"] = renderer.decode(errors="replace") if renderer else ""

        prog = gl.glCreateProgram()
        vs = _shader(gl, 0x8B31, _VS)  # GL_VERTEX_SHADER
        fs = _shader(gl, 0x8B30, _FS)  # GL_FRAGMENT_SHADER
        gl.glAttachShader(prog, vs)
        gl.glAttachShader(prog, fs)
        gl.glLinkProgram(prog)
        ok = ctypes.c_int(0)
        gl.glGetProgramiv(prog, _GL_LINK_STATUS, ctypes.byref(ok))
        if not ok.value:
            log = ctypes.create_string_buffer(1024)
            gl.glGetProgramInfoLog(prog, 1024, None, log)
            raise _GlError(f"program link failed: {log.value.decode(errors='replace')}")

        vao = ctypes.c_uint(0)
        gl.glGenVertexArrays(1, ctypes.byref(vao))
        gl.glBindVertexArray(vao)

        fbo = ctypes.c_uint(0)
        tex = ctypes.c_uint(0)
        gl.glGenFramebuffers(1, ctypes.byref(fbo))
        gl.glGenTextures(1, ctypes.byref(tex))
        gl.glBindFramebuffer(_GL_FRAMEBUFFER, fbo)
        gl.glBindTexture(_GL_TEXTURE_2D, tex)
        gl.glTexImage2D(_GL_TEXTURE_2D, 0, _GL_RGBA8, _W, _H, 0, _GL_RGBA, _GL_UNSIGNED_BYTE, None)
        gl.glFramebufferTexture2D(_GL_FRAMEBUFFER, _GL_COLOR_ATTACHMENT0, _GL_TEXTURE_2D, tex, 0)
        if gl.glCheckFramebufferStatus(_GL_FRAMEBUFFER) != _GL_FRAMEBUFFER_COMPLETE:
            raise _GlError("framebuffer incomplete")
        gl.glViewport(0, 0, _W, _H)
        gl.glUseProgram(prog)
        gl.glDrawArrays(_GL_TRIANGLES, 0, 3)

        got = np.empty((_H, _W, 4), dtype=np.uint8)
        gl.glReadPixels(0, 0, _W, _H, _GL_RGBA, _GL_UNSIGNED_BYTE, got.ctypes.data)
        expected = _expected_pixels()
        max_diff = int(np.abs(got.astype(np.int16) - expected.astype(np.int16)).max())
        metrics["render_max_abs_diff"] = max_diff
        if max_diff > 1:
            notes.append(f"FBO render differs from CPU reference by up to {max_diff}")

        # CUDA <-> GL interop: map the rendered texture into CUDA and read it back.
        cudart = _load_cudart()
        res = ctypes.c_void_p()
        st = cudart.cudaGraphicsGLRegisterImage(
            ctypes.byref(res), ctypes.c_uint(tex), ctypes.c_uint(_GL_TEXTURE_2D), ctypes.c_uint(0)
        )
        if st != 0:
            raise _GlError(f"cudaGraphicsGLRegisterImage failed: {st}")
        try:
            st = cudart.cudaGraphicsMapResources(1, ctypes.byref(res), ctypes.c_void_p(0))
            if st != 0:
                raise _GlError(f"cudaGraphicsMapResources failed: {st}")
            arr = ctypes.c_void_p()
            st = cudart.cudaGraphicsSubResourceGetMappedArray(ctypes.byref(arr), res, ctypes.c_size_t(0), ctypes.c_size_t(0))
            if st != 0:
                raise _GlError(f"cudaGraphicsSubResourceGetMappedArray failed: {st}")
            dev = torch.empty(_H * _W * 4, dtype=torch.uint8, device=device)
            st = cudart.cudaMemcpy2DFromArray(
                ctypes.c_void_p(dev.data_ptr()),
                ctypes.c_size_t(_W * 4),
                ctypes.c_void_p(arr.value),
                ctypes.c_size_t(0),
                ctypes.c_size_t(0),
                ctypes.c_size_t(_W * 4),
                ctypes.c_size_t(_H),
                ctypes.c_int(3),  # cudaMemcpyDeviceToDevice
            )
            if st != 0:
                raise _GlError(f"cudaMemcpy2DFromArray failed: {st}")
            torch.cuda.synchronize()
            via_cuda = dev.cpu().numpy().reshape(_H, _W, 4)
            interop_diff = int(np.abs(via_cuda.astype(np.int16) - got.astype(np.int16)).max())
            metrics["interop_max_abs_diff"] = interop_diff
            if interop_diff > 1:
                notes.append(f"CUDA-GL interop roundtrip differs by up to {interop_diff}")
            passed = max_diff <= 1 and interop_diff <= 1
        finally:
            cudart.cudaGraphicsUnmapResources(1, ctypes.byref(res), ctypes.c_void_p(0))
            cudart.cudaGraphicsUnregisterResource(res)
    finally:
        egl.eglMakeCurrent(dpy, None, None, None)
        if ctx:
            egl.eglDestroyContext(dpy, ctx)
        if surf:
            egl.eglDestroySurface(dpy, surf)
        egl.eglTerminate(dpy)

    return Result(name="gl", passed=passed, metrics=metrics, notes=notes)
