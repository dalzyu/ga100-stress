"""OFA (optical flow accelerator) verification via the nvofapi C API.

Binds libnvidia-opticalflow.so.1 directly (no SDK headers needed at runtime);
struct layouts and enum values mirror the public NVIDIA Optical Flow SDK
headers (nvOpticalFlowCommon.h, nvOpticalFlowCuda.h).
"""

from __future__ import annotations

import ctypes

import numpy as np
import torch

from .cudart import load_cudart
from .guard import require_bare_gpu
from .report import Result

# --- enum values (nvOpticalFlowCommon.h) ---
_NV_OF_MODE_OPTICALFLOW = 1
_NV_OF_OUTPUT_VECTOR_GRID_SIZE_1 = 1
_NV_OF_HINT_VECTOR_GRID_SIZE_1 = 1
_NV_OF_PERF_LEVEL_FAST = 20
_NV_OF_BUFFER_USAGE_INPUT = 1
_NV_OF_BUFFER_USAGE_OUTPUT = 2
_NV_OF_BUFFER_FORMAT_NV12 = 2
_NV_OF_BUFFER_FORMAT_SHORT2 = 5
_NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR = 2

# cudaMemcpyKind
_MEMCPY_D2D = 3
_MEMCPY_D2H = 2

_STATUS_NAMES = {
    0: "SUCCESS",
    1: "ERR_OF_NOT_AVAILABLE",
    2: "ERR_UNSUPPORTED_DEVICE",
    3: "ERR_DEVICE_DOES_NOT_EXIST",
    4: "ERR_INVALID_PTR",
    5: "ERR_INVALID_PARAM",
    6: "ERR_INVALID_CALL",
    7: "ERR_INVALID_VERSION",
    8: "ERR_OUT_OF_MEMORY",
    9: "ERR_NOT_INITIALIZED",
    10: "ERR_UNSUPPORTED_FEATURE",
    11: "ERR_GENERIC",
}


class _OfaError(RuntimeError):
    pass


class _InitParams(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("outGridSize", ctypes.c_uint32),
        ("hintGridSize", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("perfLevel", ctypes.c_uint32),
        ("enableExternalHints", ctypes.c_uint32),
        ("enableOutputCost", ctypes.c_uint32),
        ("hPrivData", ctypes.c_void_p),
        ("disparityRange", ctypes.c_uint32),
        ("enableRoi", ctypes.c_uint32),
    ]


class _BufferDescriptor(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("bufferUsage", ctypes.c_uint32),
        ("bufferFormat", ctypes.c_uint32),
    ]


class _ExecuteInputParams(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("inputFrame", ctypes.c_void_p),
        ("referenceFrame", ctypes.c_void_p),
        ("externalHints", ctypes.c_void_p),
        ("disableTemporalHints", ctypes.c_uint32),
        ("padding", ctypes.c_uint32),
        ("hPrivData", ctypes.c_void_p),
        ("padding2", ctypes.c_uint32),
        ("numRois", ctypes.c_uint32),
        ("roiData", ctypes.c_void_p),
    ]


class _ExecuteOutputParams(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("outputBuffer", ctypes.c_void_p),
        ("outputCostBuffer", ctypes.c_void_p),
        ("hPrivData", ctypes.c_void_p),
    ]


class _Stride(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [("strideXInBytes", ctypes.c_uint32), ("strideYInBytes", ctypes.c_uint32)]


class _StrideInfo(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [("strideInfo", _Stride * 3), ("numPlanes", ctypes.c_uint32)]


_PFN_CREATE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
_PFN_INIT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
_PFN_CREATEBUF = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)
)
_PFN_GETCUARRAY = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)
_PFN_GETCUDEVPTR = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)
_PFN_GETSTRIDE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
_PFN_SETSTREAMS = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64)
_PFN_EXECUTE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
_PFN_DESTROYBUF = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_PFN_DESTROY = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_PFN_LASTERR = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)
)
_PFN_GETCAPS = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_uint32),
)


class _ApiFunctionList(ctypes.Structure):
    # Order mirrors NV_OF_CUDA_API_FUNCTION_LIST in nvOpticalFlowCuda.h.
    _fields_ = [
        ("createOpticalFlow", _PFN_CREATE),
        ("init", _PFN_INIT),
        ("createGPUBuffer", _PFN_CREATEBUF),
        ("gpuBufferGetCUarray", _PFN_GETCUARRAY),
        ("gpuBufferGetCUdeviceptr", _PFN_GETCUDEVPTR),
        ("gpuBufferGetStrideInfo", _PFN_GETSTRIDE),
        ("setIOCudaStreams", _PFN_SETSTREAMS),
        ("execute", _PFN_EXECUTE),
        ("destroyGPUBuffer", _PFN_DESTROYBUF),
        ("destroy", _PFN_DESTROY),
        ("getLastError", _PFN_LASTERR),
        ("getCaps", _PFN_GETCAPS),
    ]


def _memcpy(cudart: ctypes.CDLL, dst: int, src: int, count: int, kind: int) -> None:
    st = cudart.cudaMemcpy(
        ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(count), ctypes.c_int(kind)
    )
    if st != 0:
        raise _OfaError(f"cudaMemcpy failed with status {st}")


def _synthetic_nv12(width: int, height: int, device: int):
    # Noise texture: unique local features, so the optical flow field is
    # unambiguous (smooth gradients give the engine nothing to lock onto).
    g = torch.Generator(device=device)
    g.manual_seed(0x0FA)
    y = torch.randint(0, 256, (height, width), dtype=torch.uint8, device=device, generator=g)
    b = torch.roll(y, shifts=(3, 5), dims=(0, 1))
    uv = torch.full((height // 2, width), 128, dtype=torch.uint8, device=device)
    return torch.cat([y, uv]).contiguous(), torch.cat([b, uv]).contiguous()


def run(device: int = 0, width: int = 512, height: int = 288) -> Result:
    require_bare_gpu(device)
    torch.cuda.set_device(device)
    torch.zeros(1, device=device)  # force the CUDA context current on this thread
    dx, dy = 5, 3
    metrics: dict = {}
    notes: list[str] = []

    lib = ctypes.CDLL("libnvidia-opticalflow.so.1")
    maxver = ctypes.c_uint32(0)
    if lib.NvOFGetMaxSupportedApiVersion(ctypes.byref(maxver)) != 0:
        raise _OfaError("NvOFGetMaxSupportedApiVersion failed")
    metrics["api_version"] = f"{maxver.value >> 4}.{maxver.value & 0xF}"

    # The driver's newer ABIs extend NV_OF_INIT_PARAMS (e.g. predDirection);
    # this file implements the 2.0 layouts, so request exactly that ABI.
    _ABI_20 = 0x20
    fl = _ApiFunctionList()
    st = lib.NvOFAPICreateInstanceCuda(ctypes.c_uint32(min(maxver.value, _ABI_20)), ctypes.byref(fl))
    if st != 0:
        raise _OfaError(f"NvOFAPICreateInstanceCuda: {_STATUS_NAMES.get(st, st)}")

    cuda_lib = ctypes.CDLL("libcuda.so.1")
    ctx = ctypes.c_void_p()
    cuda_lib.cuCtxGetCurrent(ctypes.byref(ctx))
    if not ctx.value:
        raise _OfaError("no current CUDA context after torch init")

    cudart = load_cudart()

    session = ctypes.c_void_p()
    st = fl.createOpticalFlow(ctx, ctypes.byref(session))
    if st != 0:
        raise _OfaError(f"createOpticalFlow: {_STATUS_NAMES.get(st, st)}")

    buf_in = ctypes.c_void_p()
    buf_ref = ctypes.c_void_p()
    buf_out = ctypes.c_void_p()
    try:
        init = _InitParams(
            width,
            height,
            _NV_OF_OUTPUT_VECTOR_GRID_SIZE_1,
            _NV_OF_HINT_VECTOR_GRID_SIZE_1,
            _NV_OF_MODE_OPTICALFLOW,
            _NV_OF_PERF_LEVEL_FAST,
            0,
            0,
            None,
            0,
            0,
        )
        st = fl.init(session, ctypes.byref(init))
        if st != 0:
            raise _OfaError(f"init: {_err(session, fl, st)}")

        desc_in = _BufferDescriptor(width, height, _NV_OF_BUFFER_USAGE_INPUT, _NV_OF_BUFFER_FORMAT_NV12)
        desc_out = _BufferDescriptor(
            width, height, _NV_OF_BUFFER_USAGE_OUTPUT, _NV_OF_BUFFER_FORMAT_SHORT2
        )
        st = fl.createGPUBuffer(session, ctypes.byref(desc_in), _NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, ctypes.byref(buf_in))
        st += fl.createGPUBuffer(session, ctypes.byref(desc_in), _NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, ctypes.byref(buf_ref))
        st += fl.createGPUBuffer(session, ctypes.byref(desc_out), _NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, ctypes.byref(buf_out))
        if st != 0:
            raise _OfaError(f"createGPUBuffer: {_err(session, fl, st)}")

        frame_a, frame_b = _synthetic_nv12(width, height, device)
        _memcpy(cudart, fl.gpuBufferGetCUdeviceptr(buf_in), frame_a.data_ptr(), frame_a.numel(), _MEMCPY_D2D)
        _memcpy(cudart, fl.gpuBufferGetCUdeviceptr(buf_ref), frame_b.data_ptr(), frame_b.numel(), _MEMCPY_D2D)

        in_params = _ExecuteInputParams(
            buf_in, buf_ref, None, 1, 0, None, 0, 0, None
        )
        out_params = _ExecuteOutputParams(buf_out, None, None)
        st = fl.execute(session, ctypes.byref(in_params), ctypes.byref(out_params))
        if st != 0:
            raise _OfaError(f"execute: {_err(session, fl, st)}")
        torch.cuda.synchronize()

        out_ptr = fl.gpuBufferGetCUdeviceptr(buf_out)
        host = np.empty(height * width * 2, dtype=np.int16)
        _memcpy(
            cudart,
            host.ctypes.data,
            out_ptr,
            host.nbytes,
            _MEMCPY_D2H,
        )
        flow = host.reshape(height, width, 2).astype(np.float32) / 32.0  # S10.5

        m = 48  # ignore wrap/edge artifacts
        center = flow[m : height - m, m : width - m]
        medx = float(np.median(center[..., 0]))
        medy = float(np.median(center[..., 1]))
        spread_x = float(np.quantile(np.abs(center[..., 0] - medx), 0.99))
        spread_y = float(np.quantile(np.abs(center[..., 1] - medy), 0.99))
        metrics["flow_median"] = [round(medx, 3), round(medy, 3)]
        metrics["flow_spread_p99"] = [round(spread_x, 3), round(spread_y, 3)]

        # The API's sign convention is not assumed: a healthy engine returns a
        # spatially consistent field of exactly the applied shift in one sign.
        mag_ok = abs(abs(medx) - dx) <= 0.75 and abs(abs(medy) - dy) <= 0.75
        spread_ok = spread_x <= 1.5 and spread_y <= 1.5
        if not mag_ok:
            notes.append(f"flow magnitude {medx:.2f},{medy:.2f} != shift {dx},{dy}")
        if not spread_ok:
            notes.append("flow field inconsistent across the frame")
        passed = mag_ok and spread_ok
    finally:
        for b in (buf_in, buf_ref, buf_out):
            if b.value:
                fl.destroyGPUBuffer(b)
        if session.value:
            fl.destroy(session)

    return Result(name="ofa", passed=passed, metrics=metrics, notes=notes)


def _err(session, fl, st: int) -> str:
    buf = ctypes.create_string_buffer(80)
    size = ctypes.c_uint32(80)
    fl.getLastError(session, buf, ctypes.byref(size))
    detail = buf.value.decode(errors="replace") if buf.value else ""
    return f"{_STATUS_NAMES.get(st, st)} {detail}".strip()
