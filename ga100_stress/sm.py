"""SM coverage and compute-correctness sweep (triton, inline PTX %smid)."""

from __future__ import annotations

import time

import numpy as np
import torch
import triton
import triton.language as tl

from .guard import require_bare_gpu
from .report import Result
from .telemetry import Sampler, pct_of_peak

_MASK32 = 0xFFFFFFFF
_LCG_ITERS = 128
_FMA_ITERS = 32
_RATE_CHAINS = 8  # independent FMA chains per thread: hides FMA latency
_RATE_ITERS = 1024
_RATE_BLOCK = 256
# GA100 non-tensor FMA peak per SM per clock (64 FP32 / 32 FP64 FMA lanes, 2 flops each)
_PEAK_PER_SM_CLK = {torch.float32: 128, torch.float64: 64}


def _lcg_ref(pids: np.ndarray, smids: np.ndarray, seed: int) -> np.ndarray:
    """Host replica of the kernel's integer checksum chain (exact, mod 2**32)."""
    x = (pids.astype(np.uint64) * 2654435761 + smids.astype(np.uint64) * 40503 + seed) & _MASK32
    for _ in range(_LCG_ITERS):
        x = (x * 1664525 + 1013904223) & _MASK32
    return x


def _fma_ref(x: np.ndarray, fp32_constants: bool) -> np.ndarray:
    """Host replica of the FMA chain; fp32 constants rounded like the GPU sees them."""
    c = np.float32(1.0000001).astype(np.float64) if fp32_constants else 1.0000001
    y = x.astype(np.float64).copy()
    for _ in range(_FMA_ITERS):
        y = y * c + x * 0.5
    return y


@triton.jit
def _smid_lcg_kernel(smid_ptr, cksum_ptr, seed):
    pid = tl.program_id(0).to(tl.int64)
    smid = tl.inline_asm_elementwise(
        asm="mov.u32 $0, %smid;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    smid64 = smid.to(tl.int64)
    x = (pid * 2654435761 + smid64 * 40503 + seed) & 0xFFFFFFFF
    for _ in tl.static_range(128):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
    tl.store(smid_ptr + pid, smid)
    tl.store(cksum_ptr + pid, x.to(tl.int32))


@triton.jit
def _fma_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x
    for _ in tl.static_range(32):
        y = y * 1.0000001 + x * 0.5
    tl.store(y_ptr + offs, y, mask=mask)


def _check_fma(dtype: torch.dtype, device: int, n: int, rtol: float) -> tuple[bool, float]:
    x = (torch.arange(n, dtype=torch.float64, device=device) % 1024 * 1e-3).to(dtype)
    y = torch.empty_like(x)
    grid = (triton.cdiv(n, 1024),)
    _fma_kernel[grid](x, y, n, BLOCK=1024)
    torch.cuda.synchronize()
    got = y.cpu().numpy().astype(np.float64)
    ref = _fma_ref(x.cpu().numpy(), fp32_constants=(dtype == torch.float32))
    rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)
    return bool(rel.max() < rtol), float(rel.max())


@triton.jit
def _fma_rate_kernel(out_ptr, iters, BLOCK: tl.constexpr):
    dt = out_ptr.dtype.element_ty
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = offs.to(dt) * 1e-7
    c = tl.full([BLOCK], 0.9999999, dt)
    d = tl.full([BLOCK], 1e-7, dt)
    a0 = x
    a1 = x + 1.0
    a2 = x + 2.0
    a3 = x + 3.0
    a4 = x + 4.0
    a5 = x + 5.0
    a6 = x + 6.0
    a7 = x + 7.0
    for _ in range(iters):
        a0 = tl.fma(a0, c, d)
        a1 = tl.fma(a1, c, d)
        a2 = tl.fma(a2, c, d)
        a3 = tl.fma(a3, c, d)
        a4 = tl.fma(a4, c, d)
        a5 = tl.fma(a5, c, d)
        a6 = tl.fma(a6, c, d)
        a7 = tl.fma(a7, c, d)
    tl.store(out_ptr + offs, a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7)


def _rate(dtype: torch.dtype, device: int, sms: int, min_s: float = 1.0) -> tuple[float, dict]:
    """Register-resident FMA throughput (TFLOPS) and the telemetry seen while it ran."""
    programs = sms * 16
    out = torch.empty(programs * _RATE_BLOCK, dtype=dtype, device=device)
    _fma_rate_kernel[(programs,)](out, _RATE_ITERS, BLOCK=_RATE_BLOCK)
    torch.cuda.synchronize()
    flops_per_launch = out.numel() * _RATE_CHAINS * _RATE_ITERS * 2
    reps = 0
    with Sampler(device) as s:
        t0 = time.perf_counter()
        while True:
            _fma_rate_kernel[(programs,)](out, _RATE_ITERS, BLOCK=_RATE_BLOCK)
            reps += 1
            if reps % 8 == 0:
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                if dt >= min_s:
                    break
    return flops_per_launch * reps / dt / 1e12, s


def run(device: int = 0, blocks: int = 4096) -> Result:
    require_bare_gpu(device)
    torch.cuda.set_device(device)
    metrics: dict = {}
    notes: list[str] = []

    # coverage + exact integer correctness
    seed = 0x5EEDC0DE
    smid = torch.empty(blocks, dtype=torch.int32, device=device)
    cksum = torch.empty(blocks, dtype=torch.int32, device=device)
    _smid_lcg_kernel[(blocks,)](smid, cksum, seed)
    torch.cuda.synchronize()
    smids_np = smid.cpu().numpy()
    got = cksum.cpu().numpy().view(np.uint32)
    expected = _lcg_ref(np.arange(blocks, dtype=np.uint64), smids_np.astype(np.uint64), seed)
    int_ok = bool((got == expected).all())
    unique = np.unique(smids_np)
    metrics["smids_observed"] = int(unique.size)
    metrics["smid_max"] = int(unique.max())
    metrics["int_checksum_ok"] = int_ok
    if not int_ok:
        bad = int((got != expected).sum())
        notes.append(f"{bad}/{blocks} blocks produced wrong integer checksums")

    # fp32 / fp64 correctness
    fp32_ok, rel32 = _check_fma(torch.float32, device, 1 << 22, rtol=1e-4)
    fp64_ok, rel64 = _check_fma(torch.float64, device, 1 << 22, rtol=1e-12)
    metrics["fp32_max_rel_err"] = rel32
    metrics["fp64_max_rel_err"] = rel64
    metrics["fp32_ok"] = fp32_ok
    metrics["fp64_ok"] = fp64_ok

    # throughput (reported, not asserted)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    telemetry = {}
    for dtype, key in ((torch.float32, "fp32"), (torch.float64, "fp64")):
        tflops, s = _rate(dtype, device, sms)
        metrics[f"{key}_tflops"] = round(tflops, 2)
        metrics[f"{key}_pct_of_peak_at_clock"] = pct_of_peak(
            tflops, _PEAK_PER_SM_CLK[dtype], sms, s.mean_sm_mhz()
        )
        telemetry[key] = s.summary()
    metrics["fp64_to_fp32_ratio"] = round(metrics["fp64_tflops"] / metrics["fp32_tflops"], 3)
    metrics["telemetry"] = telemetry
    if metrics["fp64_to_fp32_ratio"] < 0.25:
        notes.append(
            f"fp64 FMA runs at {metrics['fp64_to_fp32_ratio']} of the fp32 rate "
            f"(GA100 design ratio 0.5)"
        )

    passed = int_ok and fp32_ok and fp64_ok
    return Result(name="sm", passed=passed, metrics=metrics, notes=notes)
