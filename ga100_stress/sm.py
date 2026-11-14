"""SM coverage and compute-correctness sweep (triton, inline PTX %smid)."""

from __future__ import annotations

import time

import numpy as np
import torch
import triton
import triton.language as tl

from .guard import require_bare_gpu
from .report import Result

_MASK32 = 0xFFFFFFFF
_LCG_ITERS = 128
_FMA_ITERS = 32


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


def _rate(dtype: torch.dtype, device: int, n: int, reps: int) -> float:
    x = torch.linspace(0.5, 1.5, n, dtype=dtype, device=device)
    y = torch.empty_like(x)
    grid = (triton.cdiv(n, 1024),)
    _fma_kernel[grid](x, y, n, BLOCK=1024)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        _fma_kernel[grid](x, y, n, BLOCK=1024)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    flops = n * _FMA_ITERS * 3 * reps  # y*c, x*0.5, add
    return flops / dt / 1e9


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
    metrics["fp32_gflops"] = round(_rate(torch.float32, device, 1 << 28, reps=50), 1)
    metrics["fp64_gflops"] = round(_rate(torch.float64, device, 1 << 27, reps=20), 1)
    if metrics["fp32_gflops"] > 0:
        metrics["fp64_to_fp32_ratio"] = round(
            metrics["fp64_gflops"] / metrics["fp32_gflops"], 3
        )

    passed = int_ok and fp32_ok and fp64_ok
    return Result(name="compute", passed=passed, metrics=metrics, notes=notes)
