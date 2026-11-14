"""Memory capacity, pattern integrity, ECC delta, and bandwidth."""

from __future__ import annotations

import time

import numpy as np
import torch

from .guard import ecc_totals, require_bare_gpu
from .report import Result

CHUNK_BYTES = 1 << 28  # 256 MiB
MIN_TAIL_BYTES = 1 << 24  # 16 MiB
PATTERNS = (0x00000000, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555, 0xCCCCCCCC, 0x33333333)
RANDOM_KIND = 15  # every 16th chunk is random-filled
SEED = 0xC0FFEE
PROBE_ELEMS = 1 << 18  # 1 MiB content-verified slice per random source


def _signed(p: int) -> int:
    return p - (1 << 32) if p >= (1 << 31) else p


def _allocate(device: int, total: int, fraction: float, floor: float):
    """Allocate ~fraction of total memory in chunks, backing off on OOM."""
    frac = fraction
    while True:
        target = int(total * frac)
        tensors: list[torch.Tensor] = []
        allocated = 0
        try:
            while allocated + CHUNK_BYTES <= target:
                tensors.append(torch.empty(CHUNK_BYTES // 4, dtype=torch.int32, device=device))
                allocated += CHUNK_BYTES
            rem = target - allocated
            if rem >= MIN_TAIL_BYTES:
                tensors.append(torch.empty(rem // 4, dtype=torch.int32, device=device))
                allocated += (rem // 4) * 4
            return tensors, allocated, frac
        except torch.cuda.OutOfMemoryError:
            del tensors
            torch.cuda.empty_cache()
            if frac - 0.02 < floor:
                raise
            frac = round(frac - 0.02, 2)


def run(device: int = 0, fraction: float = 0.95, floor: float = 0.85) -> Result:
    require_bare_gpu(device)
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    metrics: dict = {}
    notes: list[str] = []

    ecc0 = ecc_totals(device)

    try:
        tensors, allocated, frac = _allocate(device, total, fraction, floor)
    except torch.cuda.OutOfMemoryError:
        return Result(
            name="memory",
            passed=False,
            metrics={"fraction_requested": fraction, "fraction_floor": floor},
            notes=[f"could not allocate even {floor:.0%} of {total / 2**30:.1f} GiB"],
        )
    metrics["allocated_gib"] = round(allocated / 2**30, 2)
    metrics["fraction"] = frac
    n_chunks = len(tensors)

    # Fill: fixed patterns, plus one host-seeded random source copied to random chunks.
    src_idx = next((i for i in range(n_chunks) if i % 16 == RANDOM_KIND), -1)
    src_ref = None
    t0 = time.perf_counter()
    for i, t in enumerate(tensors):
        kind = i % 16
        if i == src_idx:
            rng = np.random.default_rng(SEED + i)
            src_ref = rng.integers(-(2**31), 2**31 - 1, size=t.numel(), dtype=np.int32)
            t.copy_(torch.from_numpy(src_ref).to(device))
        elif kind == RANDOM_KIND:
            t.copy_(tensors[src_idx])
        else:
            t.fill_(_signed(PATTERNS[kind % len(PATTERNS)]))
    torch.cuda.synchronize()
    fill_s = time.perf_counter() - t0

    # Verify.
    ok = True
    t0 = time.perf_counter()
    for i, t in enumerate(tensors):
        kind = i % 16
        if i == src_idx:
            n = min(PROBE_ELEMS, t.numel())
            if not np.array_equal(src_ref[:n], t[:n].cpu().numpy()):
                ok = False
                notes.append(f"random source chunk {i}: host probe mismatch")
        elif kind == RANDOM_KIND:
            if bool((t != tensors[src_idx]).any()):
                ok = False
                notes.append(f"random chunk {i}: mismatch vs source")
        else:
            p = _signed(PATTERNS[kind % len(PATTERNS)])
            if bool((t != p).any()):
                ok = False
                notes.append(f"pattern chunk {i} (0x{PATTERNS[kind % len(PATTERNS)]:08X}): mismatch")
    torch.cuda.synchronize()
    verify_s = time.perf_counter() - t0

    metrics["fill_gbps"] = round(allocated / fill_s / 1e9, 1)
    metrics["verify_gbps"] = round(allocated / verify_s / 1e9, 1)
    metrics["chunks"] = n_chunks

    if ecc0 is not None:
        ecc1 = ecc_totals(device)
        d_corr = ecc1["corrected"] - ecc0["corrected"]
        d_unc = ecc1["uncorrected"] - ecc0["uncorrected"]
        metrics["ecc_corrected_delta"] = d_corr
        metrics["ecc_uncorrected_delta"] = d_unc
        if d_unc != 0:
            ok = False
            notes.append(f"uncorrected ECC errors during test: {d_unc}")
    else:
        metrics["ecc_corrected_delta"] = None
        metrics["ecc_uncorrected_delta"] = None
        notes.append("ECC disabled; delta check skipped")

    return Result(name="memory", passed=ok, metrics=metrics, notes=notes)
