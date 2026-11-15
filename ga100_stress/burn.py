"""Sustained tensor-core load: every repeated GEMM must reproduce its first result bit-exactly.

Each round gives every format an equal time slice, so a slow unit (one GEMM can take
seconds) does not starve the fast ones of load.
"""

from __future__ import annotations

import time

import torch

from .guard import ecc_delta, ecc_totals, require_bare_gpu
from .report import Result
from .telemetry import Sampler
from .tensor import KINDS, gemm, matmul_flags, operands, random_base

_SEED = 0xB0B


def run(device: int = 0, size: int = 8192, seconds: float = 60.0) -> Result:
    require_bare_gpu(device)
    torch.cuda.set_device(device)
    metrics: dict = {"size": size, "target_s": seconds}
    notes: list[str] = []
    ecc0 = ecc_totals(device)
    gen = torch.Generator(device=device).manual_seed(_SEED)

    with matmul_flags():
        work = {}
        one_s = {}
        for kind in KINDS:
            x, y = operands(kind, *random_base(kind, size, device, gen))
            gemm(kind, x, y)  # warm-up / algorithm selection
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ref = gemm(kind, x, y)
            torch.cuda.synchronize()
            one_s[kind] = time.perf_counter() - t0
            work[kind] = (x, y, ref)
        slice_s = max(1.0, *one_s.values())
        reps = {k: max(1, int(slice_s / t)) for k, t in one_s.items()}

        iters = dict.fromkeys(KINDS, 0)
        bad = dict.fromkeys(KINDS, 0)
        with Sampler(device, period=1.0) as s:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < seconds:
                for kind, (x, y, ref) in work.items():
                    for _ in range(reps[kind]):
                        if not torch.equal(gemm(kind, x, y), ref):
                            bad[kind] += 1
                    iters[kind] += reps[kind]
            elapsed = time.perf_counter() - t0

    metrics["elapsed_s"] = round(elapsed, 1)
    metrics["slice_s"] = round(slice_s, 2)
    metrics["iterations"] = iters
    metrics["mismatches"] = bad
    t = metrics["telemetry"] = s.summary()
    if t.get("clock_event_reasons"):
        notes.append(
            f"SM {t['sm_mhz_min']}-{t['sm_mhz_max']} MHz (mean {t['sm_mhz_mean']}, "
            f"max {t['sm_max_mhz']}), {t['power_w_max']} W of {t['power_limit_w']} W, "
            f"clock events {', '.join(t['clock_event_reasons'])}"
        )
    passed = not any(bad.values())
    for kind, n in bad.items():
        if n:
            notes.append(f"{kind}: {n}/{iters[kind]} iterations differ from the first result")
    passed = ecc_delta(device, ecc0, metrics, notes) and passed
    return Result(name="burn", passed=passed, metrics=metrics, notes=notes)
