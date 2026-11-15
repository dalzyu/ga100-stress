"""Tensor-core GEMM correctness and throughput: fp16, bf16, tf32, int8, fp64."""

from __future__ import annotations

import contextlib
import time

import torch

from .guard import require_bare_gpu
from .report import Result
from .telemetry import Sampler, pct_of_peak

# GA100 dense tensor-core peak per SM per clock (A100 whitepaper: 312 / 312 / 156 /
# 624 / 19.5 TFLOPS-or-TOPS at 108 SMs, 1410 MHz).
PEAK_PER_SM_CLK = {"fp16": 2048, "bf16": 2048, "tf32": 1024, "int8": 4096, "fp64": 128}
KINDS = tuple(PEAK_PER_SM_CLK)
_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "tf32": torch.float32,
    "int8": torch.int8,
    "fp64": torch.float64,
}
_SEED = 0x7E450
_REF_ROWS = 64


@contextlib.contextmanager
def matmul_flags():
    """fp32 GEMMs on the TF32 tensor path; full-precision split-K reductions."""
    m = torch.backends.cuda.matmul
    saved = (
        m.fp32_precision,
        m.allow_fp16_reduced_precision_reduction,
        m.allow_bf16_reduced_precision_reduction,
    )
    m.fp32_precision = "tf32"
    m.allow_fp16_reduced_precision_reduction = False
    m.allow_bf16_reduced_precision_reduction = False
    try:
        yield
    finally:
        (
            m.fp32_precision,
            m.allow_fp16_reduced_precision_reduction,
            m.allow_bf16_reduced_precision_reduction,
        ) = saved


def operands(kind: str, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast to `kind`'s input dtype. int8 B goes column-major: cuBLASLt's IMMA fast path
    (row-major B falls back to a kernel ~5x slower)."""
    a, b = a.to(_DTYPE[kind]), b.to(_DTYPE[kind])
    if kind == "int8":
        b = b.t().contiguous().t()
    return a, b


def gemm(kind: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch._int_mm(a, b) if kind == "int8" else a @ b


def random_base(kind: str, n: int, device: int, gen: torch.Generator):
    """Full-range operands, so throughput and power reflect real data."""
    if kind == "int8":
        mk = lambda: torch.randint(-128, 128, (n, n), dtype=torch.int8, device=device, generator=gen)  # noqa: E731
    else:
        mk = lambda: torch.randn(n, n, device=device, generator=gen)  # noqa: E731
    return mk(), mk()


def _check_exact(n: int, device: int) -> dict[str, dict]:
    """Integer operands in {-1, 0, 1}: every product and partial sum is exact in every
    format, so each output must equal the exact result rounded once to the output dtype.

    The reference covers one row from every 128-row band (every output tile row) across all
    columns; a second run of the same GEMM must match the first over the whole matrix.
    """
    gen = torch.Generator().manual_seed(_SEED)
    a_cpu = torch.randint(-1, 2, (n, n), dtype=torch.int8, generator=gen)
    b_cpu = torch.randint(-1, 2, (n, n), dtype=torch.int8, generator=gen)
    band = n // _REF_ROWS
    idx = torch.arange(_REF_ROWS)
    rows = idx * band + (idx * 37) % band
    ref = a_cpu[rows].double() @ b_cpu.double()  # exact: |entries| <= n < 2**53
    a, b = a_cpu.to(device), b_cpu.to(device)
    rows_dev = rows.to(device)
    out: dict[str, dict] = {}
    for kind in KINDS:
        x, y = operands(kind, a, b)
        first = gemm(kind, x, y)
        second = gemm(kind, x, y)
        got = first[rows_dev].cpu()
        bad = int((got != ref.to(got.dtype)).sum())
        out[kind] = {"mismatches": bad, "repeatable": bool(torch.equal(first, second))}
        del x, y, first, second
    return out


def _throughput(kind: str, n: int, device: int, min_s: float) -> tuple[float, Sampler]:
    gen = torch.Generator(device=device).manual_seed(_SEED)
    x, y = operands(kind, *random_base(kind, n, device, gen))
    gemm(kind, x, y)
    torch.cuda.synchronize()
    reps = 0
    with Sampler(device) as s:
        t0 = time.perf_counter()
        while True:
            gemm(kind, x, y)
            reps += 1
            if reps % 4 == 0:
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                if dt >= min_s:
                    break
    return 2 * n**3 * reps / dt / 1e12, s


def run(device: int = 0, size: int = 8192, seconds: float = 2.0) -> Result:
    require_bare_gpu(device)
    torch.cuda.set_device(device)
    if size % 128:
        raise ValueError("size must be a multiple of 128")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    metrics: dict = {"size": size, "sms": sms}
    notes: list[str] = []
    telemetry: dict = {}
    passed = True

    with matmul_flags():
        for kind, r in _check_exact(size, device).items():
            metrics[f"{kind}_exact"] = r["mismatches"] == 0
            metrics[f"{kind}_repeatable"] = r["repeatable"]
            if r["mismatches"]:
                passed = False
                notes.append(f"{kind}: {r['mismatches']} reference elements wrong")
            if not r["repeatable"]:
                passed = False
                notes.append(f"{kind}: two runs of the same GEMM differ")

        # throughput (reported, not asserted)
        for kind in KINDS:
            tflops, s = _throughput(kind, size, device, seconds)
            pct = pct_of_peak(tflops, PEAK_PER_SM_CLK[kind], sms, s.mean_sm_mhz())
            metrics[f"{kind}_tflops"] = round(tflops, 1)
            metrics[f"{kind}_pct_of_peak_at_clock"] = pct
            telemetry[kind] = s.summary()
            torch.cuda.empty_cache()

    metrics["telemetry"] = telemetry
    pcts = {k: metrics[f"{k}_pct_of_peak_at_clock"] for k in KINDS}
    best = max((p for p in pcts.values() if p), default=None)
    for kind, pct in pcts.items():
        if pct is not None and best and pct < 0.5 * best:
            notes.append(
                f"{kind}: {pct}% of GA100 peak at the observed clock vs {best}% for the "
                f"best path; this unit runs well below the others"
            )
    for kind, t in telemetry.items():
        if t.get("clock_event_reasons"):
            notes.append(
                f"{kind}: SM {t['sm_mhz_min']}-{t['sm_mhz_max']} MHz (max {t['sm_max_mhz']}), "
                f"{t['power_w_max']} W of {t['power_limit_w']} W, "
                f"clock events {', '.join(t['clock_event_reasons'])}"
            )
    return Result(name="tensor", passed=passed, metrics=metrics, notes=notes)
