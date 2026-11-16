"""Clock, power, and clock-event-reason sampling while a load runs (nvidia-smi)."""

from __future__ import annotations

import subprocess
import threading

_FIELDS = (
    "clocks.sm",
    "clocks.max.sm",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks_event_reasons.active",
)

# nvmlClocksEventReason* bits
_REASONS = {
    0x1: "idle",
    0x2: "applications_clocks",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
    0x100: "display_clock",
}


def _num(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def pct_of_peak(rate_tflops: float, flop_per_sm_clk: int, sms: int, mhz: float | None) -> float | None:
    """Measured rate as % of the per-SM-per-clock peak at the clock actually observed.

    Separates a slow unit (low %) from a low clock (normal %, low MHz).
    """
    if not mhz:
        return None
    peak = flop_per_sm_clk * sms * mhz * 1e6 / 1e12
    return round(100.0 * rate_tflops / peak, 1)


class Sampler:
    """Polls nvidia-smi in a background thread for the duration of a `with` block."""

    def __init__(self, index: int = 0, period: float = 0.25):
        self.index = index
        self.period = period
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> "Sampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def _loop(self) -> None:
        cmd = [
            "nvidia-smi",
            "-i",
            str(self.index),
            f"--query-gpu={','.join(_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and proc.stdout.strip():
                c = [x.strip() for x in proc.stdout.strip().splitlines()[0].split(",")]
                try:
                    reasons = int(c[5], 16)
                except ValueError:
                    reasons = 0
                self.samples.append(
                    {
                        "sm_mhz": _num(c[0]),
                        "sm_max_mhz": _num(c[1]),
                        "power_w": _num(c[2]),
                        "power_limit_w": _num(c[3]),
                        "temp_c": _num(c[4]),
                        "reasons": reasons,
                    }
                )
            self._stop.wait(self.period)

    def mean_sm_mhz(self) -> float | None:
        clocks = [s["sm_mhz"] for s in self.samples if s["sm_mhz"]]
        return sum(clocks) / len(clocks) if clocks else None

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0}

        def pick(key, fn):
            vals = [s[key] for s in self.samples if s[key] is not None]
            return fn(vals) if vals else None

        mask = 0
        for s in self.samples:
            mask |= s["reasons"]
        mean = self.mean_sm_mhz()
        return {
            "samples": len(self.samples),
            "sm_mhz_min": pick("sm_mhz", min),
            "sm_mhz_mean": round(mean) if mean else None,
            "sm_mhz_max": pick("sm_mhz", max),
            "sm_max_mhz": pick("sm_max_mhz", max),
            "power_w_max": pick("power_w", max),
            "power_limit_w": pick("power_limit_w", max),
            "temp_c_max": pick("temp_c", max),
            "clock_event_reasons": [n for b, n in _REASONS.items() if mask & b and b != 0x1],
        }
