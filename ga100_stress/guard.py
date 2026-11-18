"""Environment guards and GPU introspection via nvidia-smi."""

from __future__ import annotations

import subprocess


class GuardError(RuntimeError):
    """The environment refuses the requested test."""


def _smi_rows(fields: list[str], index: int | None = None) -> list[list[str]]:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise GuardError(f"nvidia-smi failed: {proc.stderr.strip()}")
    rows = [[c.strip() for c in line.split(",")] for line in proc.stdout.strip().splitlines()]
    if index is not None:
        for row in rows:
            if int(row[0]) == index:
                return [row]
        raise GuardError(f"GPU {index} not found")
    return rows


def gpu_info(index: int = 0) -> dict:
    row = _smi_rows(
        ["index", "name", "driver_version", "mig.mode.current", "memory.total"], index
    )[0]
    return {
        "index": int(row[0]),
        "name": row[1],
        "driver": row[2],
        "mig_mode": row[3],
        "total_bytes": int(row[4]) * 1024**2,
    }


def require_bare_gpu(index: int = 0) -> dict:
    """Refuse to run under MIG: full-card coverage only holds on the bare device.

    A GPU without MIG support (nvidia-smi reports `[N/A]`) is always the bare device.
    """
    info = gpu_info(index)
    if info["mig_mode"].strip("[]").lower() not in ("disabled", "n/a", "not supported"):
        raise GuardError(
            f"MIG mode is {info['mig_mode']} on GPU {index}; full-card verification "
            f"requires the bare device. Tear down instances and disable MIG with "
            f"`nvidia-smi -i {index} -mig 0`, then re-run."
        )
    return info


def ecc_totals(index: int = 0) -> dict | None:
    """Aggregate ECC counters, or None when ECC is disabled."""
    row = _smi_rows(
        [
            "index",
            "ecc.mode.current",
            "ecc.errors.corrected.aggregate.total",
            "ecc.errors.uncorrected.aggregate.total",
        ],
        index,
    )[0]
    if row[1].lower() != "enabled":
        return None
    return {"corrected": int(row[2]), "uncorrected": int(row[3])}


def ecc_delta(index: int, before: dict | None, metrics: dict, notes: list[str]) -> bool:
    """Record ECC counter deltas since `before`; False iff uncorrected errors appeared."""
    if before is None:
        metrics["ecc_corrected_delta"] = None
        metrics["ecc_uncorrected_delta"] = None
        notes.append("ECC disabled; delta check skipped")
        return True
    after = ecc_totals(index)
    d_unc = after["uncorrected"] - before["uncorrected"]
    metrics["ecc_corrected_delta"] = after["corrected"] - before["corrected"]
    metrics["ecc_uncorrected_delta"] = d_unc
    if d_unc != 0:
        notes.append(f"uncorrected ECC errors during test: {d_unc}")
    return d_unc == 0
