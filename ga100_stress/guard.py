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
    """Refuse to run under MIG: full-card coverage only holds on the bare device."""
    info = gpu_info(index)
    if info["mig_mode"].lower() != "disabled":
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
            "ecc.mode.current",
            "ecc.errors.corrected.aggregate.total",
            "ecc.errors.uncorrected.aggregate.total",
        ],
        index,
    )[0]
    if row[0].lower() != "enabled":
        return None
    return {"corrected": int(row[1]), "uncorrected": int(row[2])}
