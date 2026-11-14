"""NVJPEG verification: device JPEG decode vs CPU reference pixels."""

from __future__ import annotations

import time

import torch
from torchvision.io import decode_jpeg, encode_jpeg

from .guard import require_bare_gpu
from .report import Result

# nvjpeg and libjpeg-turbo IDCT implementations differ by a couple of LSBs;
# a broken NVJPEG produces grossly different pixels.
_MAX_ABS_DIFF = 4


def run(device: int = 0, count: int = 32, size: int = 512) -> Result:
    require_bare_gpu(device)
    torch.cuda.set_device(device)
    metrics: dict = {}
    notes: list[str] = []

    g = torch.Generator().manual_seed(0x5EED)
    imgs = torch.randint(0, 256, (count, 3, size, size), dtype=torch.uint8, generator=g)
    jpegs = [encode_jpeg(img, quality=90) for img in imgs]
    refs = torch.stack([decode_jpeg(j) for j in jpegs])

    t0 = time.perf_counter()
    outs = torch.stack([decode_jpeg(j, device=f"cuda:{device}") for j in jpegs])
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0

    diff = (outs.to(torch.int16) - refs.to(device, dtype=torch.int16)).abs()
    max_diff = int(diff.max())
    metrics["images"] = count
    metrics["size"] = size
    metrics["max_abs_diff"] = max_diff
    metrics["mean_abs_diff"] = round(float(diff.float().mean()), 4)
    metrics["decode_s"] = round(decode_s, 3)
    if max_diff > _MAX_ABS_DIFF:
        notes.append(f"device decode differs from CPU reference by up to {max_diff}")
    return Result(name="nvjpeg", passed=max_diff <= _MAX_ABS_DIFF, metrics=metrics, notes=notes)
