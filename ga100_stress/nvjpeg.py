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

    # Smooth gradients, not noise: random noise is the worst case for IDCT and
    # chroma-upsampling differences between nvjpeg and libjpeg-turbo.
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    base = 128 + 60 * torch.sin(xs / 17) * torch.cos(ys / 23)
    imgs = torch.stack(
        [
            torch.stack(
                [
                    (base + 40 * torch.sin(xs / 31 + i * 0.7)).clamp(0, 255),
                    (base + 40 * torch.cos(ys / 29 + i * 0.5)).clamp(0, 255),
                    (base + i).clamp(0, 255),
                ]
            ).to(torch.uint8)
            for i in range(count)
        ]
    )
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
