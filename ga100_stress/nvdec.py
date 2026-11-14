"""NVDEC verification: on-device encode (NVENC), N parallel decode sessions."""

from __future__ import annotations

import ctypes
import hashlib
import threading
import time

import numpy as np
import torch

import PyNvVideoCodec as nvc

from .guard import require_bare_gpu
from .report import Result

WIDTH, HEIGHT = 1280, 720
PSNR_MIN_DB = 30.0


def _source_y(n: int, device: int) -> np.ndarray:
    ys, xs = torch.meshgrid(
        torch.arange(HEIGHT, device=device, dtype=torch.float32),
        torch.arange(WIDTH, device=device, dtype=torch.float32),
        indexing="ij",
    )
    frames = torch.stack(
        [
            (128 + 60 * torch.sin(xs / 24 + i * 0.35) * torch.cos(ys / 24))
            .clamp(0, 255)
            .to(torch.uint8)
            for i in range(n)
        ]
    )
    return frames.cpu().numpy()  # (n, H, W)


def _encode(frames: list[torch.Tensor]) -> list[bytes]:
    enc = nvc.CreateEncoder(WIDTH, HEIGHT, "NV12", False, codec="h264", bitrate="20M")
    packets = []
    for f in frames:
        packets += enc.Encode(f)
    packets += enc.EndEncode()
    return [bytes(p["data"]) for p in packets if p["data"]]


def _y_plane(frame, w: int, h: int) -> np.ndarray:
    mv = frame.cuda()
    stride = mv.strides[0] if mv.strides and mv.strides[0] else mv.shape[1]
    raw = bytes(mv)
    return np.frombuffer(raw, dtype=np.uint8)[: h * stride].reshape(h, stride)[:, :w]


def _decode_session(packets, gpuid: int, out: dict, idx: int, keep_pixels: bool) -> None:
    try:
        dec = nvc.CreateDecoder(
            gpuid=gpuid, codec=nvc.cudaVideoCodec.H264, usedevicememory=False
        )
        hashes: list[str] = []
        pixels: list[np.ndarray] = []
        for i, pkt in enumerate(packets):
            buf = ctypes.create_string_buffer(pkt)
            pd = nvc.PacketData()
            pd.bsl_data = ctypes.addressof(buf)
            pd.bsl = len(pkt)
            pd.pts = i
            for frame in dec.Decode(pd):
                mv = frame.cuda()
                hashes.append(hashlib.sha256(bytes(mv)).hexdigest())
                if keep_pixels:
                    pixels.append(_y_plane(frame, WIDTH, HEIGHT).copy())
        out[idx] = (hashes, pixels)
    except Exception as e:  # noqa: BLE001 - session failure is the result
        out[idx] = e


def run(device: int = 0, sessions: int = 5, frames: int = 64) -> Result:
    require_bare_gpu(device)
    torch.cuda.set_device(device)
    metrics: dict = {}
    notes: list[str] = []

    src = _source_y(frames, device)
    uv = torch.full((HEIGHT // 2, WIDTH), 128, dtype=torch.uint8, device=device)
    nv12 = [torch.cat([torch.from_numpy(s).to(device), uv]) for s in src]
    packets = _encode(nv12)
    metrics["packets"] = len(packets)

    out: dict = {}
    threads = [
        threading.Thread(target=_decode_session, args=(packets, device, out, i, i == 0))
        for i in range(sessions)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    decode_s = time.perf_counter() - t0
    metrics["sessions"] = sessions
    metrics["decode_s"] = round(decode_s, 2)

    errors = [v for v in out.values() if isinstance(v, Exception)]
    for e in errors:
        notes.append(f"session failed: {type(e).__name__}: {e}")
    hash_lists = [v[0] for v in out.values() if not isinstance(v, Exception)]

    hash_ok = len(errors) == 0 and len(hash_lists) == sessions
    if hash_ok:
        n_decoded = len(hash_lists[0])
        metrics["frames_decoded"] = n_decoded
        hash_ok = n_decoded > 0 and all(h == hash_lists[0] for h in hash_lists)
        if not hash_ok:
            notes.append("decode sessions disagree on frame hashes")
    else:
        metrics["frames_decoded"] = 0

    psnr_min = psnr_mean = None
    if len(errors) == 0 and out.get(0) is not None and len(out[0][1]) == len(src):
        psnrs = []
        for ref, got in zip(src, out[0][1]):
            mse = float(np.mean((ref.astype(np.float64) - got.astype(np.float64)) ** 2))
            psnrs.append(99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse))
        psnr_min = round(min(psnrs), 2)
        psnr_mean = round(float(np.mean(psnrs)), 2)
        metrics["psnr_min_db"] = psnr_min
        metrics["psnr_mean_db"] = psnr_mean
        if psnr_min < PSNR_MIN_DB:
            notes.append(f"worst-frame PSNR {psnr_min} dB below {PSNR_MIN_DB} dB")
    elif not errors:
        notes.append("decoded frame count does not match source")

    passed = hash_ok and not errors and psnr_min is not None and psnr_min >= PSNR_MIN_DB
    return Result(name="nvdec", passed=passed, metrics=metrics, notes=notes)
