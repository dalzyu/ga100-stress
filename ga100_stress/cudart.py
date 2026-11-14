"""Locate and load the CUDA runtime for ctypes-based tests."""

from __future__ import annotations

import ctypes
import glob
import pathlib


def load_cudart() -> ctypes.CDLL:
    import torch

    tried: list[str] = []
    majors: list[str] = []
    if torch.version.cuda:
        majors.append(torch.version.cuda.split(".")[0])
    majors += ["13", "12"]
    for major in majors:
        soname = f"libcudart.so.{major}"
        tried.append(soname)
        try:
            return ctypes.CDLL(soname)
        except OSError:
            pass
    root = pathlib.Path(torch.__file__).parent
    patterns = (
        root / "lib" / "libcudart.so*",
        root.parent / "nvidia" / "cu*" / "lib" / "libcudart.so*",
        root.parent / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so*",
    )
    for pat in patterns:
        for p in sorted(glob.glob(str(pat))):
            return ctypes.CDLL(p)
    raise RuntimeError(f"libcudart.so not found (tried {', '.join(tried)} and torch paths)")
