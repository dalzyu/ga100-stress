"""Command line interface."""

from __future__ import annotations

import argparse
import os
import sys
import time

# nvidia-smi numbers GPUs in PCI bus order; make CUDA (torch, triton, NVDEC) agree with it.
# Set before the test modules import those libraries.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from . import __version__  # noqa: E402
from . import burn, gl, mem, nvdec, nvjpeg, ofa, sm, tensor  # noqa: E402
from .guard import GuardError, require_bare_gpu  # noqa: E402
from .report import Result, json_report, print_summary  # noqa: E402

# subcommand -> (module, help, options); each option is (flag, type, default, help).
SUBCOMMANDS = {
    "sm": (sm, "SM coverage, FP32/FP64/INT correctness, FMA throughput", [
        ("--blocks", int, 4096, "blocks in the SMID sweep"),
    ]),
    "tensor": (tensor, "tensor-core GEMM exactness + throughput per dtype", [
        ("--size", int, 8192, "square GEMM size (multiple of 128)"),
    ]),
    "mem": (mem, "memory capacity, patterns, ECC delta, bandwidth", [
        ("--fraction", float, 0.95, "fraction of memory to fill"),
        ("--floor", float, 0.85, "OOM back-off floor"),
    ]),
    "nvdec": (nvdec, "NVDEC sessions over the committed synthetic clip", [
        ("--sessions", int, 5, "parallel decode sessions"),
    ]),
    "nvjpeg": (nvjpeg, "NVJPEG device decode vs CPU reference", [
        ("--count", int, 32, "images to decode"),
    ]),
    "ofa": (ofa, "optical flow accelerator, known-translation test", []),
    "gl": (gl, "headless EGL GL render + CUDA interop", []),
    "burn": (burn, "sustained tensor-core load, bit-exact repeatability", [
        ("--size", int, 8192, "square GEMM size (multiple of 128)"),
        ("--seconds", float, 60.0, "load duration"),
    ]),
}


def _add_options(p: argparse.ArgumentParser, options) -> None:
    for flag, typ, default, help_ in options:
        p.add_argument(flag, type=typ, default=default, help=f"{help_} (default {default})")
    p.add_argument("--json", metavar="PATH", help="write verdict JSON to PATH ('-' = stdout)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ga100-stress",
        description="Stress and verification suite for GA100-class datacenter GPUs",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--device", type=int, default=0, help="GPU index as nvidia-smi shows it")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_, options) in SUBCOMMANDS.items():
        _add_options(sub.add_parser(name, help=help_), options)
    every = {o[0]: o for _, _, options in SUBCOMMANDS.values() for o in options}
    _add_options(sub.add_parser("all", help="run every subcommand"), every.values())

    args = parser.parse_args(argv)

    try:
        gpu = require_bare_gpu(args.device)
    except GuardError as e:
        print(f"refusing to run: {e}", file=sys.stderr)
        return 2

    names = list(SUBCOMMANDS) if args.command == "all" else [args.command]
    results: list[Result] = []
    for name in names:
        module, _, options = SUBCOMMANDS[name]
        kw = {flag[2:]: getattr(args, flag[2:]) for flag, *_ in options}
        t0 = time.perf_counter()
        try:
            r = module.run(device=args.device, **kw)
        except Exception as e:  # noqa: BLE001 - a crashing test is a failed test
            r = Result(name=name, passed=False, notes=[f"{type(e).__name__}: {e}"])
        r.seconds = round(time.perf_counter() - t0, 2)
        results.append(r)
        print(f"[{r.name}] {'PASS' if r.passed else 'FAIL'} ({r.seconds}s)")
        for note in r.notes:
            print(f"    {note}")

    print()
    ok = print_summary(results)
    if args.json:
        doc = json_report(results, gpu, __version__)
        if args.json == "-":
            print(doc)
        else:
            with open(args.json, "w") as f:
                f.write(doc + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
