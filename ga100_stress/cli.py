"""Command line interface."""

from __future__ import annotations

import argparse
import sys
import time

from . import __version__
from . import gl, mem, nvdec, nvjpeg, ofa, sm
from .guard import GuardError, require_bare_gpu
from .report import Result, json_report, print_summary

MODULES = {
    "sm": sm,
    "mem": mem,
    "nvdec": nvdec,
    "nvjpeg": nvjpeg,
    "ofa": ofa,
    "gl": gl,
}


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", metavar="PATH", help="write verdict JSON to PATH ('-' = stdout)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ga100-stress",
        description="Stress and verification suite for GA100-class datacenter GPUs",
    )
    parser.add_argument("--device", type=int, default=0, help="GPU index (default 0)")
    _add_common(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sm", help="SM coverage and compute correctness (FP32/FP64/INT)")
    p.add_argument("--blocks", type=int, default=4096, help="blocks in the SMID sweep")
    _add_common(p)

    p = sub.add_parser("mem", help="memory capacity, patterns, ECC delta, bandwidth")
    p.add_argument("--fraction", type=float, default=0.95, help="fraction of memory to fill")
    p.add_argument("--floor", type=float, default=0.85, help="OOM back-off floor")
    _add_common(p)

    p = sub.add_parser("nvdec", help="NVDEC sessions over the committed synthetic clip")
    p.add_argument("--sessions", type=int, default=5, help="parallel decode sessions")
    _add_common(p)

    p = sub.add_parser("nvjpeg", help="NVJPEG device decode vs CPU reference")
    p.add_argument("--count", type=int, default=32, help="images to decode")
    _add_common(p)

    p = sub.add_parser("ofa", help="optical flow accelerator, known-translation test")
    _add_common(p)

    p = sub.add_parser("gl", help="headless EGL GL render + CUDA interop")
    _add_common(p)

    p = sub.add_parser("all", help="run every subcommand")
    p.add_argument("--blocks", type=int, default=4096)
    p.add_argument("--fraction", type=float, default=0.95)
    p.add_argument("--floor", type=float, default=0.85)
    p.add_argument("--sessions", type=int, default=5)
    p.add_argument("--count", type=int, default=32)
    _add_common(p)

    args = parser.parse_args(argv)

    try:
        gpu = require_bare_gpu(args.device)
    except GuardError as e:
        print(f"refusing to run: {e}", file=sys.stderr)
        return 2

    names = list(MODULES) if args.command == "all" else [args.command]
    results: list[Result] = []
    for name in names:
        module = MODULES[name]
        kw = {}
        if name == "mem":
            kw = {"fraction": args.fraction, "floor": args.floor}
        if name == "nvdec":
            kw = {"sessions": args.sessions}
        if name == "nvjpeg":
            kw = {"count": args.count}
        if name == "sm":
            kw = {"blocks": args.blocks}
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
