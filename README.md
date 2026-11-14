# ga100-stress

Stress and verification suite for GA100-class datacenter GPUs (A100 and
derivatives). Pure userspace: it drives the GPU through its public APIs —
CUDA via torch/triton, NVENC/NVDEC via PyNvVideoCodec, NVJPEG via
torchvision, optical flow via nvofapi, GL via EGL — and reports which units
work.

The suite measures and reports; it never asserts a particular card's unit
counts. A card passes when every unit it exposes computes correctly.

## Subcommands

| Subcommand | Target | Pass rule |
|---|---|---|
| `sm` | SM coverage, FP32/FP64/INT compute | every observed SM executes and computes correctly; distinct-SM count reported |
| `mem` | HBM capacity and integrity | multi-pattern fill/verify at ~95% capacity (auto-backs off to max viable), ECC uncorrected delta 0 |
| `nvdec` | NVDEC engines | N concurrent decode sessions of an on-device-encoded stream; per-frame hashes identical across sessions, PSNR vs source above threshold |
| `nvjpeg` | NVJPEG engine | device JPEG decode matches CPU reference pixels |
| `ofa` | optical flow accelerator | recovered flow field matches a known synthetic translation |
| `gl` | GL via EGL (headless) | FBO render checksum matches CPU reference, CUDA↔GL interop round-trips |
| `all` | everything above | one verdict per subcommand plus JSON summary |

## Usage

```sh
ga100-stress all                 # run everything, human-readable summary
ga100-stress sm --blocks 8192    # finer SM coverage sweep
ga100-stress mem --fraction 0.9  # cap allocation at 90%
ga100-stress all --json out.json # machine-readable verdict
```

Exit code is 0 iff every run subcommand passed. `--json -` writes the JSON to
stdout.

## MIG

Full-card coverage claims only hold on the bare device. The suite refuses to
run while MIG mode is enabled:

```sh
sudo nvidia-smi -i 0 -mig 0   # destroys instances and disables MIG mode
```

Re-enable MIG and recreate instances afterwards if needed.

## Requirements

- Linux, NVIDIA driver with GSP firmware, GA100-class GPU
- Python ≥ 3.10 with torch, triton, PyNvVideoCodec, torchvision
- `libEGL` (NVIDIA vendor EGL) for `gl`
- `libnvidia-opticalflow.so.1` for `ofa`

## License

MIT — see [LICENSE](LICENSE).
