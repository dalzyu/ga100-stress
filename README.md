# ga100-stress

Stress and verification suite for GA100-class datacenter GPUs (A100 and
derivatives). Pure userspace: it drives the GPU through its public APIs —
CUDA via torch/triton, NVDEC via PyNvVideoCodec, NVJPEG via
torchvision, optical flow via nvofapi, GL via EGL — and reports which units
work.

The suite measures and reports; it never asserts a particular card's unit
counts. A card passes when every unit it exposes computes correctly.

GA100 has no NVENC (the die has no encoder hardware), so there is
intentionally no encoder test.

## Subcommands

| Subcommand | Target | Pass rule |
|---|---|---|
| `sm` | SM coverage, FP32/FP64/INT compute | every observed SM executes and computes correctly; distinct-SM count reported |
| `tensor` | tensor cores: fp16, bf16, tf32, int8 (IMMA), fp64 (DMMA) | every format bit-exact vs an exact reference across every output tile row, and repeatable |
| `mem` | HBM capacity and integrity | multi-pattern fill/verify at ~95% capacity (auto-backs off to max viable), ECC uncorrected delta 0 |
| `nvdec` | NVDEC engines | N concurrent decode sessions of a committed synthetic clip; per-frame hashes identical across sessions, PSNR vs source above threshold |
| `nvjpeg` | NVJPEG engine | device JPEG decode matches CPU reference pixels |
| `ofa` | optical flow accelerator | recovered flow field matches a known synthetic translation |
| `gl` | GL via EGL (headless) | FBO render checksum matches CPU reference, CUDA↔GL interop round-trips |
| `burn` | sustained tensor-core load (all five formats, default 60 s) | every repeat bit-identical to its first result, ECC uncorrected delta 0 |
| `all` | everything above | one verdict per subcommand plus JSON summary |

## Throughput and telemetry

Throughput is reported, never asserted:

- `sm`: register-resident fp32/fp64 FMA rate (8 independent chains per thread).
- `tensor`: 8192³ GEMM rate per format. int8 uses cuBLASLt with a column-major B,
  the IMMA fast path.
- `mem`: device-to-device copy bandwidth (read + write bytes).

While each rate is measured, a sampler polls `nvidia-smi` for SM clock, power, temperature,
and clock-event reasons. Each rate is also reported as `*_pct_of_peak_at_clock`: the
measured rate as a percentage of the GA100 per-SM, per-clock peak at the clock actually
observed. A unit that is throttled or fused slow shows a low percentage even at full clock.
A power- or clock-capped card shows normal percentages at a low MHz. The verdict notes
flag any format far below the best one, and any active clock-event reasons.

## Usage

```sh
ga100-stress all                 # run everything, human-readable summary
ga100-stress sm --blocks 8192    # finer SM coverage sweep
ga100-stress tensor --size 4096  # smaller GEMMs
ga100-stress burn --seconds 600  # 10-minute sustained load
ga100-stress mem --fraction 0.9  # cap allocation at 90%
ga100-stress all --json out.json # machine-readable verdict
```

The `nvdec` test uses a committed synthetic clip (`ga100_stress/data/`);
regenerate it with `tools/gen_nvdec_vector.py` (needs the `dev` extra).

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
