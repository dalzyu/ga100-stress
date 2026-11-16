"""Regenerate the committed NVDEC test clip.

Requires the `dev` extra (imageio-ffmpeg, which bundles a static ffmpeg).

The clip is synthetic: smooth sine gradients identical to
`ga100_stress.nvdec._source_frames`, encoded as H.264 1280x720 without
B-frames (so decode order == display order at zero decoder latency).
"""

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from ga100_stress.nvdec import FRAMES, HEIGHT, WIDTH, _source_frames

DATA = Path(__file__).resolve().parent.parent / "ga100_stress" / "data"


def main() -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    src = _source_frames(FRAMES)  # (n, H, W) uint8
    DATA.mkdir(parents=True, exist_ok=True)
    clip = DATA / "nvdec_test_1280x720.mp4"

    with tempfile.NamedTemporaryFile(suffix=".yuv") as raw:
        with open(raw.name, "wb") as f:
            for frame in src:
                f.write(frame.tobytes())
                f.write(b"\x80" * (WIDTH * HEIGHT // 2))  # neutral UV planes
        subprocess.run(
            [
                ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-s", f"{WIDTH}x{HEIGHT}", "-r", "30", "-i", raw.name,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-bf", "0",
                "-pix_fmt", "yuv420p", str(clip),
            ],
            check=True,
            capture_output=True,
        )
    print(f"wrote {clip} ({clip.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
