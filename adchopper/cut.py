"""Audio cutting: remove ad spans from an MP3 using ffmpeg.

We keep the complement of the ad spans and concatenate them in a single
ffmpeg pass using the atrim + concat filter. A short crossfade-free hard cut
is used; an optional tiny fade can be applied at each boundary to avoid
clicks.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import List, Optional

from .segments import AdSpan, keep_ranges


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def probe_duration(audio_path: str) -> Optional[float]:
    """Return media duration in seconds via ffprobe, or None if unavailable."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def _build_filter(ranges: List[tuple[float, float]], fade: float) -> str:
    parts = []
    labels = []
    for i, (start, end) in enumerate(ranges):
        label = f"a{i}"
        seg_filters = [
            f"atrim=start={start:.3f}:end={end:.3f}",
            "asetpts=PTS-STARTPTS",
        ]
        if fade > 0:
            dur = end - start
            f = min(fade, dur / 2)
            seg_filters.append(f"afade=t=in:st=0:d={f:.3f}")
            seg_filters.append(f"afade=t=out:st={dur - f:.3f}:d={f:.3f}")
        parts.append(f"[0:a]{','.join(seg_filters)}[{label}]")
        labels.append(f"[{label}]")
    concat = f"{''.join(labels)}concat=n={len(ranges)}:v=0:a=1[out]"
    return ";".join(parts + [concat])


def cut_ads(
    audio_path: str,
    ad_spans: List[AdSpan],
    output_path: str,
    total_duration: Optional[float] = None,
    fade: float = 0.02,
    bitrate: str = "128k",
    verbose: bool = True,
) -> None:
    """Write `output_path` with the ad spans removed.

    `total_duration` should be the true media duration; if omitted we probe it.
    """
    if not ffmpeg_available():
        raise SystemExit(
            "ffmpeg not found on PATH. Install it (e.g. `apt install ffmpeg` "
            "or `brew install ffmpeg`)."
        )

    if total_duration is None:
        total_duration = probe_duration(audio_path)
    if not total_duration:
        raise SystemExit(
            "Could not determine audio duration (install ffprobe, or pass it)."
        )

    ranges = keep_ranges(ad_spans, total_duration)
    if not ranges:
        raise SystemExit("Nothing left to keep -- aborting (check ad spans).")

    if not ad_spans:
        if verbose:
            print("[cut] no ads to remove; copying input to output.")
        shutil.copyfile(audio_path, output_path)
        return

    filter_complex = _build_filter(ranges, fade)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-b:a",
        bitrate,
        output_path,
    ]
    if verbose:
        removed = total_duration - sum(b - a for a, b in ranges)
        print(
            f"[cut] keeping {len(ranges)} segment(s), removing ~{removed:.0f}s "
            f"-> {output_path}"
        )
    result = subprocess.run(cmd, capture_output=not verbose)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed (exit {result.returncode}).")
