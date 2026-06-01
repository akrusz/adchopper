"""Transcription: audio -> timestamped segments.

Default backend is faster-whisper (local, CPU-friendly). Transcripts are
cached to JSON so re-running the pipeline doesn't re-transcribe.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from .segments import Segment, Word


def transcribe(
    audio_path: str,
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "default",
    language: Optional[str] = None,
    cache_path: Optional[str] = None,
    word_timestamps: bool = True,
    verbose: bool = True,
) -> List[Segment]:
    """Transcribe `audio_path` into timestamped segments.

    If `cache_path` exists, segments are loaded from it instead of
    re-transcribing. Otherwise the result is written there.
    """
    if cache_path and os.path.exists(cache_path):
        if verbose:
            print(f"[transcribe] loading cached transcript: {cache_path}")
        return load_segments(cache_path)

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover - dependency hint
        raise SystemExit(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from e

    if verbose:
        print(
            f"[transcribe] loading whisper model '{model_size}' "
            f"(device={device}, compute_type={compute_type})"
        )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    if verbose:
        print(f"[transcribe] transcribing {audio_path} ...")
    raw_segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,  # skip long silences, improves timestamps
        word_timestamps=word_timestamps,
    )

    segments: List[Segment] = []
    for i, seg in enumerate(raw_segments):
        words = None
        if word_timestamps and getattr(seg, "words", None):
            words = [
                Word(word=w.word, start=float(w.start), end=float(w.end))
                for w in seg.words
                # word timings are occasionally None for filler tokens
                if w.start is not None and w.end is not None
            ]
        segments.append(
            Segment(
                index=i,
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text.strip(),
                words=words or None,
            )
        )
        if verbose and i % 25 == 0:
            print(f"  ... {i} segments ({seg.end:.0f}s)", end="\r", flush=True)

    if verbose:
        lang = getattr(info, "language", language) or "?"
        print(f"\n[transcribe] done: {len(segments)} segments, language={lang}")

    if cache_path:
        save_segments(segments, cache_path)
        if verbose:
            print(f"[transcribe] cached transcript -> {cache_path}")

    return segments


def save_segments(segments: List[Segment], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in segments], f, ensure_ascii=False, indent=2)


def load_segments(path: str) -> List[Segment]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Segment.from_dict(d) for d in data]


def audio_duration(segments: List[Segment]) -> float:
    """Best-effort total duration from the last segment's end time."""
    return segments[-1].end if segments else 0.0
