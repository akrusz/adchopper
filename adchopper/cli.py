"""Command-line interface for adchopper.

Typical use:

    adchopper episode.mp3                  # review-only: show detected ads
    adchopper episode.mp3 -o clean.mp3 -y  # detect and cut

The pipeline transcribes (cached), asks a local LLM which segments are ads,
prints a report, and -- when confirmed -- removes those spans with ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from . import __version__
from .segments import AdSpan, fmt_ts
from .transcribe import transcribe, audio_duration
from .classify import classify_ads
from .cut import cut_ads, probe_duration


def _default_paths(audio_path: str):
    base, _ = os.path.splitext(audio_path)
    return {
        "transcript": base + ".transcript.json",
        "report": base + ".ads.json",
        "output": base + ".noads.mp3",
    }


def _print_report(ad_spans: List[AdSpan], duration: float) -> None:
    print("\n=== Detected ad spans ===")
    if not ad_spans:
        print("  (none)")
        return
    total = 0.0
    for i, sp in enumerate(ad_spans, 1):
        total += sp.duration
        print(
            f"  {i:2d}. {fmt_ts(sp.start)} - {fmt_ts(sp.end)} "
            f"({sp.duration:.0f}s)  {sp.reason}"
        )
    pct = (total / duration * 100) if duration else 0
    print(f"  -> {total:.0f}s of {duration:.0f}s flagged ({pct:.1f}%)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="adchopper",
        description="Remove ads from podcast MP3s "
        "(transcribe -> LLM ad detection -> ffmpeg cut).",
    )
    p.add_argument("audio", help="Path to the input podcast audio (mp3, etc.)")
    p.add_argument("-o", "--output", help="Output mp3 path (default: <input>.noads.mp3)")
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Cut without the interactive confirmation prompt.",
    )
    p.add_argument(
        "--review-only",
        action="store_true",
        help="Only detect ads and write the report; never cut audio.",
    )

    g = p.add_argument_group("transcription")
    g.add_argument(
        "--whisper-model",
        default="base",
        help="faster-whisper model size (tiny/base/small/medium/large-v3). "
        "Default: base.",
    )
    g.add_argument("--device", default="auto", help="cpu / cuda / auto (default: auto)")
    g.add_argument(
        "--compute-type",
        default="default",
        help="faster-whisper compute type (e.g. int8, float16). Default: default.",
    )
    g.add_argument("--language", default=None, help="Force language code (e.g. en).")
    g.add_argument(
        "--transcript",
        help="Path to read/write the cached transcript JSON.",
    )
    g.add_argument(
        "--retranscribe",
        action="store_true",
        help="Ignore any cached transcript and transcribe again.",
    )

    g2 = p.add_argument_group("ad detection (LLM)")
    g2.add_argument(
        "--llm-model",
        default="llama3.1:8b",
        help="Ollama model name. Default: llama3.1:8b.",
    )
    g2.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama base URL. Default: http://localhost:11434.",
    )
    g2.add_argument(
        "--window",
        type=int,
        default=220,
        help="Transcript lines per LLM window. Default: 220.",
    )
    g2.add_argument(
        "--overlap",
        type=int,
        default=20,
        help="Overlap (lines) between windows. Default: 20.",
    )
    g2.add_argument(
        "--ads-from",
        help="Skip detection and load ad spans from this JSON report "
        "(lets you hand-edit before cutting).",
    )

    g3 = p.add_argument_group("cutting")
    g3.add_argument(
        "--fade",
        type=float,
        default=0.02,
        help="Fade (seconds) at each cut boundary to avoid clicks. Default: 0.02.",
    )
    g3.add_argument(
        "--bitrate", default="128k", help="Output audio bitrate. Default: 128k."
    )

    p.add_argument("-q", "--quiet", action="store_true", help="Less output.")
    p.add_argument("--version", action="version", version=f"adchopper {__version__}")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verbose = not args.quiet

    if not os.path.exists(args.audio):
        print(f"error: input not found: {args.audio}", file=sys.stderr)
        return 2

    paths = _default_paths(args.audio)
    transcript_path = args.transcript or paths["transcript"]
    report_path = paths["report"]
    output_path = args.output or paths["output"]

    # 1. Ad spans: either loaded from a report, or detected fresh.
    if args.ads_from:
        with open(args.ads_from, "r", encoding="utf-8") as f:
            ad_data = json.load(f)
        ad_spans = [AdSpan(**d) for d in ad_data]
        duration = probe_duration(args.audio) or 0.0
    else:
        cache = None if args.retranscribe else transcript_path
        segments = transcribe(
            args.audio,
            model_size=args.whisper_model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            cache_path=transcript_path if not args.retranscribe else cache,
            verbose=verbose,
        )
        # Always (re)write the cache after a fresh transcription.
        if args.retranscribe:
            from .transcribe import save_segments

            save_segments(segments, transcript_path)

        duration = probe_duration(args.audio) or audio_duration(segments)
        ad_spans = classify_ads(
            segments,
            model=args.llm_model,
            host=args.ollama_host,
            window=args.window,
            overlap=args.overlap,
            verbose=verbose,
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in ad_spans], f, indent=2)
        if verbose:
            print(f"[report] ad spans written -> {report_path}")

    _print_report(ad_spans, duration)

    if args.review_only:
        print("\nReview-only mode: no audio was cut.")
        print(f"Edit {report_path} if needed, then run with --ads-from to cut.")
        return 0

    if not ad_spans:
        print("\nNo ads detected; nothing to cut.")
        return 0

    # 2. Confirm, then cut.
    if not args.yes:
        try:
            answer = input(f"\nCut these spans into {output_path}? [y/N] ").strip().lower()
        except EOFError:
            answer = "n"
        if answer not in ("y", "yes"):
            print("Aborted. (Re-run with -y to skip this prompt.)")
            return 0

    cut_ads(
        args.audio,
        ad_spans,
        output_path,
        total_duration=duration or None,
        fade=args.fade,
        bitrate=args.bitrate,
        verbose=verbose,
    )
    print(f"\nDone -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
