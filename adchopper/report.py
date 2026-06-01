"""Human-readable audit report of what got cut.

The whole point is verifiability: for every span the tool removed, the report
shows the timestamp, the duration, the LLM's stated reason, *and the exact
transcript text that was cut*. A skeptic can read the report top to bottom and
confirm the tool only removed ads -- without ever opening the audio.
"""

from __future__ import annotations

from typing import List, Optional

from .segments import AdSpan, Segment, fmt_ts


def _text_for_span(span: AdSpan, segments: Optional[List[Segment]]) -> str:
    if not segments:
        return ""
    parts = [s.text for s in segments if s.end > span.start and s.start < span.end]
    return " ".join(p.strip() for p in parts).strip()


def build_markdown(
    audio_path: str,
    ad_spans: List[AdSpan],
    duration: float,
    segments: Optional[List[Segment]] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Return a Markdown audit report as a string."""
    removed = sum(s.duration for s in ad_spans)
    pct = (removed / duration * 100) if duration else 0.0
    kept = max(0.0, duration - removed)

    lines: List[str] = []
    lines.append(f"# adchopper report — {audio_path}")
    lines.append("")
    detector = ""
    if backend:
        detector = f" using **{backend}**" + (f" / `{model}`" if model else "")
    lines.append(f"Detected {len(ad_spans)} ad span(s){detector}.")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Original length | {fmt_ts(duration)} |")
    lines.append(f"| Removed (ads) | {fmt_ts(removed)} ({pct:.1f}%) |")
    lines.append(f"| Kept | {fmt_ts(kept)} |")
    lines.append("")
    lines.append(
        "> The original file is left untouched; the cut audio is written to a "
        "new file. Every removed span is listed below with the exact "
        "transcript text that was cut, so you can verify nothing real was lost."
    )
    lines.append("")

    if not ad_spans:
        lines.append("## No ads detected")
        lines.append("")
        lines.append("Nothing was removed.")
        return "\n".join(lines) + "\n"

    lines.append("## Removed spans")
    lines.append("")
    for i, sp in enumerate(ad_spans, 1):
        lines.append(
            f"### {i}. {fmt_ts(sp.start)} – {fmt_ts(sp.end)} "
            f"({sp.duration:.0f}s)"
        )
        lines.append("")
        if sp.reason:
            lines.append(f"**Why:** {sp.reason}")
            lines.append("")
        text = _text_for_span(sp, segments)
        if text:
            lines.append("**Transcript removed:**")
            lines.append("")
            lines.append("> " + text.replace("\n", "\n> "))
            lines.append("")
    return "\n".join(lines) + "\n"


def write_markdown(path: str, **kwargs) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_markdown(**kwargs))
