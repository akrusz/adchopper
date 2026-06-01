"""Shared data types and helpers for transcript segments and ad spans."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Segment:
    """A timestamped chunk of transcript."""

    index: int
    start: float  # seconds
    end: float  # seconds
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(index=d["index"], start=d["start"], end=d["end"], text=d["text"])


@dataclass
class AdSpan:
    """A time range identified as an ad."""

    start: float  # seconds
    end: float  # seconds
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def duration(self) -> float:
        return self.end - self.start


def fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS (or MM:SS when under an hour)."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def merge_spans(spans: List[AdSpan], gap: float = 2.0) -> List[AdSpan]:
    """Merge overlapping or near-adjacent ad spans (within `gap` seconds)."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: s.start)
    merged = [AdSpan(ordered[0].start, ordered[0].end, ordered[0].reason)]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start <= last.end + gap:
            last.end = max(last.end, span.end)
            if span.reason and span.reason not in last.reason:
                last.reason = (last.reason + "; " + span.reason).strip("; ")
        else:
            merged.append(AdSpan(span.start, span.end, span.reason))
    return merged


def keep_ranges(
    ad_spans: List[AdSpan], total_duration: float
) -> List[tuple[float, float]]:
    """Return the complement of ad spans: the (start, end) ranges to keep."""
    ads = merge_spans(ad_spans)
    keep: List[tuple[float, float]] = []
    cursor = 0.0
    for span in ads:
        start = max(0.0, span.start)
        if start > cursor:
            keep.append((cursor, start))
        cursor = max(cursor, span.end)
    if cursor < total_duration:
        keep.append((cursor, total_duration))
    # Drop empty / negative ranges.
    return [(a, b) for a, b in keep if b - a > 0.05]
