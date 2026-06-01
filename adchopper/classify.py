"""Ad detection: timestamped segments -> ad spans, via a local LLM.

The transcript is presented to the LLM as numbered, timestamped lines. The
LLM returns *segment index ranges* it judges to be ads -- never raw
timestamps -- so the cut points stay grounded in the real transcript data
rather than hallucinated times. Index ranges are then mapped back to time
spans and merged.

Long transcripts are sent in overlapping windows to stay within the model's
context, and the resulting spans are merged across windows.
"""

from __future__ import annotations

import json
import re
from typing import List

import requests

from .segments import Segment, AdSpan, fmt_ts, merge_spans


SYSTEM_PROMPT = """\
You are an expert at detecting advertisements and sponsor reads inside \
podcast transcripts. You will be given a numbered, timestamped transcript. \
Your job is to identify which line ranges are ADVERTISEMENTS and should be \
removed.

What counts as an ad / sponsor read:
- "This episode is brought to you by ...", "Today's sponsor is ...".
- Promo/discount codes ("use code POD20", "20% off at checkout").
- Reading out URLs or phone numbers for a product/service.
- Calls to action to buy/subscribe/sign up for a third-party product.
- Host-read native ads (sound like the host but pitch a product).
- Dynamically inserted ad breaks (different audio/voice, abrupt topic shift).

What is NOT an ad (do NOT remove):
- The host promoting their OWN show, Patreon, merch, or newsletter is a gray
  area -- only flag it if it is clearly a transactional pitch with a CTA.
- Normal show content, interviews, banter, intros/outros without a product
  pitch.
- Mentions of a product as part of the actual discussion (e.g. reviewing it).

Rules:
- Ads are usually CONTIGUOUS blocks of several lines. Group them.
- Be conservative: when unsure, do NOT flag it. Cutting real content is worse
  than leaving an ad in.
- Respond with ONLY a JSON object, no prose, of the form:
  {"ads": [{"start_line": <int>, "end_line": <int>, "reason": "<short>"}]}
  where start_line/end_line are inclusive line numbers from the transcript.
- If there are no ads, respond with {"ads": []}.
"""


def _format_window(segments: List[Segment]) -> str:
    lines = []
    for s in segments:
        lines.append(f"[{s.index}] ({fmt_ts(s.start)}) {s.text}")
    return "\n".join(lines)


def _windows(segments: List[Segment], window: int, overlap: int):
    """Yield overlapping slices of the segment list."""
    if window <= 0:
        yield segments
        return
    step = max(1, window - overlap)
    i = 0
    n = len(segments)
    while i < n:
        yield segments[i : i + window]
        if i + window >= n:
            break
        i += step


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    # Strip code fences if present.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"ads": []}


def _spans_from_response(
    data: dict, by_index: dict[int, Segment]
) -> List[AdSpan]:
    spans: List[AdSpan] = []
    for ad in data.get("ads", []) or []:
        try:
            start_line = int(ad["start_line"])
            end_line = int(ad["end_line"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_line < start_line:
            start_line, end_line = end_line, start_line
        # Map line indices to actual segment times (ignore unknown indices).
        start_seg = by_index.get(start_line)
        end_seg = by_index.get(end_line)
        if start_seg is None or end_seg is None:
            continue
        # Use word-level boundaries when available so the cut starts at the
        # first spoken word of the ad and ends at its last spoken word,
        # trimming the silence padding that segment timestamps include.
        spans.append(
            AdSpan(
                start=start_seg.word_start,
                end=end_seg.word_end,
                reason=str(ad.get("reason", "")).strip(),
            )
        )
    return spans


def _call_ollama(
    system: str, user: str, model: str, host: str, timeout: float
) -> str:
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        raise SystemExit(
            f"Could not reach Ollama at {host}. Is it running? "
            f"(start it with `ollama serve` and `ollama pull {model}`)"
        ) from e
    if resp.status_code == 404:
        raise SystemExit(
            f"Ollama model '{model}' not found. Pull it with: ollama pull {model}"
        )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def classify_ads(
    segments: List[Segment],
    model: str = "llama3.1:8b",
    host: str = "http://localhost:11434",
    window: int = 220,
    overlap: int = 20,
    timeout: float = 300.0,
    verbose: bool = True,
) -> List[AdSpan]:
    """Detect ad spans in the transcript using a local Ollama model."""
    by_index = {s.index: s for s in segments}
    all_spans: List[AdSpan] = []

    windows = list(_windows(segments, window, overlap))
    for wi, win in enumerate(windows):
        if verbose:
            print(
                f"[classify] window {wi + 1}/{len(windows)} "
                f"(lines {win[0].index}-{win[-1].index})"
            )
        user = (
            "Here is the transcript window. Identify the advertisement line "
            "ranges.\n\n" + _format_window(win)
        )
        content = _call_ollama(SYSTEM_PROMPT, user, model, host, timeout)
        data = _extract_json(content)
        spans = _spans_from_response(data, by_index)
        if verbose and spans:
            for sp in spans:
                print(
                    f"    ad: {fmt_ts(sp.start)}-{fmt_ts(sp.end)} "
                    f"({sp.duration:.0f}s) {sp.reason}"
                )
        all_spans.extend(spans)

    merged = merge_spans(all_spans)
    if verbose:
        total = sum(s.duration for s in merged)
        print(
            f"[classify] {len(merged)} ad span(s), "
            f"{total:.0f}s total flagged for removal"
        )
    return merged
