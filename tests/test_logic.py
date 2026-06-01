"""Tests for the pure logic that doesn't need ffmpeg/whisper/ollama."""

from adchopper.segments import AdSpan, Segment, Word, merge_spans, keep_ranges, fmt_ts
from adchopper.classify import _extract_json, _spans_from_response, _windows
from adchopper.cli import _text_in_span


def test_fmt_ts():
    assert fmt_ts(0) == "00:00"
    assert fmt_ts(65) == "01:05"
    assert fmt_ts(3661) == "1:01:01"


def test_merge_spans_overlap_and_gap():
    spans = [
        AdSpan(10, 20, "a"),
        AdSpan(21, 30, "b"),  # within 2s gap of previous -> merge
        AdSpan(100, 110, "c"),  # far away -> separate
    ]
    merged = merge_spans(spans, gap=2.0)
    assert len(merged) == 2
    assert (merged[0].start, merged[0].end) == (10, 30)
    assert "a" in merged[0].reason and "b" in merged[0].reason
    assert (merged[1].start, merged[1].end) == (100, 110)


def test_merge_spans_unsorted():
    spans = [AdSpan(100, 110), AdSpan(10, 20)]
    merged = merge_spans(spans)
    assert [(s.start, s.end) for s in merged] == [(10, 20), (100, 110)]


def test_keep_ranges_basic():
    ads = [AdSpan(30, 60)]
    keep = keep_ranges(ads, total_duration=120)
    assert keep == [(0.0, 30.0), (60.0, 120.0)]


def test_keep_ranges_ad_at_start_and_end():
    ads = [AdSpan(0, 30), AdSpan(90, 120)]
    keep = keep_ranges(ads, total_duration=120)
    assert keep == [(30.0, 90.0)]


def test_keep_ranges_no_ads():
    keep = keep_ranges([], total_duration=120)
    assert keep == [(0.0, 120.0)]


def test_extract_json_plain():
    assert _extract_json('{"ads": []}') == {"ads": []}


def test_extract_json_fenced():
    text = '```json\n{"ads": [{"start_line": 1, "end_line": 3}]}\n```'
    data = _extract_json(text)
    assert data["ads"][0]["start_line"] == 1


def test_extract_json_with_prose():
    text = 'Sure! Here you go:\n{"ads": [{"start_line": 2, "end_line": 2}]}\nHope that helps.'
    data = _extract_json(text)
    assert data["ads"][0]["end_line"] == 2


def test_extract_json_garbage_returns_empty():
    assert _extract_json("not json at all") == {"ads": []}


def test_spans_from_response_maps_indices_to_times():
    segs = [
        Segment(0, 0.0, 5.0, "hi"),
        Segment(1, 5.0, 10.0, "sponsor read"),
        Segment(2, 10.0, 15.0, "use code FOO"),
        Segment(3, 15.0, 20.0, "back to show"),
    ]
    by_index = {s.index: s for s in segs}
    data = {"ads": [{"start_line": 1, "end_line": 2, "reason": "ad"}]}
    spans = _spans_from_response(data, by_index)
    assert len(spans) == 1
    assert spans[0].start == 5.0 and spans[0].end == 15.0


def test_spans_from_response_ignores_unknown_indices():
    by_index = {0: Segment(0, 0.0, 5.0, "x")}
    data = {"ads": [{"start_line": 99, "end_line": 100}]}
    assert _spans_from_response(data, by_index) == []


def test_spans_from_response_swaps_reversed_lines():
    segs = [Segment(i, i * 5.0, i * 5.0 + 5, "t") for i in range(4)]
    by_index = {s.index: s for s in segs}
    data = {"ads": [{"start_line": 3, "end_line": 1}]}
    spans = _spans_from_response(data, by_index)
    assert spans[0].start == 5.0 and spans[0].end == 20.0


def test_windows_overlap():
    segs = [Segment(i, 0, 0, "") for i in range(10)]
    wins = list(_windows(segs, window=4, overlap=1))
    # step = 3 -> windows start at 0,3,6; the window at 6 covers 6-9 (all 10).
    starts = [w[0].index for w in wins]
    assert starts == [0, 3, 6]
    assert wins[0][-1].index == 3
    # every segment is covered by at least one window
    covered = {s.index for w in wins for s in w}
    assert covered == set(range(10))


def test_windows_no_chunking_when_window_zero():
    segs = [Segment(i, 0, 0, "") for i in range(5)]
    wins = list(_windows(segs, window=0, overlap=0))
    assert len(wins) == 1 and len(wins[0]) == 5


def test_word_boundaries_default_to_segment_times():
    seg = Segment(0, 1.0, 9.0, "hi")
    assert seg.word_start == 1.0 and seg.word_end == 9.0


def test_word_boundaries_trim_silence_padding():
    # Segment spans 1.0-9.0 but speech is only 1.4-8.2 (VAD padding around it).
    seg = Segment(
        0,
        1.0,
        9.0,
        "buy now",
        words=[Word("buy", 1.4, 1.8), Word("now", 7.9, 8.2)],
    )
    assert seg.word_start == 1.4 and seg.word_end == 8.2


def test_spans_use_word_boundaries_when_available():
    segs = [
        Segment(0, 0.0, 5.0, "hi"),
        Segment(1, 5.0, 10.0, "sponsor", words=[Word("sponsor", 5.3, 9.6)]),
    ]
    by_index = {s.index: s for s in segs}
    data = {"ads": [{"start_line": 1, "end_line": 1}]}
    spans = _spans_from_response(data, by_index)
    assert spans[0].start == 5.3 and spans[0].end == 9.6


def test_segment_word_roundtrip():
    seg = Segment(2, 5.0, 10.0, "use code FOO", words=[Word("use", 5.0, 5.4)])
    restored = Segment.from_dict(seg.to_dict())
    assert restored.words[0].word == "use"
    assert restored.words[0].start == 5.0
    assert restored.word_end == 5.4


def test_segment_roundtrip_without_words():
    seg = Segment(0, 0.0, 1.0, "hi")
    restored = Segment.from_dict(seg.to_dict())
    assert restored.words is None


def test_text_in_span_collects_overlapping_text():
    segs = [
        Segment(0, 0.0, 5.0, "intro"),
        Segment(1, 5.0, 10.0, "this episode is sponsored by"),
        Segment(2, 10.0, 15.0, "use code FOO"),
        Segment(3, 15.0, 20.0, "back to the show"),
    ]
    span = AdSpan(5.0, 15.0)
    text = _text_in_span(span, segs)
    assert "sponsored" in text and "FOO" in text
    assert "intro" not in text and "back to the show" not in text


def test_text_in_span_truncates():
    segs = [Segment(0, 0.0, 5.0, "x" * 500)]
    text = _text_in_span(AdSpan(0.0, 5.0), segs, max_chars=50)
    assert len(text) <= 50 and text.endswith("…")


def test_text_in_span_no_segments():
    assert _text_in_span(AdSpan(0, 10), None) == ""
