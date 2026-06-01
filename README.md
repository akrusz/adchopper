# adchopper

Remove ads from podcast MP3s, fully locally:

1. **Transcribe** the audio into timestamped segments with
   [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper).
2. **Detect ads** by sending the numbered, timestamped transcript to a local
   LLM (via [Ollama](https://ollama.com)) and asking which line ranges are
   sponsor reads / ads.
3. **Cut** those spans out of the MP3 with `ffmpeg`.

The LLM returns *segment line ranges*, not raw timestamps, so cut points stay
grounded in the actual transcript instead of hallucinated times.

## Why this design

- **STT, not TTS.** You want speech→text with timestamps; Whisper is the tool.
  `faster-whisper` runs on CPU and needs no separate server.
- **Conservative by default.** Cutting real content is worse than leaving an ad
  in, so the prompt errs toward *not* flagging, and the default flow is
  **review-then-cut**: you see the detected spans (and can hand-edit them)
  before any audio is touched.
- **Cached transcript.** Transcription is the slow step; the transcript is
  saved to JSON so re-running detection/cutting is fast.

## Install

```bash
pip install -e .

# system dependency:
#   macOS:  brew install ffmpeg
#   Debian: sudo apt install ffmpeg

# local LLM:
ollama serve              # if not already running
ollama pull llama3.1:8b   # or any chat model you like
```

## Usage

```bash
# Review only: transcribe, detect ads, print + save a report. No cutting.
adchopper episode.mp3

# Detect and cut (asks for confirmation first):
adchopper episode.mp3 -o episode.clean.mp3

# Skip the prompt:
adchopper episode.mp3 -o episode.clean.mp3 -y

# Hand-edit the detected spans, then cut from your edited report:
adchopper episode.mp3                       # writes episode.ads.json
$EDITOR episode.ads.json                     # tweak start/end (seconds)
adchopper episode.mp3 --ads-from episode.ads.json -o episode.clean.mp3 -y
```

### Useful options

| Flag | Purpose | Default |
|------|---------|---------|
| `--whisper-model` | tiny/base/small/medium/large-v3 (bigger = better + slower) | `base` |
| `--device` / `--compute-type` | e.g. `cuda` / `int8` for speed | `auto` / `default` |
| `--llm-model` | Ollama model name | `llama3.1:8b` |
| `--ollama-host` | Ollama base URL (or `OLLAMA_HOST` env) | `http://localhost:11434` |
| `--window` / `--overlap` | transcript lines per LLM call / overlap | `220` / `20` |
| `--ads-from` | cut from an existing/edited report instead of detecting | – |
| `--fade` | seconds of fade at each cut to avoid clicks | `0.02` |
| `--review-only` | never cut, just report | off |

## Files produced

For `episode.mp3` you get:

- `episode.transcript.json` — cached transcript (reused on re-runs)
- `episode.ads.json` — detected ad spans `[{start, end, reason}]` (editable)
- `episode.noads.mp3` — the cut output (or your `-o` path)

## How well does it work?

Ad detection quality tracks the LLM. A small 8B model catches most explicit
sponsor reads with promo codes/URLs; a larger model (or a cloud model) is more
reliable on subtle host-read native ads. Always skim the report before cutting
the first few episodes of a new show to calibrate. Transcription quality
(and thus cut precision) improves with a larger `--whisper-model`.

## Limitations / ideas

- Dynamically inserted ads that change between downloads will differ per file.
- Boundaries are segment-accurate (typically within a second or two), not
  sample-accurate; the small fade hides the seams.
- Possible extensions: an OpenAI/Anthropic classifier backend (scaffolded in
  `pyproject.toml` extras), word-level timestamps for tighter cuts, and an
  audio-fingerprint pass to catch repeated jingles.

## Development

```bash
pip install pytest
pytest            # exercises the pure logic (no ffmpeg/whisper/ollama needed)
```
