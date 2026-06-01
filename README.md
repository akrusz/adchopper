# adchopper

Remove ads from podcast MP3s, fully locally:

1. **Transcribe** the audio into timestamped segments with
   [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper).
2. **Detect ads** by sending the numbered, timestamped transcript to an LLM and
   asking which line ranges are sponsor reads / ads. The backend is pluggable:
   a local [Ollama](https://ollama.com) model (default, fully offline), or the
   **Claude** / **OpenAI** API for higher accuracy.
3. **Cut** those spans out of the MP3 with `ffmpeg`.

The LLM returns *segment line ranges*, not raw timestamps, so cut points stay
grounded in the actual transcript instead of hallucinated times. Whisper's
**word-level timestamps** are then used to snap each cut to the first/last
*spoken word* of the ad, trimming the silence padding that segment timestamps
include.

One run does the whole job for one MP3 — transcribe, detect, preview, cut:

```bash
adchopper episode.mp3 -o clean.mp3     # shows the cuts, asks once, then cuts
adchopper episode.mp3 -o clean.mp3 -y  # trust it: no prompt, just cut
```

## Why this design

- **STT, not TTS.** You want speech→text with timestamps; Whisper is the tool.
  `faster-whisper` runs on CPU and needs no separate server.
- **Conservative by default.** Cutting real content is worse than leaving an ad
  in, so the prompt errs toward *not* flagging, and the default flow is
  **preview-then-cut**: it prints each detected span *with the transcript text
  that will be removed* and asks once before touching audio. Trust it with
  `-y` to skip the prompt; hand-edit with `--ads-from` when you don't.
- **Auditable.** Every run writes a Markdown report (`<input>.report.md`)
  listing each removed span with its timestamp *and the exact transcript text
  that was cut* — so you (or a skeptic) can verify nothing real was lost
  without opening the audio. The original file is never modified.
- **Word-accurate cuts.** Per-word timestamps tighten each boundary to the
  spoken word, so cuts land cleanly instead of clipping speech or leaving gaps.
- **Cached transcript.** Transcription is the slow step; the transcript is
  saved to JSON so re-running detection/cutting is fast.

## Install

```bash
pip install -e .                 # core (faster-whisper + Ollama backend)
pip install -e '.[anthropic]'    # optional: Claude backend
pip install -e '.[openai]'       # optional: OpenAI backend

# system dependency:
#   macOS:  brew install ffmpeg
#   Debian: sudo apt install ffmpeg

# for the default (local) backend:
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

### Choosing a detection backend

```bash
# Local, offline (default):
adchopper episode.mp3

# Claude — usually best on subtle host-read ads (needs ANTHROPIC_API_KEY):
export ANTHROPIC_API_KEY=sk-ant-...
adchopper episode.mp3 --backend anthropic

# OpenAI (needs OPENAI_API_KEY):
export OPENAI_API_KEY=sk-...
adchopper episode.mp3 --backend openai
```

The cloud backends use **structured outputs** so the model returns
schema-valid JSON, and the Claude backend **caches** the (large, identical)
system prompt across transcript windows, so only the first window pays for it.
Note that the cloud backends send the transcript text (not the audio) to the
provider; the local backend keeps everything on your machine.

### Useful options

| Flag | Purpose | Default |
|------|---------|---------|
| `--whisper-model` | tiny/base/small/medium/large-v3 (bigger = better + slower) | `base` |
| `--device` / `--compute-type` | e.g. `cuda` / `int8` for speed | `auto` / `default` |
| `--no-word-timestamps` | coarser segment-level cuts (slightly faster) | word timing on |
| `--backend` | `ollama` / `anthropic` / `openai` | `ollama` |
| `--llm-model` | model name | per-backend default |
| `--ollama-host` | Ollama base URL (or `OLLAMA_HOST` env) | `http://localhost:11434` |
| `--window` / `--overlap` | transcript lines per LLM call / overlap | `220` / `20` |
| `--ads-from` | cut from an existing/edited report instead of detecting | – |
| `--fade` | seconds of fade at each cut to avoid clicks | `0.02` |
| `--review-only` | never cut, just report | off |

## Files produced

For `episode.mp3` you get:

- `episode.transcript.json` — cached transcript (reused on re-runs)
- `episode.ads.json` — detected ad spans `[{start, end, reason}]` (editable)
- `episode.report.md` — human-readable audit report of what was cut
- `episode.noads.mp3` — the cut output (or your `-o` path)

## How well does it work?

Ad detection quality tracks the LLM. A small 8B model catches most explicit
sponsor reads with promo codes/URLs; a larger model (or a cloud model) is more
reliable on subtle host-read native ads. Always skim the report before cutting
the first few episodes of a new show to calibrate. Transcription quality
(and thus cut precision) improves with a larger `--whisper-model`.

## Limitations / ideas

- Dynamically inserted ads that change between downloads will differ per file.
- Boundaries snap to word timestamps (typically within ~0.1–0.3s), not
  sample-accurate; the small fade hides the seams. Pass `--no-word-timestamps`
  for coarser segment-level boundaries.
- Possible extensions: an audio-fingerprint pass to catch repeated jingles
  across episodes, and an RSS-feed watcher that auto-chops new episodes.

## Development

```bash
pip install pytest
pytest            # exercises the pure logic (no ffmpeg/whisper/ollama needed)
```
