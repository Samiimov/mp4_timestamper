# MP4 Timestamper

Create a text file of video topics, timestamps, and short summaries using **local Whisper transcription and Codex signed in with ChatGPT**. No OpenAI API key is needed.

Example output:

```text
Topic outline: lecture.mp4

00:00:05 — Introduction to solar energy
  Explains the goals of the session and the basics of solar power.

00:03:42 — How photovoltaic cells work
  Describes how solar cells convert sunlight into electricity.
```

## Setup

Requires Python 3.10+, FFmpeg, and a current Codex CLI with ChatGPT access.

Install FFmpeg using your package manager (`sudo apt install ffmpeg` on Ubuntu or `brew install ffmpeg` on macOS). Install [Codex CLI](https://learn.chatgpt.com/docs/cli) if needed, for example with `npm install -g @openai/codex`, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
codex login
codex login status
```

Choose **Sign in with ChatGPT**. The status should say `Logged in using ChatGPT`. If Codex is already signed in this way, no new login is needed. On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.

The tool uses [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) on your computer. The default `base` model downloads from Hugging Face the first time it is used and is cached for later runs. CPU transcription uses int8 precision and requires no GPU. The default cache location is managed by Hugging Face; use `--model-dir` to choose a directory.

## Usage

```bash
mp4-timestamper "video.mp4"
```

Writes `video.topics.txt` beside the input. `python -m mp4_timestamper` also works.

### Process a folder

Pass a full folder path to process all MP4 files directly inside it:

```bash
mp4-timestamper "/home/sami/Videos/My videos" --model-dir .models
```

Each video gets its own outline beside it, for example `lecture.mp4` becomes `lecture.topics.txt`. Files are processed sequentially in filename order; `.mp4` and `.MP4` are both accepted. Subfolders are not scanned.

For folder input, `-o` and `--save-transcript` accept **existing directories**:

```bash
mp4-timestamper "/home/sami/Videos" \
  -o "/home/sami/Documents/outlines" \
  --save-transcript "/home/sami/Documents/transcripts" \
  --model-dir .models

# Create just the local transcripts for a folder.
mp4-timestamper "/home/sami/Videos" --transcribe-only --model-dir .models
```

Existing outlines (or JSON outputs with `--transcribe-only`) are skipped unless `--force` is supplied, so you can rerun a folder without repeating completed videos. A failed video is reported and processing continues. The final summary lists processed, skipped, and failed counts; the exit code is 1 if any video failed. Conflicting output names are rejected before processing begins. Existing optional saved transcripts are also protected; use `--force` to replace them or `--from-transcript` to reuse one.

### More options

```bash
# Save the transcript before outlining so it survives a Codex failure.
mp4-timestamper "lecture.mp4" --save-transcript lecture.transcript.json

# Generate another outline without transcribing again.
mp4-timestamper --from-transcript lecture.transcript.json -o brief.txt --detail brief

# Use a larger local model and produce an English outline of Finnish speech.
mp4-timestamper "lecture.mp4" --whisper-model small --language fi --outline-language English

# Keep downloaded Whisper models in this directory.
mp4-timestamper "video.mp4" --model-dir .models

# Transcribe only: no Codex or ChatGPT login required.
mp4-timestamper "video.mp4" --transcribe-only

# Transcribe with an already cached model, without model-download requests.
mp4-timestamper "video.mp4" --transcribe-only --model-dir .models --offline
```

`--transcribe-only` writes `video.transcript.json`; `-o` can choose another JSON path. Otherwise `-o` chooses the outline text path.

| Option | Purpose |
| --- | --- |
| `--detail brief\|balanced\|detailed` | Chapter granularity; default `balanced` |
| `--whisper-model NAME_OR_PATH` | Local speech model; default `base`. Larger models need more memory and time |
| `--device cpu\|cuda` | Transcription hardware; default `cpu`. CUDA needs compatible NVIDIA libraries as described in Faster Whisper's documentation |
| `--language en` | Spoken language code; automatically detected by default |
| `--outline-language English` | Language for titles and summaries; defaults to the transcript's language |
| `--model MODEL` | Override the Codex outline model; otherwise uses the CLI's built-in default |
| `--codex-bin PATH` | Codex executable if it is not on PATH |
| `--codex-timeout 900` | Seconds allowed per outline request; default 600 |
| `--model-dir PATH` | Directory for downloaded Whisper models |
| `--offline` | Prevent Whisper model downloads; **Codex still needs internet** |
| `--force` | Replace existing output files |

Output directories must already exist. Input files are protected even with `--force`. Progress and errors go to stderr. Exit codes: 0 success, 1 processing failure, 2 invalid arguments, 130 cancellation.

## How it works

1. Verify that Codex is available and signed in with ChatGPT (skipped for `--transcribe-only`).
2. FFmpeg extracts the first audio track into temporary ten-minute mono WAV chunks, retaining leading silence for delayed audio tracks.
3. Faster Whisper transcribes locally with voice activity detection. Measured chunk durations produce timestamps relative to the original video. Estimated segment ends are clamped to the audio duration.
4. Codex groups transcript segments into topics and returns structured JSON. The app validates chapter positions and converts the chosen segment IDs to timestamps. Large transcripts are processed in batches, with context to merge a continuing chapter.
5. The app writes a UTF-8 outline and removes temporary audio and working files.

Transcripts saved with `--save-transcript` or `--transcribe-only` can be reused through `--from-transcript`, including transcripts created by the earlier API version. Times are seconds from the video start:

```json
{
  "source": "lecture.mp4",
  "segments": [
    {"start": 5.0, "end": 12.8, "text": "Today we will discuss solar energy."}
  ]
}
```

## Subscription and privacy

- **Audio stays on your computer.** Model downloads need internet on first use; speech inference is local.
- **Transcript text is sent to Codex.** It uses your ChatGPT/Codex allowance and is subject to your plan's limits. This is not a fully offline workflow. See [Codex authentication](https://learn.chatgpt.com/docs/auth) and [plan usage](https://learn.chatgpt.com/docs/pricing).
- The tool requires a ChatGPT login, removes API-key environment overrides from the Codex subprocess, and restricts Codex to ChatGPT authentication. It never reads or copies your login tokens.
- Codex runs in an ephemeral session in a temporary directory, with a read-only sandbox, shell tools and web search disabled. It ignores user `config.toml` for these requests, so custom models, providers, and configured integrations are not inherited; use `--model` for model selection. Codex still uses its normal authentication and local runtime state. This does not change OpenAI's server-side data policies.

The previous direct OpenAI API workflow has been replaced. `OPENAI_API_KEY` is no longer required or used by this app.

## Limits and troubleshooting

- Only spoken content is analyzed. Silent slides and on-screen text are not covered. Audio quality, overlapping speakers, music, and chunk boundaries can affect results. Timestamps use speech-segment precision and are displayed as `HH:MM:SS`, rounded down.
- Extraction temporarily uses about 115 MB per hour of video, plus the model's disk and memory requirements. Chunks are deleted as processing proceeds. Transcription does not resume partway through an interrupted run.
- If a model download fails, check internet access or supply a downloaded model directory via `--whisper-model`. `--offline` requires a cached model.
- If Codex cannot be found, add it to PATH or use `--codex-bin`. If it reports an unsupported flag, update Codex CLI; the tool needs `exec`, `--ignore-user-config`, `--ephemeral`, and `--output-schema` support.
- For login problems, run `codex login status` and then `codex login` if needed. If a usage limit is reached, retry later with a saved transcript. Increase `--codex-timeout` for slow responses.

The Codex integration uses its documented [non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode).

## Development

```bash
python -m unittest discover -s tests -v
```

Tests use simulated local recognition and Codex responses, plus real FFmpeg extraction against generated video fixtures. They do not download models or call Codex. FFmpeg tests are skipped when FFmpeg is unavailable.
