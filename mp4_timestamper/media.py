"""Transcribe audio locally and preserve positions in the video."""

from pathlib import Path
import math
import shutil
import subprocess
import sys
import tempfile
import wave

from .models import Segment, ToolError, Transcript


CHUNK_SECONDS = 600  # Bound the audio decoded into memory by Faster Whisper.


def extract_audio(video: Path, directory: Path) -> list[Path]:
    if not shutil.which("ffmpeg"):
        raise ToolError("FFmpeg is missing. Install it and ensure 'ffmpeg' is on PATH.")
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(video.resolve()), "-map", "0:a:0", "-vn",
            # Retain leading silence when the audio track starts after the video.
            "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", "-f", "segment",
            "-segment_time", str(CHUNK_SECONDS), "-reset_timestamps", "1",
            str(directory / "audio-%06d.wav"),
        ],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise ToolError(
            "Could not extract audio. Check that the video is readable and has an audio track.\n"
            + result.stderr.strip()[-1500:]
        )
    paths = sorted(directory.glob("audio-*.wav"))
    if not paths:
        raise ToolError("The video contains no usable audio.")
    return paths


def load_model(model_name: str, device: str, model_dir: Path | None, offline: bool):
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise ToolError("Local transcription requires faster-whisper. Run: pip install -e .") from error
    loading = "from cache" if offline else "downloads on first use"
    print(f"Loading Whisper '{model_name}' on {device} ({loading})…", file=sys.stderr)
    try:
        return WhisperModel(
            model_name, device=device, compute_type="int8" if device == "cpu" else "float16",
            download_root=str(model_dir) if model_dir else None, local_files_only=offline,
        )
    except Exception as error:
        raise ToolError(
            "Could not load the local Whisper model. Check --whisper-model, model download access "
            "(or the cache with --offline), and available memory. For GPU issues try --device cpu.\n"
            + str(error)
        ) from error


def transcribe(
    video: Path, language: str | None = None, model_name: str = "base",
    device: str = "cpu", model_dir: Path | None = None, offline: bool = False,
) -> Transcript:
    segments = []
    offset = 0.0
    with tempfile.TemporaryDirectory(prefix="mp4-timestamper-") as temporary:
        print("Extracting audio…", file=sys.stderr)
        paths = extract_audio(video, Path(temporary))
        model = load_model(model_name, device, model_dir, offline)
        for index, path in enumerate(paths, start=1):
            with wave.open(str(path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
            print(f"Transcribing locally, audio {index}/{len(paths)}…", file=sys.stderr)
            try:
                items, info = model.transcribe(
                    str(path), language=language, beam_size=5, vad_filter=True,
                )
                # Faster Whisper is lazy: inference errors occur while iterating, too.
                items = list(items)
            except Exception as error:
                raise ToolError(
                    f"Local transcription failed in audio chunk {index}. "
                    "Check the language code and available memory; for GPU issues try --device cpu.\n"
                    + str(error)
                ) from error
            if items and language is None:
                language = info.language
                print(f"Detected spoken language: {language}", file=sys.stderr)
            for item in items:
                text = item.text.strip()
                if not text:
                    continue
                if not (math.isfinite(item.start) and math.isfinite(item.end)
                        and 0 <= item.start <= item.end):
                    raise ToolError("Transcription returned an invalid timestamp. Please retry.")
                # Whisper may extend its final segment past the physical end of the
                # recording. Keep valid speech starts and clamp the estimated end.
                if item.start >= duration:
                    continue
                start = item.start + offset
                end = min(item.end, duration) + offset
                segments.append(Segment(start=start, end=end, text=text))
            # Use measured sample counts; segment muxing can round chunk lengths.
            offset += duration
            path.unlink()
    segments.sort(key=lambda segment: segment.start)
    return Transcript(source=video.name, segments=segments)
