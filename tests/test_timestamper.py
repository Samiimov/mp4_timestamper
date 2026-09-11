import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

from pydantic import ValidationError

from mp4_timestamper.cli import check_output, main, write_text
from mp4_timestamper.media import extract_audio, transcribe
from mp4_timestamper.models import ChapterSelections, Segment, ToolError, Topic, Transcript
from mp4_timestamper.outline import generate_outline, render_outline, timestamp, transcript_batches


def sample_transcript():
    return Transcript(source="lesson.mp4", segments=[
        Segment(start=5, end=15, text="An introduction to growing tomatoes."),
        Segment(start=80, end=120, text="Plant tomatoes in rich soil in full sun."),
        Segment(start=3605, end=3630, text="Now we discuss watering and pruning."),
    ])


def selection(ids, continues=False):
    return ChapterSelections(
        continues_previous=continues,
        chapters=[{"start_segment": i, "title": f"Topic {i}", "summary": f"Summary {i}."} for i in ids],
    )


class OutlineTests(unittest.TestCase):
    def test_timestamps_cover_hours_without_wrapping(self):
        self.assertEqual(timestamp(3661.9), "01:01:01")
        self.assertEqual(timestamp(360000), "100:00:00")

    def test_render_plain_text(self):
        text = render_outline("lesson.mp4", [Topic(65.9, "Planting", "Prepare the soil.")])
        self.assertIn("00:01:05 — Planting\n  Prepare the soil.\n", text)

    def test_topic_positions_come_from_transcript(self):
        client = Mock()
        client.select_chapters.return_value = selection([0, 2])
        topics = generate_outline(sample_transcript(), client)
        self.assertEqual([topic.start for topic in topics], [5, 3605])

    def test_rejects_invalid_or_incomplete_chapter_positions(self):
        for ids in ([0, 99], [0, 2, 1], [0, 0], [1], []):
            with self.subTest(ids=ids):
                client = Mock()
                client.select_chapters.return_value = selection(ids)
                with self.assertRaises(ToolError):
                    generate_outline(sample_transcript(), client)

    def test_refusal_and_empty_transcript(self):
        client = Mock()
        client.select_chapters.return_value = None
        with self.assertRaises(ToolError):
            generate_outline(sample_transcript(), client)
        client.reset_mock()
        with self.assertRaises(ToolError):
            generate_outline(Transcript(source="silent.mp4", segments=[]), client)
        client.select_chapters.assert_not_called()

    def test_batched_continuation_keeps_original_timestamp(self):
        client = Mock()
        client.select_chapters.side_effect = [selection([0]), selection([1, 2], continues=True)]
        batches = [[{"id": 0, "text": "Introduction"}],
                   [{"id": 1, "text": "Continued"}, {"id": 2, "text": "New topic"}]]
        with patch("mp4_timestamper.outline.transcript_batches", return_value=iter(batches)):
            topics = generate_outline(sample_transcript(), client)
        self.assertEqual([topic.start for topic in topics], [5, 3605])
        self.assertEqual(topics[0].summary, "Summary 1.")
        second_input = json.loads(client.select_chapters.call_args.kwargs["content"])
        self.assertEqual(second_input["previous_chapter"]["title"], "Topic 0")

    def test_batches_cover_all_unicode_segments_with_bounded_size(self):
        transcript = Transcript(source="x", segments=[
            Segment(start=i, end=i + 1, text="日本語" * 10) for i in range(12)
        ])
        batches = list(transcript_batches(transcript, limit=300))
        self.assertGreater(len(batches), 1)
        self.assertEqual([row["id"] for batch in batches for row in batch], list(range(12)))
        for batch in batches:
            self.assertLessEqual(len(json.dumps(batch, ensure_ascii=False).encode()), 300)

    def test_invalid_transcript_timestamps(self):
        for start, end in [(-1, 1), (2, 1), (float("nan"), 2), (0, float("inf"))]:
            with self.subTest(start=start, end=end), self.assertRaises(ValidationError):
                Segment(start=start, end=end, text="Speech")
        with self.assertRaises(ValidationError):
            Transcript(source="x", segments=list(reversed(sample_transcript().segments)))



class FileAndCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_no_overwrite_and_atomic_replacement(self):
        output = self.root / "outline.txt"
        write_text(output, "original", False)
        with self.assertRaises(FileExistsError):
            write_text(output, "replacement", False)
        self.assertEqual(output.read_text(), "original")
        write_text(output, "replacement", True)
        self.assertEqual(output.read_text(), "replacement")
        self.assertEqual(list(self.root.iterdir()), [output])

    def test_input_protected_even_when_forcing(self):
        source = self.root / "video.mp4"
        source.touch()
        alias = self.root / "alias.mp4"
        os.link(source, alias)
        for target in (source, alias):
            with self.assertRaises(ToolError):
                check_output(target, [source], True)

    def test_missing_codex_has_actionable_error(self):
        source = self.root / "video.mp4"
        source.touch()
        stderr = io.StringIO()
        with patch("mp4_timestamper.codex.shutil.which", return_value=None), contextlib.redirect_stderr(stderr):
            self.assertEqual(main([str(source)]), 1)
        self.assertIn("Codex CLI was not found", stderr.getvalue())
        self.assertFalse(source.with_suffix(".topics.txt").exists())

    def test_transcribe_only_needs_no_codex_or_key(self):
        source = self.root / "video.mp4"
        source.touch()
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
                patch("mp4_timestamper.cli.CodexClient") as codex, \
                patch("mp4_timestamper.cli.transcribe", return_value=sample_transcript()):
            self.assertEqual(main([str(source), "--transcribe-only"]), 0)
        codex.assert_not_called()
        saved = Transcript.model_validate_json(source.with_suffix(".transcript.json").read_text())
        self.assertEqual(saved, sample_transcript())

    def test_reuse_transcript_skips_audio_and_writes_outline(self):
        source = self.root / "saved.json"
        source.write_text(sample_transcript().model_dump_json(), encoding="utf-8")
        output = self.root / "topics.txt"
        client = Mock()
        client.select_chapters.return_value = selection([0, 2])
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
                patch("mp4_timestamper.cli.CodexClient") as constructor, \
                patch("mp4_timestamper.cli.transcribe") as audio:
            constructor.return_value = client
            self.assertEqual(main(["--from-transcript", str(source), "-o", str(output)]), 0)
        audio.assert_not_called()
        self.assertIn("01:00:05 — Topic 2", output.read_text(encoding="utf-8"))

    def test_transcript_survives_failed_outline(self):
        video = self.root / "video.mp4"
        video.touch()
        saved = self.root / "saved.json"
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
                patch("mp4_timestamper.cli.CodexClient"), \
                patch("mp4_timestamper.cli.transcribe", return_value=sample_transcript()), \
                patch("mp4_timestamper.cli.generate_outline", side_effect=ToolError("Failed")):
            self.assertEqual(main([str(video), "--save-transcript", str(saved)]), 1)
        self.assertEqual(Transcript.model_validate_json(saved.read_text()), sample_transcript())
        self.assertFalse(video.with_suffix(".topics.txt").exists())


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
class MediaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.video = self.root / "video with spaces.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "color=c=black:s=32x32:r=10:d=5", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=16000:duration=5",
            "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(self.video),
        ], check=True, capture_output=True)

    def test_real_extraction_and_local_transcription_keep_chunk_offsets(self):
        durations = []
        paths = []
        def recognize(path, **options):
            paths.append(Path(path))
            with wave.open(path, "rb") as audio:
                self.assertEqual(audio.getframerate(), 16000)
                self.assertEqual(audio.getnchannels(), 1)
                durations.append(audio.getnframes() / audio.getframerate())
            self.assertTrue(options["vad_filter"])
            return iter([SimpleNamespace(start=0.1, end=0.5, text="A spoken topic.")]), SimpleNamespace(language="en")
        model = Mock()
        model.transcribe.side_effect = recognize
        with patch("mp4_timestamper.media.load_model", return_value=model), \
                patch("mp4_timestamper.media.CHUNK_SECONDS", 2):
            transcript = transcribe(self.video)
        self.assertEqual(len(durations), 3)
        self.assertAlmostEqual(transcript.segments[1].start, durations[0] + 0.1)
        self.assertAlmostEqual(transcript.segments[2].start, sum(durations[:2]) + 0.1)
        self.assertEqual(model.transcribe.call_args.kwargs["language"], "en")
        self.assertTrue(all(not path.exists() for path in paths))

    def test_lazy_recognition_error_cleans_up_audio(self):
        paths = []
        def broken_segments():
            yield SimpleNamespace(start=0, end=0.5, text="First sentence")
            raise RuntimeError("Inference failed")
        def recognize(path, **options):
            paths.append(Path(path))
            return broken_segments(), SimpleNamespace(language="en")
        model = Mock()
        model.transcribe.side_effect = recognize
        with patch("mp4_timestamper.media.load_model", return_value=model):
            with self.assertRaisesRegex(ToolError, "Local transcription failed"):
                transcribe(self.video)
        self.assertTrue(all(not path.exists() for path in paths))

    def test_missing_audio_gives_actionable_error(self):
        silent = self.root / "no-audio.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(self.video),
            "-an", "-c:v", "copy", str(silent),
        ], check=True, capture_output=True)
        with self.assertRaisesRegex(ToolError, "audio track"):
            extract_audio(silent, self.root)

    def test_whisper_end_overshoot_is_clamped_and_phantom_segments_are_skipped(self):
        model = Mock()
        model.transcribe.return_value = (iter([
            SimpleNamespace(start=0.1, end=15, text="Real speech."),
            SimpleNamespace(start=15, end=20, text="Beyond the recording."),
        ]), SimpleNamespace(language="en"))
        with patch("mp4_timestamper.media.load_model", return_value=model):
            transcript = transcribe(self.video)
        self.assertEqual(len(transcript.segments), 1)
        self.assertEqual(transcript.segments[0].start, 0.1)
        self.assertAlmostEqual(transcript.segments[0].end, 5.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
