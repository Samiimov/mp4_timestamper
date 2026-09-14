import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mp4_timestamper.cli import main
from mp4_timestamper.models import ChapterSelections, Segment, ToolError, Transcript


class FolderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.videos = self.root / "My videos"
        self.videos.mkdir()
        self.codex = self.start_patch("mp4_timestamper.cli.CodexClient")
        self.codex.return_value.select_chapters.return_value = ChapterSelections(
            continues_previous=False,
            chapters=[{"start_segment": 0, "title": "Planting", "summary": "Prepare the soil."}],
        )
        self.audio = self.start_patch("mp4_timestamper.cli.transcribe")
        self.audio.side_effect = self.transcript

    def start_patch(self, name):
        patcher = patch(name)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    @staticmethod
    def transcript(source, **kwargs):
        return Transcript(source=source.name, segments=[Segment(start=5, end=10, text="Prepare the soil.")])

    def video(self, name):
        path = self.videos / name
        path.touch()
        return path

    def run_folder(self, *options):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([str(self.videos), *map(str, options)])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_folder_writes_one_outline_per_mp4_in_name_order(self):
        second = self.video("B lesson.MP4")
        first = self.video("a lesson.mp4")
        self.video("ignore.txt")
        nested = self.videos / "nested"
        nested.mkdir()
        (nested / "hidden.mp4").touch()
        code, stdout, _ = self.run_folder()
        self.assertEqual(code, 0)
        self.assertIn("2 processed, 0 skipped, 0 failed", stdout)
        self.assertEqual([call.args[0] for call in self.audio.call_args_list], [first, second])
        for video in (first, second):
            output = video.with_suffix(".topics.txt").read_text()
            self.assertIn(video.name, output)
            self.assertIn("00:00:05 — Planting", output)
        self.codex.assert_called_once()
        self.codex.return_value.check_login.assert_called_once()
        self.assertFalse((nested / "hidden.topics.txt").exists())

    def test_output_and_transcript_directories(self):
        source = self.video("lecture.mp4")
        output, saved = self.root / "outlines", self.root / "transcripts"
        output.mkdir()
        saved.mkdir()
        code, _, _ = self.run_folder("-o", output, "--save-transcript", saved)
        self.assertEqual(code, 0)
        self.assertTrue((output / "lecture.topics.txt").is_file())
        transcript = Transcript.model_validate_json((saved / "lecture.transcript.json").read_text())
        self.assertEqual(transcript.source, source.name)
        self.assertFalse(source.with_suffix(".topics.txt").exists())

    def test_transcribe_only_folder_without_codex(self):
        first, second = self.video("one.mp4"), self.video("two.mp4")
        code, _, _ = self.run_folder("--transcribe-only")
        self.assertEqual(code, 0)
        self.codex.assert_not_called()
        for source in (first, second):
            self.assertTrue(source.with_suffix(".transcript.json").is_file())

    def test_existing_outputs_are_skipped_without_loading_backends(self):
        source = self.video("lecture.mp4")
        output = source.with_suffix(".topics.txt")
        output.write_text("Keep this outline")
        code, stdout, _ = self.run_folder()
        self.assertEqual(code, 0)
        self.assertIn("0 processed, 1 skipped, 0 failed", stdout)
        self.assertEqual(output.read_text(), "Keep this outline")
        self.audio.assert_not_called()
        self.codex.assert_not_called()

    def test_force_replaces_existing_outline(self):
        source = self.video("lecture.mp4")
        output = source.with_suffix(".topics.txt")
        output.write_text("Old outline")
        code, _, _ = self.run_folder("--force")
        self.assertEqual(code, 0)
        self.assertIn("Planting", output.read_text())

    def test_failed_video_does_not_stop_later_videos(self):
        self.video("a-broken.mp4")
        good = self.video("b-good.mp4")
        def transcribe(source, **kwargs):
            if source.name == "a-broken.mp4":
                raise ToolError("No audio track")
            return self.transcript(source)
        self.audio.side_effect = transcribe
        code, stdout, stderr = self.run_folder()
        self.assertEqual(code, 1)
        self.assertIn("1 processed, 0 skipped, 1 failed", stdout)
        self.assertIn("a-broken.mp4: No audio track", stderr)
        self.assertTrue(good.with_suffix(".topics.txt").exists())

    def test_empty_folder_is_reported_without_starting_backends(self):
        self.video("notes.txt")
        code, _, stderr = self.run_folder()
        self.assertEqual(code, 1)
        self.assertIn("No MP4 files found", stderr)
        self.audio.assert_not_called()
        self.codex.assert_not_called()

    def test_folder_requires_directory_for_output_options(self):
        self.video("lecture.mp4")
        for flag in ("-o", "--save-transcript"):
            with self.subTest(flag=flag):
                code, _, stderr = self.run_folder(flag, self.root / "not-a-directory.txt")
                self.assertEqual(code, 1)
                self.assertIn("must be an existing directory", stderr)
        self.audio.assert_not_called()

    def test_duplicate_output_names_fail_before_any_processing_even_with_force(self):
        self.video("lecture.mp4")
        self.video("lecture.MP4")
        code, _, stderr = self.run_folder("--force")
        self.assertEqual(code, 1)
        self.assertIn("Multiple batch outputs", stderr)
        self.audio.assert_not_called()

    def test_output_alias_cannot_overwrite_another_input(self):
        first = self.video("a.mp4")
        second = self.video("b.mp4")
        second.write_bytes(b"Original video content")
        os.link(second, first.with_suffix(".topics.txt"))
        code, stdout, stderr = self.run_folder("--force")
        self.assertEqual(code, 1)
        self.assertIn("1 processed, 0 skipped, 1 failed", stdout)
        self.assertIn("must not replace an input", stderr)
        self.assertEqual(second.read_bytes(), b"Original video content")

    def test_keyboard_interrupt_stops_batch(self):
        self.video("a.mp4")
        self.video("b.mp4")
        self.audio.side_effect = KeyboardInterrupt()
        code, _, stderr = self.run_folder()
        self.assertEqual(code, 130)
        self.assertIn("Cancelled", stderr)
        self.assertEqual(self.audio.call_count, 1)


if __name__ == "__main__":
    unittest.main()
