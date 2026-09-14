import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from mp4_timestamper.cli import main
from mp4_timestamper.codex import CodexClient
from mp4_timestamper.compilation import (
    SourceTopic, discover_outlines, generate_compilation, item_batches,
    read_outline, render_compilation,
)
from mp4_timestamper.models import Topic, TopicCompilation, ToolError
from mp4_timestamper.outline import render_outline


def compiled(groups):
    return TopicCompilation(
        overview="The lessons cover preparing soil and watering plants.",
        sections=[{"title": f"Theme {index}", "summary": "Related ideas from the lessons.",
                   "topic_ids": ids} for index, ids in enumerate(groups)],
    )


class CompilationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def outline(self, filename="one.topics.txt", video="one.mp4"):
        path = self.root / filename
        path.write_text(render_outline(video, [
            Topic(5, "Prepare soil", "Add compost to the soil."),
            Topic(3665, "Watering", "Water deeply at the roots."),
        ]), encoding="utf-8")
        return path

    def test_parse_generated_outline_preserves_provenance(self):
        path = self.outline("Finnish lesson.topics.txt", "Finnish lesson.mp4")
        topics = read_outline(path)
        self.assertEqual([t.timestamp for t in topics], ["00:00:05", "01:01:05"])
        self.assertEqual(topics[1].video, "Finnish lesson.mp4")
        self.assertEqual(topics[1].outline, path.name)
        self.assertEqual(topics[0].summary, "Add compost to the soil.")

    def test_parser_accepts_bom_crlf_and_multiline_summary(self):
        path = self.root / "custom.txt"
        path.write_bytes("\ufeffTopic outline: demo.mp4\r\n\r\n00:00:01 — Soil\r\n  First line.\r\n  Second line.\r\n".encode())
        self.assertEqual(read_outline(path)[0].summary, "First line. Second line.")

    def test_rejects_malformed_or_empty_outlines(self):
        for text in ("", "Unrelated notes", "Topic outline: x.mp4\n",
                     "Topic outline: x.mp4\n00:99:00 — Invalid\n  Summary",
                     "Topic outline: x.mp4\n00:00:00 — Missing summary",
                     "Topic outline: x.mp4\n00:00:10 — First\n  A\n00:00:05 — Second\n  B"):
            with self.subTest(text=text):
                path = self.root / "invalid.topics.txt"
                path.write_text(text)
                with self.assertRaises(ToolError):
                    read_outline(path)

    def test_discovery_custom_names_and_previous_compilation_exclusion(self):
        source = self.outline("custom.TXT")
        self.outline("other.topics.txt")
        output = self.root / "compilation.txt"
        output.write_text("Topic compilation: previous run\nOld compilation")
        files = discover_outlines(self.root, "*.txt", output)
        self.assertEqual(files, [source, self.root / "other.topics.txt"])
        self.assertEqual(discover_outlines(self.root, "*.topics.txt", output), [files[1]])

    def test_output_cannot_be_an_input_or_hardlink_alias(self):
        source = self.outline()
        alias = self.root / "alias.txt"
        os.link(source, alias)
        for output in (source, alias):
            with self.subTest(output=output), self.assertRaisesRegex(ToolError, "must not replace"):
                discover_outlines(self.root, "*.topics.txt", output)

    def test_cross_video_sections_use_original_timestamps(self):
        topics = read_outline(self.outline()) + read_outline(self.outline("two.topics.txt", "two.mp4"))
        client = Mock()
        client.compile_topics.return_value = compiled([[0, 2], [1, 3]])
        result = generate_compilation(topics, client, language="English")
        output = render_compilation(self.root, topics, result)
        first_section = output.split("2. Theme 1")[0]
        self.assertIn("one.mp4 @ 00:00:05", first_section)
        self.assertIn("two.mp4 @ 00:00:05", first_section)
        self.assertIn("[two.topics.txt]", output)
        self.assertIn("one.mp4 @ 01:01:05", output)
        self.assertIn("2 outlines, 4 topics", output)
        self.assertIn("Write in English", client.compile_topics.call_args.kwargs["instructions"])

    def test_missing_duplicate_and_invented_references_are_rejected(self):
        topics = read_outline(self.outline())
        for groups in ([[0]], [[0, 1, 3]], [[0, 1], [0]], [[]], []):
            with self.subTest(groups=groups):
                client = Mock()
                client.compile_topics.return_value = compiled(groups)
                with self.assertRaises(ToolError):
                    generate_compilation(topics, client)

    def test_staged_synthesis_preserves_all_original_references(self):
        topics = [SourceTopic(f"{i}.topics.txt", f"{i}.mp4", f"00:00:0{i}",
                              f"Topic {i}", "Long input. " * 40) for i in range(6)]
        client = Mock()
        def synthesize(**kwargs):
            items = json.loads(kwargs["content"])
            ids = [item["id"] for item in items]
            return compiled([ids])
        client.compile_topics.side_effect = synthesize
        result = generate_compilation(topics, client, batch_bytes=1400)
        self.assertGreater(client.compile_topics.call_count, 1)
        self.assertEqual(result.sections[0].topic_ids, list(range(6)))
        output = render_compilation(self.root, topics, result)
        for topic in topics:
            self.assertIn(f"{topic.video} @ {topic.timestamp}", output)
        self.assertIn("final compilation", client.compile_topics.call_args.kwargs["instructions"])

    def test_unicode_batches_and_oversized_single_topic(self):
        items = [{"id": i, "summary": "日本語" * 10} for i in range(10)]
        batches = list(item_batches(items, 300))
        self.assertEqual([item["id"] for batch in batches for item in batch], list(range(10)))
        for batch in batches:
            self.assertLessEqual(len(json.dumps(batch, ensure_ascii=False).encode()), 300)
        with self.assertRaisesRegex(ToolError, "too large"):
            list(item_batches([{"id": 0, "summary": "x" * 400}], 300))

    def test_cli_compiles_without_audio_and_protects_existing_output(self):
        self.outline()
        self.outline("two.topics.txt", "two.mp4")
        with patch("mp4_timestamper.cli.CodexClient") as constructor, \
                patch("mp4_timestamper.cli.transcribe") as transcribe, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            constructor.return_value.compile_topics.return_value = compiled([[0, 2], [1, 3]])
            self.assertEqual(main(["--compile", str(self.root)]), 0)
            constructor.return_value.check_login.assert_called_once()
            transcribe.assert_not_called()
            output = self.root / "compilation.txt"
            original = output.read_text()
            constructor.reset_mock()
            self.assertEqual(main(["--compile", str(self.root)]), 1)
            constructor.assert_not_called()
            self.assertEqual(output.read_text(), original)
            self.assertEqual(main(["--compile", str(self.root), "--topic-pattern", "*.txt", "--force"]), 0)
            items = json.loads(constructor.return_value.compile_topics.call_args.kwargs["content"])
            self.assertEqual(len(items), 4)

    def test_cli_empty_or_malformed_folder_never_calls_codex(self):
        with patch("mp4_timestamper.cli.CodexClient") as constructor, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--compile", str(self.root)]), 1)
            (self.root / "bad.topics.txt").write_text("Not an outline")
            self.assertEqual(main(["--compile", str(self.root)]), 1)
            constructor.assert_not_called()

    def test_conflicting_cli_modes(self):
        for extra in (["video.mp4"], ["--from-transcript", "saved.json"],
                      ["--transcribe-only"], ["--save-transcript", "saved.json"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["--compile", str(self.root), *extra])
                self.assertEqual(error.exception.code, 2)

    def test_codex_uses_compilation_schema_and_parses_result(self):
        def run(command, **kwargs):
            schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
            self.assertIn("sections", schema["properties"])
            self.assertFalse(schema["$defs"]["CompilationSection"]["additionalProperties"])
            output = Path(command[command.index("--output-last-message") + 1])
            output.write_text(compiled([[0]]).model_dump_json())
            return subprocess.CompletedProcess(command, 0, "", "")
        with patch("mp4_timestamper.codex.shutil.which", return_value="/usr/bin/codex"), \
                patch("mp4_timestamper.codex.subprocess.run", side_effect=run):
            result = CodexClient().compile_topics("Compile", "[]")
        self.assertEqual(result.sections[0].topic_ids, [0])


if __name__ == "__main__":
    unittest.main()
