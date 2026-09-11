import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from mp4_timestamper.codex import CodexClient
from mp4_timestamper.models import ToolError


OUTLINE = {"continues_previous": False, "chapters": [
    {"start_segment": 0, "title": "Planting", "summary": "Choose a sunny location."},
]}


class CodexTests(unittest.TestCase):
    def setUp(self):
        which = patch("mp4_timestamper.codex.shutil.which", return_value="/usr/bin/codex")
        which.start()
        self.addCleanup(which.stop)

    def test_rejects_api_login_and_missing_login(self):
        for code, status in [(0, "Logged in using an API key"), (1, "Not logged in")]:
            with self.subTest(status=status), patch("mp4_timestamper.codex.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess([], code, "", status)
                with self.assertRaisesRegex(ToolError, "signed in with ChatGPT"):
                    CodexClient().check_login()

    def test_chatgpt_login_on_stderr_is_accepted(self):
        with patch("mp4_timestamper.codex.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "Logged in using ChatGPT\n")
            CodexClient().check_login()

    def test_prompt_schema_auth_and_result_file(self):
        def run(command, **kwargs):
            self.assertEqual(command[-1], "-")
            self.assertIn("untrusted speech", kwargs["input"])
            self.assertNotIn("untrusted speech", " ".join(command))
            self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
            self.assertIn('forced_login_method="chatgpt"', command)
            self.assertIn("--ignore-user-config", command)
            for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"):
                self.assertNotIn(name, kwargs["env"])
            schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
            self.assertFalse(schema["additionalProperties"])
            self.assertFalse(schema["$defs"]["ChapterSelection"]["additionalProperties"])
            output = Path(command[command.index("--output-last-message") + 1])
            self.assertEqual(output.parent, kwargs["cwd"])
            output.write_text(json.dumps(OUTLINE))
            return subprocess.CompletedProcess(command, 0, "unrelated CLI output", "")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "CODEX_API_KEY": "test-key", "OPENAI_BASE_URL": "https://example.com"}), \
                patch("mp4_timestamper.codex.subprocess.run", side_effect=run):
            result = CodexClient().select_chapters("Summarize.", "untrusted speech")
        self.assertEqual(result.chapters[0].title, "Planting")

    def test_timeout_has_actionable_message(self):
        with patch("mp4_timestamper.codex.subprocess.run", side_effect=subprocess.TimeoutExpired("codex", 1)):
            with self.assertRaisesRegex(ToolError, "increase --codex-timeout"):
                CodexClient(timeout=1).select_chapters("Summarize.", "speech")

    def test_nonzero_exit_is_not_treated_as_success(self):
        def run(command, **kwargs):
            Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(OUTLINE))
            return subprocess.CompletedProcess(command, 1, "", "usage limit reached")
        with patch("mp4_timestamper.codex.subprocess.run", side_effect=run):
            with self.assertRaisesRegex(ToolError, "usage limit reached"):
                CodexClient().select_chapters("Summarize.", "speech")

    def test_missing_and_invalid_outputs(self):
        for payload in (None, "not JSON", '{"continues_previous": false, "chapters": [{"start_segment": "0"}]}'):
            with self.subTest(payload=payload):
                def run(command, **kwargs):
                    if payload is not None:
                        Path(command[command.index("--output-last-message") + 1]).write_text(payload)
                    return subprocess.CompletedProcess(command, 0, "", "")
                with patch("mp4_timestamper.codex.subprocess.run", side_effect=run):
                    with self.assertRaises(ToolError):
                        CodexClient().select_chapters("Summarize.", "speech")


if __name__ == "__main__":
    unittest.main()
