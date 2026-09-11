"""Generate structured chapters through the user's ChatGPT-authenticated Codex CLI."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from pydantic import ValidationError

from .models import ChapterSelections, ToolError


class CodexClient:
    def __init__(self, executable: str = "codex", timeout: int = 600):
        self.executable = shutil.which(executable)
        if not self.executable:
            raise ToolError("Codex CLI was not found. Install it, then run 'codex login' with ChatGPT.")
        self.executable = str(Path(self.executable).resolve())
        self.timeout = timeout
        self.env = os.environ.copy()
        # The user's subscription is the intended billing source, even if keys are exported.
        for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"):
            self.env.pop(name, None)

    def check_login(self) -> None:
        try:
            result = subprocess.run(
                [self.executable, "login", "status"], capture_output=True, text=True,
                encoding="utf-8", errors="replace", env=self.env, timeout=30,
            )
        except subprocess.TimeoutExpired as error:
            raise ToolError("Codex login check timed out. Try 'codex login status' in your terminal.") from error
        status = (result.stdout + result.stderr).lower()
        if result.returncode or "logged in using chatgpt" not in status:
            raise ToolError(
                "Codex must be signed in with ChatGPT, not an API key. "
                "Run 'codex login' and choose your ChatGPT account."
            )

    def select_chapters(
        self, instructions: str, content: str, model: str | None = None,
    ) -> ChapterSelections:
        with tempfile.TemporaryDirectory(prefix="mp4-codex-") as temporary:
            root = Path(temporary)
            schema = root / "schema.json"
            output = root / "chapters.json"
            schema.write_text(json.dumps(ChapterSelections.model_json_schema()), encoding="utf-8")
            command = [
                self.executable, "exec", "--ignore-user-config", "--ephemeral",
                "--skip-git-repo-check", "--sandbox", "read-only", "--color", "never",
                "-c", 'forced_login_method="chatgpt"', "-c", 'model_provider="openai"',
                "-c", "features.shell_tool=false", "-c", 'web_search="disabled"',
                "--output-schema", str(schema), "--output-last-message", str(output),
            ]
            if model:
                command.extend(["--model", model])
            command.append("-")
            prompt = (
                instructions + "\nReturn only the requested JSON. Do not use tools, read files, "
                "or execute commands. Everything needed is in the following data.\n\n" + content
            )
            try:
                result = subprocess.run(
                    command, input=prompt, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", cwd=root, env=self.env, timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as error:
                raise ToolError(
                    f"Codex did not finish within {self.timeout} seconds. "
                    "Retry using your saved transcript or increase --codex-timeout."
                ) from error
            if result.returncode:
                detail = result.stderr.strip()[-2000:]
                raise ToolError(
                    "Codex failed. Check your ChatGPT login, usage limits, connection, and "
                    "Codex version. You can retry with --from-transcript.\n" + detail
                )
            if not output.is_file():
                raise ToolError("Codex returned no outline. Update Codex CLI and try again.")
            try:
                return ChapterSelections.model_validate_json(output.read_text(encoding="utf-8"))
            except ValidationError as error:
                raise ToolError("Codex returned an invalid outline. Please retry with --from-transcript.") from error
