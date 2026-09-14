import argparse
import os
from pathlib import Path
import sys
import tempfile

from pydantic import ValidationError

from .codex import CodexClient
from .media import transcribe
from .models import ToolError, Transcript
from .outline import DETAIL, generate_outline, render_outline


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Outline an MP4 or a folder of MP4s using local Whisper and your ChatGPT Codex login.",
        epilog="Audio stays local. Transcript text is sent to Codex and uses your ChatGPT plan allowance.",
    )
    result.add_argument("video", type=Path, nargs="?", help="path to an MP4 video or a folder of MP4 videos")
    result.add_argument("-o", "--output", type=Path, help="output file, or existing output directory for folder input (default: beside each video)")
    result.add_argument("--language", help="spoken language code, e.g. en or fi (default: auto-detect)")
    result.add_argument("--outline-language", help="language for topic titles and summaries, e.g. English")
    result.add_argument("--detail", choices=DETAIL, default="balanced")
    result.add_argument("--model", help="Codex outline model (default: Codex CLI's built-in default)")
    result.add_argument("--whisper-model", default="base", help="local Whisper model name or path (default: base)")
    result.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="transcription device (default: cpu)")
    result.add_argument("--model-dir", type=Path, help="Whisper model cache directory")
    result.add_argument("--offline", action="store_true", help="use a cached Whisper model without downloading; Codex still needs internet")
    result.add_argument("--codex-bin", default="codex", help="Codex executable name or path")
    result.add_argument("--codex-timeout", type=int, default=600, help="seconds allowed per Codex request (default: 600)")
    result.add_argument("--transcribe-only", action="store_true", help="save only local transcript JSON; no Codex login required")
    result.add_argument("--save-transcript", type=Path, help="also save transcript JSON: file path for a video, existing directory for folder input")
    result.add_argument("--from-transcript", type=Path, help="reuse a saved transcript instead of an MP4")
    result.add_argument("--force", action="store_true", help="replace existing output files")
    return result


def check_output(path: Path, inputs: list[Path], force: bool) -> None:
    if any(path.resolve() == source.resolve() or
           (path.exists() and source.exists() and path.samefile(source)) for source in inputs):
        raise ToolError(f"Output must not replace an input file: {path}")
    if path.exists() and (not force or not path.is_file()):
        raise ToolError(f"Output already exists: {path}. Choose another path or use --force.")
    if not path.parent.is_dir():
        raise ToolError(f"Output directory does not exist: {path.parent}")


def write_text(path: Path, text: str, force: bool) -> None:
    # Stage complete UTF-8 output beside its destination; preserve prior files on errors.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(text)
            f.close()
            if force:
                os.replace(temporary, path)
            else:
                # Atomic creation without overwriting a file created since preflight.
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def process_file(args, get_client, protected_inputs=()) -> int:
    source = args.video or args.from_transcript
    if not source.is_file():
        raise ToolError(f"Input file not found: {source}")
    if args.video and source.suffix.lower() != ".mp4":
        raise ToolError("Please provide an .mp4 file.")
    output = args.output or source.with_suffix(".transcript.json" if args.transcribe_only else ".topics.txt")
    check_output(output, [source, *protected_inputs], args.force)
    if args.save_transcript:
        check_output(args.save_transcript, [source, output, *protected_inputs], args.force)
    transcript = None
    if args.from_transcript:
        transcript = Transcript.model_validate_json(source.read_text(encoding="utf-8"))
        if not transcript.segments:
            raise ToolError("The saved transcript contains no speech segments.")
    client = None
    if not args.transcribe_only:
        client = get_client()
    if transcript is None:
        transcript = transcribe(
            source, language=args.language, model_name=args.whisper_model,
            device=args.device, model_dir=args.model_dir, offline=args.offline,
        )
    if args.transcribe_only:
        write_text(output, transcript.model_dump_json(indent=2) + "\n", args.force)
        print(f"Saved {len(transcript.segments)} transcript segments to {output}")
        return 0
    if args.save_transcript:
        write_text(args.save_transcript, transcript.model_dump_json(indent=2) + "\n", args.force)
        print(f"Saved transcript: {args.save_transcript}", file=sys.stderr)
    topics = generate_outline(
        transcript, client, args.model, args.detail, args.outline_language,
    )
    write_text(output, render_outline(transcript.source, topics), args.force)
    print(f"Saved {len(topics)} topics to {output}")
    return 0


def process_folder(args, get_client) -> int:
    sources = sorted(
        (path for path in args.video.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not sources:
        raise ToolError(f"No MP4 files found directly inside: {args.video}")
    for label, directory in (("--output", args.output), ("--save-transcript", args.save_transcript)):
        if directory is not None and not directory.is_dir():
            raise ToolError(f"With folder input, {label} must be an existing directory: {directory}")

    # Plan every destination before processing. For example, lesson.mp4 and
    # lesson.MP4 would otherwise silently share one outline, even with --force.
    jobs = []
    destinations = set()
    inodes = set()
    suffix = ".transcript.json" if args.transcribe_only else ".topics.txt"
    for source in sources:
        output = (args.output or source.parent) / source.with_suffix(suffix).name
        saved = (args.save_transcript / source.with_suffix(".transcript.json").name
                 if args.save_transcript else None)
        for target in (output, saved):
            if target is None:
                continue
            resolved = target.resolve()
            stat = target.stat() if target.exists() else None
            inode = (stat.st_dev, stat.st_ino) if stat else None
            if resolved in destinations or (inode is not None and inode in inodes):
                raise ToolError(f"Multiple batch outputs refer to the same file: {target}. Rename the conflicting files.")
            destinations.add(resolved)
            if inode is not None:
                inodes.add(inode)
        jobs.append(argparse.Namespace(**(vars(args) | {
            "video": source, "output": output, "save_transcript": saved,
        })))

    completed = skipped = failed = 0
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job.video.name}", file=sys.stderr)
        if job.output.is_file() and not args.force:
            print(f"Skipping existing output: {job.output}", file=sys.stderr)
            skipped += 1
            continue
        try:
            process_file(job, get_client, protected_inputs=sources)
            completed += 1
        except (ToolError, OSError, UnicodeError, ValidationError) as error:
            message = ("Invalid transcript or model response." if isinstance(error, ValidationError)
                       else str(error))
            print(f"Error processing {job.video.name}: {message}", file=sys.stderr)
            failed += 1
    print(f"Finished: {completed} processed, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if bool(args.video) == bool(args.from_transcript):
        arg_parser.error("provide either an MP4 video/folder or --from-transcript, but not both")
    if args.transcribe_only and (args.from_transcript or args.save_transcript):
        arg_parser.error("--transcribe-only takes a video/folder and optional -o; do not combine it with transcript flags")
    if args.codex_timeout <= 0:
        arg_parser.error("--codex-timeout must be positive")
    try:
        client = None

        def get_client():
            nonlocal client
            if client is None:
                candidate = CodexClient(args.codex_bin, args.codex_timeout)
                candidate.check_login()
                client = candidate
            return client

        if args.video and args.video.is_dir():
            return process_folder(args, get_client)
        return process_file(args, get_client)
    except ValidationError:
        message = "Invalid transcript or model response. Check the saved JSON, or retry the request."
    except (ToolError, OSError, UnicodeError) as error:
        message = str(error)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    print(f"Error: {message}", file=sys.stderr)
    return 1
