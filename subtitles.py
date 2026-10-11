import html
import json
import re
import subprocess
from pathlib import Path
from caption_cleaner import clean_rolling_captions


def parse_timestamp(value):
    """Convert a subtitle timestamp into seconds."""
    parts = value.replace(",", ".").split(":")

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)

    raise ValueError("Invalid timestamp")


def parse_subtitles(path):
    """Read VTT/SRT subtitles into timestamped segments."""
    content = path.read_text(encoding="utf-8-sig")
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    segments = []

    for block in re.split(r"\n\s*\n", content):
        lines = block.strip().splitlines()

        for index, line in enumerate(lines):
            if "-->" not in line:
                continue

            match = re.match(r"(\S+)\s+-->\s+(\S+)", line)

            if not match:
                continue

            try:
                start = parse_timestamp(match.group(1))
                end = parse_timestamp(match.group(2))
            except ValueError:
                continue

            text = " ".join(lines[index + 1:])
            text = re.sub(r"<[^>]+>", "", text)
            text = html.unescape(text).strip()

            if text and 0 <= start < end:
                segments.append({
                    "start": start,
                    "end": end,
                    "text": text
                })

            break

    return clean_rolling_captions(segments)


def subtitle_candidates(metadata):
    """List English subtitles, prioritizing manual captions."""
    candidates = []

    sources = [
        ("youtube_manual", "subtitles"),
        ("youtube_auto", "automatic_captions")
    ]

    for source_name, key in sources:
        available = metadata.get(key) or {}

        languages = [
            lang for lang in available
            if lang == "en" or lang.startswith("en-")
        ]

        def priority(lang):
            if source_name == "youtube_manual":
                return (lang != "en", lang)

            return (
                0 if lang == "en-orig" else
                1 if lang == "en" else 2,
                lang
            )

        for language in sorted(languages, key=priority):
            candidates.append((source_name, language))

    return candidates


def save_transcript(video_dir, source, segments, language_code=None):
    """Save subtitles using a common JSON format."""
    data = {
        "source": source,
        "language": "en",
        "subtitle_language": language_code,
        "segments": segments
    }

    path = video_dir / "transcript.json"
    temporary = video_dir / "transcript.json.tmp"

    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    temporary.replace(path)

    print(f"Transcript saved: {path}")
    print(f"Segments: {len(segments)}")

    return path


def valid_transcript(data):
    """Check whether a saved transcript is usable."""
    if data.get("language") != "en":
        return False

    segments = data.get("segments")

    if not isinstance(segments, list) or not segments:
        return False

    return all(
        isinstance(s.get("start"), (int, float))
        and isinstance(s.get("end"), (int, float))
        and 0 <= s["start"] < s["end"]
        and isinstance(s.get("text"), str)
        and bool(s["text"].strip())
        for s in segments
        if isinstance(s, dict)
    ) and all(isinstance(s, dict) for s in segments)


def find_video(video_dir):
    """Find the downloaded source video."""
    for extension in ("mp4", "mkv", "webm", "mov"):
        path = video_dir / f"source.{extension}"

        if path.is_file():
            return path

    raise FileNotFoundError("Source video not found.")


def transcribe_with_whisper(video_dir):
    """Transcribe English speech locally."""
    from faster_whisper import WhisperModel

    video_path = find_video(video_dir)

    print("\nSwitching to local Whisper transcription...")

    model = WhisperModel(
        "small.en",
        device="cpu",
        compute_type="int8"
    )

    results, _ = model.transcribe(
        str(video_path),
        language="en",
        vad_filter=True
    )

    segments = []

    for segment in results:
        if segment.text.strip():
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

    if not segments:
        raise RuntimeError("Whisper produced no transcript.")

    return save_transcript(video_dir, "whisper", segments)


def ensure_transcript(video_dir, url=None):
    """Automatically obtain or generate an English transcript."""
    video_dir = Path(video_dir)

    transcript_path = video_dir / "transcript.json"

    # 1. Reuse an existing transcript.
    if transcript_path.exists():
        try:
            data = json.loads(
                transcript_path.read_text(encoding="utf-8")
            )

            if valid_transcript(data):
                print("Using cached transcript.")
                return transcript_path

        except (OSError, ValueError):
            pass

        print("Cached transcript is invalid. Regenerating...")

    # 2. Read the available subtitle languages.
    metadata_path = video_dir / "source.info.json"

    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    candidates = subtitle_candidates(metadata)

    print("\nChecking available English subtitles...")

    for source, language in candidates:
        print(f"Trying {language} ({source})...")

        def find_caption():
            for extension in ("vtt", "srt"):
                path = video_dir / f"source.{language}.{extension}"

                if path.exists():
                    segments = parse_subtitles(path)

                    if segments:
                        return segments

                    print(f"Invalid subtitle file: {path.name}")

            return None

        # First try already downloaded subtitles.
        segments = find_caption()

        if segments:
            return save_transcript(
                video_dir, source, segments, language
            )

        # Download the selected subtitle track.
        if url:
            option = (
                "--write-subs"
                if source == "youtube_manual"
                else "--write-auto-subs"
            )

            command = [
                "yt-dlp",
                "--no-playlist",
                "--skip-download",
                option,
                "--sub-langs", f"^{re.escape(language)}$",
                "--sub-format", "vtt/srt",
                "--retries", "1",
                "--retry-sleep", "http:3",
                "-o", str(video_dir / "source.%(ext)s"),
                url
            ]

            result = subprocess.run(command, check=False)

            if result.returncode != 0:
                print("Subtitle download failed. Trying another source.")
                continue

            segments = find_caption()

            if segments:
                return save_transcript(
                    video_dir, source, segments, language
                )

    # 3. Fall back to local speech recognition.
    print("\nNo usable English subtitles found.")

    return transcribe_with_whisper(video_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python subtitles.py VIDEO_ID")
        sys.exit(1)

    base_dir = Path(__file__).resolve().parent
    video_dir = base_dir / "input" / sys.argv[1]

    try:
        ensure_transcript(video_dir)
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)
