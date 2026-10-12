import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from selector import select_clip

from subtitles import ensure_transcript


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"


def check_dependencies():
    """Check required external programs."""
    for program in ("yt-dlp", "ffmpeg"):
        if shutil.which(program) is None:
            print(f"Error: {program} is not installed.")
            return False

    return True


def get_video_id(url):
    """Get the YouTube video ID without downloading the video."""
    result = subprocess.run(
        ["yt-dlp", "--no-playlist", "--skip-download",
         "--print", "id", url],
        capture_output=True,
        text=True,
        check=True
    )

    lines = result.stdout.strip().splitlines()

    if not lines:
        raise ValueError("Could not identify the video.")

    video_id = lines[-1]

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Invalid YouTube video ID.")

    return video_id


def has_video(video_dir):
    """Check whether the source video already exists."""
    return any(
        (video_dir / f"source.{ext}").is_file()
        for ext in ("mp4", "mkv", "webm", "mov")
    )


def fetch(url):
    """Fetch a video and prepare its English transcript."""

    if not check_dependencies():
        return 1

    INPUT_DIR.mkdir(exist_ok=True)

    try:
        video_id = get_video_id(url)
    except (subprocess.CalledProcessError, ValueError) as error:
        print(f"Error: Could not identify video: {error}")
        return 1

    video_dir = INPUT_DIR / video_id
    metadata_path = video_dir / "source.info.json"

    print("\n========================================")
    print("               TransClip")
    print("========================================")
    print(f"\nVideo ID: {video_id}")

    # Step 1: Download video and metadata if necessary.
    if has_video(video_dir) and metadata_path.exists():
        print("\nUsing cached video and metadata.")
    else:
        print("\nDownloading video and metadata...")

        output_template = str(
            INPUT_DIR / "%(id)s" / "source.%(ext)s"
        )

        command = [
            "yt-dlp",
            "--no-playlist",
            "--no-overwrites",
            "--write-info-json",
            "--merge-output-format", "mp4",
            "-o", output_template,
            url
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            print("Error: Video download failed.")
            return 1

    if not has_video(video_dir) or not metadata_path.exists():
        print("Error: Video or metadata is missing.")
        return 1

    # Step 2: Obtain or generate the transcript.
    print("\nPreparing English transcript...")

    try:
        transcript_path = ensure_transcript(
            video_dir,
            url=url
        )
    except Exception as error:
        print(f"Error: Transcript preparation failed: {error}")
        return 1

    # Step 3: Report the result.
    print("\n========================================")
    print("             Fetch Complete")
    print("========================================")

    print(f"Video: {video_dir}")
    print(f"Transcript: {transcript_path}")

    return select_clip(video_id)


def main():
    parser = argparse.ArgumentParser(
        description="TransClip - AI Video Dubbing"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Download a video and prepare its transcript"
    )

    fetch_parser.add_argument("url")

    args = parser.parse_args()

    if args.command == "fetch":
        return fetch(args.url)

    return 0


if __name__ == "__main__":
    sys.exit(main())
