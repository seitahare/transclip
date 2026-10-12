import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"


def create_clip(video_id):
    """Create a video clip using a saved time range."""

    video_dir = INPUT_DIR / video_id

    video_path = video_dir / "source.mp4"
    selection_path = video_dir / "selection.json"

    if not video_path.exists():
        print("Error: Source video not found.")
        return 1

    if not selection_path.exists():
        print("Error: Selection not found.")
        return 1

    with selection_path.open(encoding="utf-8") as f:
        selection = json.load(f)

    start = selection["start"]
    end = selection["end"]

    if not 0 <= start < end:
        print("Error: Invalid time range.")
        return 1

    output_dir = OUTPUT_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"clip_{start}_{end}.mp4"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(end - start),
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output_path),
    ]

    print("Creating video clip...")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to create clip.")
        return 1

    print(f"Clip created: {output_path}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clipper.py VIDEO_ID")
        sys.exit(1)

    sys.exit(create_clip(sys.argv[1]))
