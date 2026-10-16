import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"

MODEL = "small.en"


def save_json(path, data):
    """Save JSON atomically."""

    temporary = path.with_name(path.name + ".tmp")

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def extract_audio(video_path, output_path, start, duration):
    """Extract selected audio as mono 16 kHz WAV."""

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]

    subprocess.run(command, check=True)


def generate_word_timestamps(video_id, max_seconds=None):
    """Transcribe selected audio with word-level timestamps."""

    video_dir = INPUT_DIR / video_id

    video_path = video_dir / "source.mp4"
    selection_path = video_dir / "selection.json"
    output_path = video_dir / "word_timestamps.json"

    if not video_path.exists():
        raise FileNotFoundError("Source video not found.")

    with selection_path.open(encoding="utf-8") as f:
        selection = json.load(f)

    clip_start = float(selection["start"])
    clip_end = float(selection["end"])

    if not 0 <= clip_start < clip_end:
        raise ValueError("Invalid clip time range.")

    if max_seconds is not None:
        if max_seconds <= 0:
            raise ValueError("max_seconds must be positive.")

        clip_end = min(
            clip_end,
            clip_start + max_seconds,
        )

    duration = clip_end - clip_start

    # Build a cache key for this exact audio range.
    cache_info = {
        "model": MODEL,
        "start": clip_start,
        "end": clip_end,
        "source_size": video_path.stat().st_size,
        "source_mtime": video_path.stat().st_mtime_ns,
    }

    cache_key = hashlib.sha256(
        json.dumps(
            cache_info,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            cached = json.load(f)

        if cached.get("cache_key") == cache_key:
            print("Using cached word timestamps.")
            return output_path

    print("\nPreparing audio...")
    print(f"Duration: {duration:.2f} seconds")

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "clip.wav"

        extract_audio(
            video_path,
            audio_path,
            clip_start,
            duration,
        )

        print("\nLoading Whisper model...")

        model = WhisperModel(
            MODEL,
            device="cpu",
            compute_type="int8",
        )

        print("Transcribing with word timestamps...")

        segments, _ = model.transcribe(
            str(audio_path),
            language="en",
            word_timestamps=True,
            vad_filter=False,
            beam_size=3,
            condition_on_previous_text=False,
        )

        words = []

        for segment in segments:
            for word in segment.words or []:

                if word.start is None or word.end is None:
                    continue

                text = word.word.strip()

                if not text:
                    continue

                relative_start = max(0.0, float(word.start))
                relative_end = min(
                    duration,
                    float(word.end),
                )

                if relative_end <= relative_start:
                    continue

                words.append({
                    "word": text,
                    "start": round(
                        clip_start + relative_start, 3
                    ),
                    "end": round(
                        clip_start + relative_end, 3
                    ),
                    "relative_start": round(
                        relative_start, 3
                    ),
                    "relative_end": round(
                        relative_end, 3
                    ),
                })

    if not words:
        raise RuntimeError("No words were detected.")

    data = {
        "source": "faster-whisper",
        "model": MODEL,
        "cache_key": cache_key,
        "clip": {
            "start": clip_start,
            "end": clip_end,
        },
        "words": words,
    }

    save_json(output_path, data)

    print("\nWord transcription completed!")
    print(f"Words detected: {len(words)}")
    print(f"Saved to: {output_path}")

    print("\nFirst 20 words:")

    for word in words[:20]:
        print(
            f"{word['relative_start']:6.2f} - "
            f"{word['relative_end']:6.2f}  "
            f"{word['word']}"
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate word timestamps for TransClip"
    )

    parser.add_argument("video_id")

    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    try:
        generate_word_timestamps(
            args.video_id,
            max_seconds=args.max_seconds,
        )
    except Exception as error:
        print(f"Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
