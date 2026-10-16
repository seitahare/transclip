import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch

from pyannote.audio import Pipeline


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"

MODEL = "pyannote/speaker-diarization-community-1"
FORMAT_VERSION = 1


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


def extract_audio(video_path, audio_path, start, duration):
    """Extract mono 16 kHz WAV from the selected video range."""

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
        str(audio_path),
    ]

    subprocess.run(command, check=True)


def load_waveform(audio_path):
    """Load PCM16 WAV without TorchCodec."""

    with wave.open(str(audio_path), "rb") as audio:

        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()

        if channels != 1:
            raise ValueError("Expected mono audio.")

        if sample_rate != 16000:
            raise ValueError("Expected 16 kHz audio.")

        if sample_width != 2:
            raise ValueError("Expected 16-bit PCM audio.")

        samples = audio.readframes(
            audio.getnframes()
        )

    pcm = np.frombuffer(
        samples,
        dtype="<i2",
    ).copy()

    waveform = (
        torch.from_numpy(pcm)
        .float()
        .unsqueeze(0)
        / 32768.0
    )

    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


def collect_turns(annotation, clip_start):
    """Convert speaker turns into JSON records."""

    turns = []

    for segment, _, speaker in annotation.itertracks(
        yield_label=True
    ):

        start = float(segment.start)
        end = float(segment.end)

        if end <= start:
            continue

        turns.append({
            "speaker": speaker,
            "start": round(clip_start + start, 3),
            "end": round(clip_start + end, 3),
            "relative_start": round(start, 3),
            "relative_end": round(end, 3),
        })

    turns.sort(
        key=lambda item: (
            item["relative_start"],
            item["relative_end"],
        )
    )

    return turns


def diarize_clip(video_id, max_seconds=None):
    """Identify speakers in the selected clip."""

    video_dir = INPUT_DIR / video_id

    video_path = video_dir / "source.mp4"
    selection_path = video_dir / "selection.json"
    output_path = video_dir / "diarization.json"

    # ----------------------------------------
    # Step 1: Validate input files
    # ----------------------------------------

    if not video_path.exists():
        raise FileNotFoundError(
            "Source video not found."
        )

    if not selection_path.exists():
        raise FileNotFoundError(
            "Selection not found."
        )

    with selection_path.open(encoding="utf-8") as f:
        selection = json.load(f)

    clip_start = float(selection["start"])
    clip_end = float(selection["end"])

    if not 0 <= clip_start < clip_end:
        raise ValueError(
            "Invalid clip time range."
        )

    # Limit processing time for testing.
    if max_seconds is not None:

        if max_seconds <= 0:
            raise ValueError(
                "max_seconds must be positive."
            )

        clip_end = min(
            clip_end,
            clip_start + max_seconds,
        )

    duration = clip_end - clip_start

    # ----------------------------------------
    # Step 2: Prepare cache information
    # ----------------------------------------

    cache_info = {
        "model": MODEL,
        "format_version": FORMAT_VERSION,
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
            print("Using cached diarization.")
            return output_path

    # ----------------------------------------
    # Step 3: Extract audio
    # ----------------------------------------

    print("\nPreparing audio...")
    print(f"Clip duration: {duration:.2f} seconds")

    with tempfile.TemporaryDirectory() as temp_dir:

        audio_path = Path(temp_dir) / "clip.wav"

        extract_audio(
            video_path,
            audio_path,
            clip_start,
            duration,
        )

        audio_data = load_waveform(audio_path)

        # ----------------------------------------
        # Step 4: Load pyannote model
        # ----------------------------------------

        print("\nLoading speaker diarization model...")

        token = os.getenv("HF_TOKEN")

        if not token:
            raise RuntimeError(
                "HF_TOKEN is not configured."
            )

        pipeline = Pipeline.from_pretrained(
            MODEL,
            token=token,
        )

        if pipeline is None:
            raise RuntimeError(
                "Could not load diarization model."
            )

        # Start with CPU for compatibility.
        pipeline.to(torch.device("cpu"))

        # ----------------------------------------
        # Step 5: Identify speakers
        # ----------------------------------------

        print("\nIdentifying speakers...")

        result = pipeline(audio_data)

    # ----------------------------------------
    # Step 6: Collect speaker turns
    # ----------------------------------------

    # Regular diarization preserves overlapping speech.
    turns = collect_turns(
        result.speaker_diarization,
        clip_start,
    )

    # Exclusive diarization is useful for subtitle alignment.
    exclusive_turns = collect_turns(
        result.exclusive_speaker_diarization,
        clip_start,
    )

    if not turns:
        raise RuntimeError(
            "No speaker turns detected."
        )

    speakers = sorted({
        turn["speaker"]
        for turn in turns
    })

    # ----------------------------------------
    # Step 7: Save results
    # ----------------------------------------

    data = {
        "model": MODEL,
        "cache_key": cache_key,
        "clip": {
            "start": clip_start,
            "end": clip_end,
        },
        "speakers": speakers,
        "turns": turns,
        "exclusive_turns": exclusive_turns,
    }

    save_json(output_path, data)

    # ----------------------------------------
    # Step 8: Display results
    # ----------------------------------------

    print("\nDiarization completed!")

    print(f"Speakers detected: {len(speakers)}")
    print(f"Speaker turns: {len(turns)}")
    print(f"Saved to: {output_path}")

    print("\nFirst 10 speaker turns:")
    print("-" * 40)

    for turn in turns[:10]:

        print(
            f"{turn['relative_start']:7.2f} - "
            f"{turn['relative_end']:7.2f}  "
            f"{turn['speaker']}"
        )

    return output_path


def main():

    parser = argparse.ArgumentParser(
        description="Identify speakers in a TransClip video"
    )

    parser.add_argument("video_id")

    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Analyze only the first N seconds of the selected clip",
    )

    args = parser.parse_args()

    try:
        diarize_clip(
            args.video_id,
            max_seconds=args.max_seconds,
        )

    except Exception as error:
        print(f"Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
