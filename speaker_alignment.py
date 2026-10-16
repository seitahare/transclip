import json
import sys
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"


def overlap(a_start, a_end, b_start, b_end):
    """Calculate the overlap between two time intervals."""
    return max(
        0.0,
        min(a_end, b_end) - max(a_start, b_start)
    )


def covered_duration(turns, start, end):
    """Measure speaker coverage without double-counting intervals."""

    intervals = []

    for turn in turns:
        left = max(start, turn["relative_start"])
        right = min(end, turn["relative_end"])

        if left < right:
            intervals.append((left, right))

    if not intervals:
        return 0.0

    intervals.sort()

    total = 0.0
    left, right = intervals[0]

    for next_left, next_right in intervals[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            total += right - left
            left, right = next_left, next_right

    total += right - left

    return total


def has_simultaneous_speech(turns, start, end):
    """Detect overlapping speech from different speakers."""

    for i, first in enumerate(turns):
        for second in turns[i + 1:]:
            if first["speaker"] == second["speaker"]:
                continue

            left = max(
                start,
                first["relative_start"],
                second["relative_start"],
            )

            right = min(
                end,
                first["relative_end"],
                second["relative_end"],
            )

            if right - left >= 0.05:
                return True

    return False


def align_speakers(video_id):
    """Match translated subtitles with estimated speaker turns."""

    video_dir = INPUT_DIR / video_id

    translation_path = video_dir / "translation.json"
    diarization_path = video_dir / "diarization.json"
    output_path = video_dir / "speaker_segments.json"

    with translation_path.open(encoding="utf-8") as f:
        translation = json.load(f)

    with diarization_path.open(encoding="utf-8") as f:
        diarization = json.load(f)

    # Check that both files describe the same clip.
    for key in ("start", "end"):
        difference = abs(
            float(translation["clip"][key])
            - float(diarization["clip"][key])
        )

        if difference > 0.1:
            raise ValueError(
                "Translation and diarization use different clip ranges."
            )

    turns = diarization["turns"]
    results = []

    for segment in translation["segments"]:
        start = float(segment["relative_start"])
        end = float(segment["relative_end"])

        if end <= start:
            raise ValueError("Invalid subtitle time range.")

        duration = end - start

        grouped_turns = defaultdict(list)

        for turn in turns:
            if overlap(
                start,
                end,
                turn["relative_start"],
                turn["relative_end"],
            ) > 0:
                grouped_turns[turn["speaker"]].append(turn)

        candidates = []

        for speaker, speaker_turns in grouped_turns.items():
            seconds = covered_duration(
                speaker_turns,
                start,
                end,
            )

            if seconds > 0:
                candidates.append({
                    "speaker": speaker,
                    "overlap_seconds": round(seconds, 3),
                    "coverage": round(seconds / duration, 3),
                })

        candidates.sort(
            key=lambda item: item["overlap_seconds"],
            reverse=True,
        )

        # Ignore very small overlaps when deciding whether
        # a subtitle contains multiple speakers.
        threshold = max(0.15, duration * 0.05)

        meaningful = [
            candidate
            for candidate in candidates
            if candidate["overlap_seconds"] >= threshold
        ]

        if len(meaningful) >= 2:
            status = "mixed"
            speaker = None

        elif candidates and candidates[0]["coverage"] >= 0.25:
            status = "single"
            speaker = candidates[0]["speaker"]

        elif candidates:
            status = "uncertain"
            speaker = None

        else:
            status = "no_speech"
            speaker = None

        results.append({
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "relative_start": start,
            "relative_end": end,
            "speaker": speaker,
            "status": status,
            "overlapping_speech": has_simultaneous_speech(
                turns, start, end
            ),
            "speaker_candidates": candidates,
            "text_en": segment["text_en"],
            "text_ja": segment["text_ja"],
        })

    output = {
        "video_id": video_id,
        "clip": translation["clip"],
        "speakers": diarization["speakers"],
        "turns": turns,
        "segments": results,
    }

    temporary = output_path.with_name(
        output_path.name + ".tmp"
    )

    temporary.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary.replace(output_path)

    print("\nSpeaker alignment completed!")
    print(f"Segments: {len(results)}")

    for status in ("single", "mixed", "uncertain", "no_speech"):
        count = sum(
            segment["status"] == status
            for segment in results
        )

        print(f"{status}: {count}")

    print(f"\nSaved to: {output_path}")

    print("\nFirst 10 segments:")
    print("-" * 50)

    for segment in results[:10]:
        print(
            f"[{segment['relative_start']:.2f} - "
            f"{segment['relative_end']:.2f}] "
            f"{segment['speaker'] or segment['status']}"
        )

        print(f"EN: {segment['text_en']}")
        print(f"JA: {segment['text_ja']}")
        print()

    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python speaker_alignment.py VIDEO_ID")
        sys.exit(1)

    try:
        align_speakers(sys.argv[1])
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)
