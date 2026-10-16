import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"

FORMAT_VERSION = 1


def overlap(a_start, a_end, b_start, b_end):
    """Calculate overlap between two time intervals."""
    return max(
        0.0,
        min(a_end, b_end) - max(a_start, b_start)
    )


def covered_duration(turns, start, end):
    """Measure coverage without double-counting intervals."""

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

    return total + right - left


def has_overlapping_speech(turns, start, end):
    """Check for simultaneous speech by different speakers."""

    active = [
        turn for turn in turns
        if overlap(
            start,
            end,
            turn["relative_start"],
            turn["relative_end"]
        ) > 0
    ]

    for i, first in enumerate(active):
        for second in active[i + 1:]:

            if first["speaker"] == second["speaker"]:
                continue

            common_start = max(
                start,
                first["relative_start"],
                second["relative_start"]
            )

            common_end = min(
                end,
                first["relative_end"],
                second["relative_end"]
            )

            if common_end - common_start >= 0.02:
                return True

    return False


def assign_speaker(word, turns):
    """Estimate a word's speaker using temporal overlap."""

    start = float(word["relative_start"])
    end = float(word["relative_end"])
    duration = end - start

    if duration <= 0:
        raise ValueError("Invalid word time range.")

    grouped = defaultdict(list)

    for turn in turns:
        seconds = overlap(
            start,
            end,
            turn["relative_start"],
            turn["relative_end"]
        )

        if seconds > 0:
            grouped[turn["speaker"]].append(turn)

    candidates = []

    for speaker, speaker_turns in grouped.items():
        seconds = covered_duration(
            speaker_turns,
            start,
            end
        )

        candidates.append({
            "speaker": speaker,
            "overlap_seconds": round(seconds, 3),
            "coverage": round(
                min(1.0, seconds / duration),
                3
            ),
        })

    candidates.sort(
        key=lambda item: item["overlap_seconds"],
        reverse=True
    )

    overlapping = has_overlapping_speech(
        turns,
        start,
        end
    )

    speaker = None

    if not candidates:
        status = "no_speech"

    else:
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        second_coverage = (
            second["coverage"] if second else 0.0
        )

        # Both speakers have meaningful evidence.
        if (
            second
            and second["overlap_seconds"] >= 0.03
            and second_coverage >= 0.25
        ):
            status = "overlap" if overlapping else "ambiguous"

        # Very short words are difficult to assign reliably.
        elif duration < 0.08:
            status = "uncertain"

        # Assign a speaker only with sufficient coverage.
        elif (
            best["coverage"] >= 0.60
            and best["coverage"] - second_coverage >= 0.35
        ):
            status = "single"
            speaker = best["speaker"]

        else:
            status = "uncertain"

    return {
        "speaker": speaker,
        "status": status,
        "overlapping_speech": overlapping,
        "speaker_candidates": candidates,
    }


def save_json(path, data):
    """Save JSON atomically."""

    temporary = path.with_name(path.name + ".tmp")

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    temporary.replace(path)


def align_words(video_id):
    """Combine word timestamps with speaker diarization."""

    video_dir = INPUT_DIR / video_id

    words_path = video_dir / "word_timestamps.json"
    diarization_path = video_dir / "diarization.json"
    output_path = video_dir / "word_speakers.json"

    words_bytes = words_path.read_bytes()
    diarization_bytes = diarization_path.read_bytes()

    words_data = json.loads(words_bytes)
    diarization = json.loads(diarization_bytes)

    # Ensure both analyses cover the same clip.
    for key in ("start", "end"):
        difference = abs(
            float(words_data["clip"][key])
            - float(diarization["clip"][key])
        )

        if difference > 0.1:
            raise ValueError(
                "Word timestamps and diarization "
                "use different clip ranges."
            )

    cache_key = hashlib.sha256(
        str(FORMAT_VERSION).encode("utf-8")
        + words_bytes
        + diarization_bytes
    ).hexdigest()

    if output_path.exists():
        cached = json.loads(
            output_path.read_text(encoding="utf-8")
        )

        if cached.get("cache_key") == cache_key:
            print("Using cached word-speaker alignment.")
            return output_path

    turns = diarization["turns"]
    words = words_data["words"]

    if not turns or not words:
        raise ValueError("Speaker turns or words are empty.")

    print("\nAligning words with speakers...")

    results = []

    for word in words:
        assignment = assign_speaker(word, turns)

        results.append({
            **word,
            **assignment,
        })

    output = {
        "video_id": video_id,
        "cache_key": cache_key,
        "clip": words_data["clip"],
        "speakers": diarization["speakers"],
        "words": results,
    }

    save_json(output_path, output)

    print("\nWord-speaker alignment completed!")
    print(f"Words: {len(results)}")

    for status in (
        "single",
        "overlap",
        "ambiguous",
        "uncertain",
        "no_speech",
    ):
        count = sum(
            word["status"] == status
            for word in results
        )

        print(f"{status}: {count}")

    print(f"\nSaved to: {output_path}")

    print("\nFirst 25 words:")
    print("-" * 55)

    for word in results[:25]:
        label = word["speaker"] or word["status"]

        print(
            f"{word['relative_start']:6.2f} - "
            f"{word['relative_end']:6.2f} "
            f"{label:12} "
            f"{word['word']}"
        )

    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python word_speaker_alignment.py VIDEO_ID"
        )
        sys.exit(1)

    try:
        align_words(sys.argv[1])

    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)
