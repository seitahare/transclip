import argparse
import hashlib
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"

FORMAT_VERSION = 3

PAUSE_THRESHOLD = 0.75
SOFT_SPLIT_DURATION = 8.0
MAX_UTTERANCE_DURATION = 14.0

# Conservative speaker inference.
MAX_NEIGHBOR_GAP = 0.45
MAX_UNKNOWN_SPAN = 1.5


def save_json(path, data):
    """Write JSON atomically."""

    temporary = path.with_name(path.name + ".tmp")

    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary.replace(path)


def overlap(a_start, a_end, b_start, b_end):
    """Return the overlap between two time intervals."""

    return max(
        0.0,
        min(a_end, b_end) - max(a_start, b_start),
    )


def exclusive_speaker_hint(word, turns):
    """Estimate a speaker from exclusive diarization."""

    start = float(word["relative_start"])
    end = float(word["relative_end"])
    duration = end - start

    if duration <= 0:
        return None

    coverage = {}

    for turn in turns:
        seconds = overlap(
            start,
            end,
            turn["relative_start"],
            turn["relative_end"],
        )

        if seconds <= 0:
            continue

        speaker = turn["speaker"]

        coverage[speaker] = (
            coverage.get(speaker, 0.0) + seconds
        )

    if not coverage:
        return None

    ranked = sorted(
        coverage.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_speaker, best_seconds = ranked[0]

    second_seconds = (
        ranked[1][1] if len(ranked) > 1 else 0.0
    )

    if (
        best_seconds / duration >= 0.45
        and (best_seconds - second_seconds) / duration >= 0.20
    ):
        return best_speaker

    return None


def prepare_words(original_words, exclusive_turns):
    """Preserve original assignments and add conservative hints."""

    words = []

    for index, original in enumerate(original_words):
        word = dict(original)

        word["word_id"] = index + 1
        word["original_status"] = original["status"]

        word["exclusive_hint"] = exclusive_speaker_hint(
            word,
            exclusive_turns,
        )

        word["assignment_method"] = (
            "direct"
            if word["status"] == "single"
            else "unresolved"
        )

        word["review_required"] = (
            word["status"] != "single"
            or bool(word.get("overlapping_speech"))
        )

        if (
            word["status"] != "uncertain"
            or word.get("overlapping_speech")
        ):
            words.append(word)
            continue

        candidates = word.get("speaker_candidates", [])

        if candidates:
            best = candidates[0]

            best_coverage = float(best["coverage"])

            second_coverage = (
                float(candidates[1]["coverage"])
                if len(candidates) > 1
                else 0.0
            )

            duration = (
                word["relative_end"]
                - word["relative_start"]
            )

            if (
                duration >= 0.08
                and best_coverage >= 0.45
                and second_coverage <= 0.15
                and best["speaker"] == word["exclusive_hint"]
            ):
                word["speaker"] = best["speaker"]
                word["assignment_method"] = "inferred_exclusive"

        words.append(word)

    return words


def find_anchor(words, index, direction):
    """
    Find a nearby word with a direct speaker assignment.

    Never search through overlapping or ambiguous speech.
    """

    position = index + direction
    previous = words[index]

    while 0 <= position < len(words):
        neighbor = words[position]

        if direction < 0:
            gap = (
                previous["relative_start"]
                - neighbor["relative_end"]
            )
        else:
            gap = (
                neighbor["relative_start"]
                - previous["relative_end"]
            )

        if gap > MAX_NEIGHBOR_GAP:
            return None

        if neighbor["original_status"] in (
            "overlap",
            "ambiguous",
        ):
            return None

        if neighbor.get("overlapping_speech"):
            return None

        if (
            neighbor["original_status"] == "single"
            and neighbor["speaker"] is not None
        ):
            return position, neighbor["speaker"]

        if neighbor["original_status"] != "uncertain":
            return None

        previous = neighbor
        position += direction

    return None


def bridge_short_unknowns(words):
    """
    Fill short uncertain gaps only when direct speakers
    on both sides agree.

    Never change an overlapping or ambiguous word.
    """

    for index, word in enumerate(words):
        if word["speaker"] is not None:
            continue

        if word["original_status"] != "uncertain":
            continue

        if word.get("overlapping_speech"):
            continue

        left = find_anchor(words, index, -1)
        right = find_anchor(words, index, 1)

        if left is None or right is None:
            continue

        left_index, left_speaker = left
        right_index, right_speaker = right

        if left_speaker != right_speaker:
            continue

        # Avoid filling long gaps across possible speaker changes.
        unknown_span = (
            words[right_index]["relative_start"]
            - words[left_index]["relative_end"]
        )

        if unknown_span > MAX_UNKNOWN_SPAN:
            continue

        # Never contradict exclusive diarization.
        hint = word.get("exclusive_hint")

        if hint is not None and hint != left_speaker:
            continue

        # Never contradict strong evidence from another speaker.
        candidates = word.get("speaker_candidates", [])

        if candidates:
            best = candidates[0]

            if (
                best["speaker"] != left_speaker
                and float(best["coverage"]) >= 0.45
            ):
                continue

        word["speaker"] = left_speaker
        word["assignment_method"] = "inferred_context"

        # An inferred assignment always needs review.
        word["review_required"] = True

    return words


def ends_sentence(text):
    """Detect a likely sentence ending."""

    text = text.strip().rstrip('"\u201d\u2019')

    abbreviations = {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "e.g.",
        "i.e.",
        "u.s.",
    }

    if text.lower() in abbreviations:
        return False

    return text.endswith((".", "?", "!"))


def should_break(current, next_word):
    """Return a reason to start another utterance."""

    if not current:
        return None

    first = current[0]
    previous = current[-1]

    if previous["speaker"] != next_word["speaker"]:
        return "speaker_change"

    if previous["speaker"] is None:
        left_hint = previous.get("exclusive_hint")
        right_hint = next_word.get("exclusive_hint")

        if (
            left_hint is not None
            and right_hint is not None
            and left_hint != right_hint
        ):
            return "speaker_hint_change"

    if ends_sentence(previous["word"]):
        return "sentence_end"

    gap = (
        next_word["relative_start"]
        - previous["relative_end"]
    )

    if gap >= PAUSE_THRESHOLD:
        return "pause"

    current_duration = (
        previous["relative_end"]
        - first["relative_start"]
    )

    if (
        current_duration >= SOFT_SPLIT_DURATION
        and previous["word"].endswith((",", ";", ":"))
    ):
        return "soft_length_limit"

    next_duration = (
        next_word["relative_end"]
        - first["relative_start"]
    )

    if next_duration > MAX_UTTERANCE_DURATION:
        return "max_duration"

    return None


def make_utterance(words, utterance_id, end_reason):
    """Build a draft utterance from consecutive words."""

    review_reasons = set()

    for word in words:
        if word["assignment_method"] != "direct":
            review_reasons.add("inferred_or_unresolved")

        if word["speaker"] is None:
            review_reasons.add("unknown_speaker")

        if word.get("overlapping_speech"):
            review_reasons.add("overlapping_speech")

    if len(words) == 1:
        review_reasons.add("single_word_utterance")

    if end_reason == "max_duration":
        review_reasons.add("forced_duration_split")

    start = min(word["relative_start"] for word in words)
    end = max(word["relative_end"] for word in words)

    clip_start = words[0]["start"] - words[0]["relative_start"]

    return {
        "id": utterance_id,
        "speaker": words[0]["speaker"],
        "start": round(clip_start + start, 3),
        "end": round(clip_start + end, 3),
        "relative_start": round(start, 3),
        "relative_end": round(end, 3),
        "text_en": " ".join(word["word"] for word in words),
        "word_count": len(words),
        "end_reason": end_reason,
        "review_required": bool(review_reasons),
        "review_reasons": sorted(review_reasons),
        "words": words,
    }


def build_utterances(video_id):
    """Group words into speaker-specific draft utterances."""

    video_dir = INPUT_DIR / video_id

    words_path = video_dir / "word_speakers.json"
    diarization_path = video_dir / "diarization.json"
    output_path = video_dir / "utterances.json"

    words_bytes = words_path.read_bytes()
    diarization_bytes = diarization_path.read_bytes()

    source = json.loads(words_bytes)
    diarization = json.loads(diarization_bytes)

    for key in ("start", "end"):
        difference = abs(
            float(source["clip"][key])
            - float(diarization["clip"][key])
        )

        if difference > 0.1:
            raise ValueError(
                "Word alignment and diarization "
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
            print("Using cached utterances.")
            return output_path

    original_words = source["words"]

    if not original_words:
        raise ValueError("No aligned words found.")

    for word in original_words:
        if word["relative_end"] <= word["relative_start"]:
            raise ValueError("Invalid word time range.")

    words = prepare_words(
        original_words,
        diarization["exclusive_turns"],
    )

    words = bridge_short_unknowns(words)

    print("\nBuilding speaker utterances...")

    groups = []
    current = []

    for word in words:
        reason = should_break(current, word)

        if reason:
            groups.append((current, reason))
            current = []

        current.append(word)

    if current:
        groups.append((current, "end_of_clip"))

    utterances = [
        make_utterance(group, index + 1, reason)
        for index, (group, reason) in enumerate(groups)
    ]

    recovered_count = sum(
        utterance["word_count"]
        for utterance in utterances
    )

    if recovered_count != len(original_words):
        raise RuntimeError("Some words were lost.")

    output = {
        "video_id": video_id,
        "format_version": FORMAT_VERSION,
        "cache_key": cache_key,
        "clip": source["clip"],
        "speakers": source["speakers"],
        "source_word_count": len(original_words),
        "utterances": utterances,
    }

    save_json(output_path, output)

    assigned = sum(
        word["speaker"] is not None
        for word in words
    )

    context_inferred = sum(
        word["assignment_method"] == "inferred_context"
        for word in words
    )

    review_count = sum(
        utterance["review_required"]
        for utterance in utterances
    )

    print("\nUtterance building completed!")
    print(f"Source words: {len(original_words)}")
    print(f"Words with speaker: {assigned}")
    print(f"Unresolved words: {len(words) - assigned}")
    print(f"Context-inferred words: {context_inferred}")
    print(f"Utterances: {len(utterances)}")
    print(f"Utterances needing review: {review_count}")
    print(f"Saved to: {output_path}")

    print("\nUtterances between 20 and 60 seconds:")
    print("-" * 60)

    for utterance in utterances:
        if (
            utterance["relative_end"] <= 20
            or utterance["relative_start"] >= 60
        ):
            continue

        speaker = utterance["speaker"] or "UNKNOWN"

        review = (
            " [REVIEW]"
            if utterance["review_required"]
            else ""
        )

        print(
            f"\n[{utterance['relative_start']:.2f} - "
            f"{utterance['relative_end']:.2f}] "
            f"{speaker}{review}"
        )

        print(utterance["text_en"])

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build speaker utterances for TransClip"
    )

    parser.add_argument("video_id")

    args = parser.parse_args()

    try:
        build_utterances(args.video_id)
    except Exception as error:
        print(f"Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
