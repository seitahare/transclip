import re


def normalize_word(word):
    """Normalize a word for overlap comparison."""
    return re.sub(r"[^\w]+", "", word).casefold()


def clean_rolling_captions(segments):
    """Merge repeated and progressively updated captions."""

    cleaned = []
    history = []
    last_new_end = float("-inf")

    for segment in segments:
        words = segment["text"].split()

        if not words:
            continue

        keys = [normalize_word(w) for w in words]
        recent = history[-100:]

        # Skip captions already seen in a recent update.
        duplicated = (
            len(keys) >= 3
            and segment["start"] - last_new_end < 2.5
            and any(
                recent[i:i + len(keys)] == keys
                for i in range(len(recent) - len(keys) + 1)
            )
        )

        if duplicated:
            continue

        # Find overlapping words between consecutive captions.
        overlap = 0

        for size in range(
            min(len(keys), len(history), 100), 1, -1
        ):
            if history[-size:] == keys[:size]:
                overlap = size
                break

        new_words = words[overlap:]

        if not new_words:
            continue

        new_text = " ".join(new_words)

        # Combine nearby captions into readable segments.
        can_merge = (
            cleaned
            and segment["start"] - cleaned[-1]["end"] < 1.5
            and len(cleaned[-1]["text"].split())
                + len(new_words) <= 35
            and not cleaned[-1]["text"].endswith(
                (".", "!", "?")
            )
        )

        if can_merge:
            cleaned[-1]["text"] += " " + new_text
            cleaned[-1]["end"] = max(
                cleaned[-1]["end"],
                segment["end"]
            )
        else:
            cleaned.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": new_text
            })

        history.extend(keys[overlap:])
        last_new_end = segment["end"]

    return cleaned
