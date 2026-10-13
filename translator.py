import argparse
import hashlib
import json
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"

MODEL = "gpt-5.6-luna"
BATCH_SIZE = 12
PROMPT_VERSION = 1


class TranslationItem(BaseModel):
    id: int
    text_ja: str


class TranslationBatch(BaseModel):
    translations: list[TranslationItem]


def save_json(path, data):
    """Write JSON safely."""
    temporary = path.with_name(path.name + ".tmp")

    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary.replace(path)


def prepare_segments(video_dir):
    """Extract only the selected portion of the transcript."""

    selection_path = video_dir / "selection.json"
    transcript_path = video_dir / "transcript.json"

    with selection_path.open(encoding="utf-8") as f:
        selection = json.load(f)

    with transcript_path.open(encoding="utf-8") as f:
        transcript = json.load(f)

    clip_start = float(selection["start"])
    clip_end = float(selection["end"])

    if not 0 <= clip_start < clip_end:
        raise ValueError("Invalid clip time range.")

    segments = []

    for segment in transcript["segments"]:
        start = float(segment["start"])
        end = float(segment["end"])
        text = segment["text"].strip()

        # Ignore subtitles outside the selected range.
        if end <= clip_start or start >= clip_end:
            continue

        if not text:
            continue

        # Restrict timestamps to the selected clip.
        start = max(start, clip_start)
        end = min(end, clip_end)

        if start >= end:
            continue

        segments.append({
            "id": len(segments) + 1,
            "start": round(start, 2),
            "end": round(end, 2),
            "relative_start": round(start - clip_start, 2),
            "relative_end": round(end - clip_start, 2),
            "text_en": text,
        })

    if not segments:
        raise ValueError("No subtitles found in selected range.")

    return clip_start, clip_end, segments


def translate_clip(video_id, preview=False):
    """Translate the selected clip into Japanese."""

    video_dir = INPUT_DIR / video_id

    clip_start, clip_end, segments = prepare_segments(video_dir)

    print("\nSelected Clip")
    print("-" * 40)
    print(f"Start: {clip_start:.2f}")
    print(f"End: {clip_end:.2f}")
    print(f"Duration: {clip_end - clip_start:.2f} seconds")
    print(f"Subtitle segments: {len(segments)}")

    # Preview without calling the API.
    if preview:
        print("\nFirst 5 English segments:")

        for segment in segments[:5]:
            print(
                f"[{segment['relative_start']:.2f} - "
                f"{segment['relative_end']:.2f}] "
                f"{segment['text_en']}"
            )

        return None

    # Cache depends on the model, prompt and selected text.
    cache_data = {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "clip_start": clip_start,
        "clip_end": clip_end,
        "segments": segments,
    }

    cache_key = hashlib.sha256(
        json.dumps(
            cache_data,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    output_path = video_dir / "translation.json"
    progress_path = video_dir / "translation.progress.json"

    # Reuse a completed translation.
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            cached = json.load(f)

        if cached.get("cache_key") == cache_key:
            print("\nUsing cached translation.")
            return output_path

    # Recover successfully translated batches.
    translated = {}

    if progress_path.exists():
        with progress_path.open(encoding="utf-8") as f:
            progress = json.load(f)

        if progress.get("cache_key") == cache_key:
            translated = progress.get("translations", {})

    client = OpenAI()

    total = len(segments)

    for offset in range(0, total, BATCH_SIZE):
        batch = segments[offset:offset + BATCH_SIZE]

        # Skip lines that were already translated.
        pending = [
            segment
            for segment in batch
            if str(segment["id"]) not in translated
        ]

        if not pending:
            continue

        print(
            f"\nTranslating segments "
            f"{offset + 1}-{min(offset + BATCH_SIZE, total)}"
            f"/{total}..."
        )

        request_lines = [
            {
                "id": segment["id"],
                "text_en": segment["text_en"],
            }
            for segment in pending
        ]

        response = client.responses.parse(
            model=MODEL,
            reasoning={"effort": "none"},
            store=False,
            max_output_tokens=4000,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Translate English interview subtitles "
                        "into natural spoken Japanese. "
                        "Preserve the meaning, tone, names, "
                        "technical terminology and uncertainty. "
                        "Keep every input ID exactly once. "
                        "Do not omit or invent dialogue. "
                        "Do not merge or split input items. "
                        "Keep speaker-change markers such as >> "
                        "when they appear in the source. "
                        "Use Japanese suitable for later dubbing."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        request_lines,
                        ensure_ascii=False,
                    ),
                },
            ],
            text_format=TranslationBatch,
        )

        result = response.output_parsed

        if result is None:
            raise RuntimeError("Translation response is empty.")

        expected_ids = {item["id"] for item in pending}
        returned_ids = [
            item.id for item in result.translations
        ]

        if (
            len(returned_ids) != len(expected_ids)
            or set(returned_ids) != expected_ids
            or any(
                not item.text_ja.strip()
                for item in result.translations
            )
        ):
            raise RuntimeError(
                "Translation IDs or texts are invalid."
            )

        for item in result.translations:
            translated[str(item.id)] = item.text_ja.strip()

        # Save progress after every successful batch.
        save_json(progress_path, {
            "cache_key": cache_key,
            "translations": translated,
        })

    # Combine English, Japanese and timestamps.
    translated_segments = []

    for segment in segments:
        translated_segments.append({
            **segment,
            "text_ja": translated[str(segment["id"])],
        })

    output = {
        "source": "openai",
        "model": MODEL,
        "cache_key": cache_key,
        "clip": {
            "start": clip_start,
            "end": clip_end,
        },
        "segments": translated_segments,
    }

    save_json(output_path, output)

    print("\nTranslation completed!")
    print(f"Translated segments: {len(translated_segments)}")
    print(f"Saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Translate a selected TransClip video segment"
    )

    parser.add_argument("video_id")

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview selected subtitles without using the API",
    )

    args = parser.parse_args()

    try:
        translate_clip(args.video_id, preview=args.preview)
    except Exception as error:
        print(f"Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
