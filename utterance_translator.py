import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"

MODEL = "gpt-5.6-luna"
BATCH_SIZE = 8
CONTEXT_SIZE = 2
PROMPT_VERSION = 2


class TranslationItem(BaseModel):
    id: int
    text_ja: str


class TranslationBatch(BaseModel):
    translations: list[TranslationItem]


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


def simplify_utterance(item):
    """Prepare one utterance for the translation model."""

    return {
        "id": item["id"],
        "speaker": item["speaker"],
        "text_en": item["text_en"],
    }


def build_request(utterances, indexes):
    """Build a request with surrounding conversation context."""

    first = min(indexes)
    last = max(indexes)

    context_start = max(0, first - CONTEXT_SIZE)
    context_end = min(
        len(utterances),
        last + CONTEXT_SIZE + 1,
    )

    return {
        "context_before": [
            simplify_utterance(item)
            for item in utterances[context_start:first]
        ],
        "translate": [
            simplify_utterance(utterances[index])
            for index in indexes
        ],
        "context_after": [
            simplify_utterance(item)
            for item in utterances[last + 1:context_end]
        ],
    }


SYSTEM_PROMPT = """
Translate English interview dialogue into natural spoken Japanese
for video dubbing.

The input JSON contains:
- context_before: previous dialogue for reference
- translate: utterances that MUST be translated
- context_after: following dialogue for reference

STRICT OUTPUT RULES:
1. Return exactly one translation for every item in "translate".
2. Copy each input integer ID exactly.
3. Do not return IDs from context_before or context_after.
4. Do not omit, duplicate, or invent IDs.
5. Every text_ja must contain non-empty Japanese text.
6. Translate even short fragments, fillers, and incomplete sentences.
7. Do not combine two IDs into one translation.

TRANSLATION QUALITY:
- Use natural conversational Japanese.
- Preserve the original meaning and tone.
- Use surrounding context to understand sentence fragments.
- Preserve technical terminology and proper names.
- Do not invent missing dialogue.
- Speaker labels are estimates, not verified identities.
- UNKNOWN does not identify a person.
- Do not add explanations, notes, or commentary.

Return only the required structured translation data.
"""


def request_translations(client, utterances, indexes):
    """Request translations and report malformed results."""

    request_data = build_request(
        utterances,
        indexes,
    )

    expected_ids = {
        utterances[index]["id"]
        for index in indexes
    }

    response = client.responses.parse(
        model=MODEL,
        reasoning={"effort": "none"},
        store=False,
        max_output_tokens=5000,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    request_data,
                    ensure_ascii=False,
                ),
            },
        ],
        text_format=TranslationBatch,
    )

    result = response.output_parsed

    if result is None:
        print("Warning: No parsed translations returned.")
        return {}

    returned_ids = [
        item.id
        for item in result.translations
    ]

    counts = Counter(returned_ids)

    duplicates = sorted(
        item_id
        for item_id, count in counts.items()
        if count > 1
    )

    extra_ids = sorted(
        set(returned_ids) - expected_ids
    )

    empty_ids = sorted(
        item.id
        for item in result.translations
        if not item.text_ja.strip()
    )

    valid = {}

    for item in result.translations:
        if item.id not in expected_ids:
            continue

        if counts[item.id] != 1:
            continue

        if not item.text_ja.strip():
            continue

        valid[item.id] = item.text_ja.strip()

    missing_ids = sorted(
        expected_ids - set(valid)
    )

    if (
        missing_ids
        or extra_ids
        or duplicates
        or empty_ids
    ):
        print("\nTranslation response validation:")
        print("Expected IDs:", sorted(expected_ids))
        print("Returned IDs:", returned_ids)
        print("Missing/invalid IDs:", missing_ids)
        print("Extra IDs:", extra_ids)
        print("Duplicate IDs:", duplicates)
        print("Empty translation IDs:", empty_ids)

    return valid


def translate_utterances(video_id, preview=False, limit=None):
    """Translate speaker utterances into Japanese."""

    video_dir = INPUT_DIR / video_id

    source_path = video_dir / "utterances.json"
    output_path = video_dir / "utterance_translations.json"
    progress_path = (
        video_dir / "utterance_translation.progress.json"
    )

    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)

    utterances = source["utterances"]

    if not utterances:
        raise ValueError("No utterances found.")

    ids = [item["id"] for item in utterances]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate utterance IDs.")

    total = len(utterances)

    if limit is not None and limit <= 0:
        raise ValueError("Limit must be positive.")

    target_total = (
        total
        if limit is None
        else min(limit, total)
    )

    print("\nSelected clip")
    print("-" * 45)
    print(f"Video: {video_id}")
    print(f"Total utterances: {total}")
    print(f"Target utterances: {target_total}")
    print(f"Model: {MODEL}")

    if preview:
        print("\nFirst utterances:")

        for item in utterances[:target_total][:8]:
            speaker = item["speaker"] or "UNKNOWN"

            print(
                f"\n[{item['relative_start']:.2f} - "
                f"{item['relative_end']:.2f}] "
                f"{speaker}"
            )

            print(item["text_en"])

        return None

    cache_info = {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "source_hash": hashlib.sha256(
            source_bytes
        ).hexdigest(),
    }

    cache_key = hashlib.sha256(
        json.dumps(
            cache_info,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    # Reuse completed translations.
    if output_path.exists():
        cached = json.loads(
            output_path.read_text(encoding="utf-8")
        )

        if cached.get("cache_key") == cache_key:
            print("\nUsing cached translation.")
            return output_path

    # Recover successfully translated items.
    translated = {}

    if progress_path.exists():
        progress = json.loads(
            progress_path.read_text(encoding="utf-8")
        )

        if progress.get("cache_key") == cache_key:
            translated = progress.get(
                "translations",
                {},
            )

    client = OpenAI()

    def save_progress():
        save_json(
            progress_path,
            {
                "cache_key": cache_key,
                "translations": translated,
            },
        )

    # Translate in batches.
    for start in range(0, target_total, BATCH_SIZE):
        end = min(
            start + BATCH_SIZE,
            target_total,
        )

        indexes = [
            index
            for index in range(start, end)
            if str(utterances[index]["id"]) not in translated
        ]

        if not indexes:
            continue

        print(
            f"\nTranslating utterances "
            f"{start + 1}-{end}/{total}..."
        )

        # First attempt: translate the whole pending batch.
        received = request_translations(
            client,
            utterances,
            indexes,
        )

        for item_id, text_ja in received.items():
            translated[str(item_id)] = text_ja

        if received:
            save_progress()

        # Retry only IDs that were not translated correctly.
        for index in indexes:
            item_id = utterances[index]["id"]

            if str(item_id) in translated:
                continue

            print(
                f"\nRetrying utterance ID {item_id} "
                f"individually..."
            )

            success = False

            for attempt in range(1, 3):
                received = request_translations(
                    client,
                    utterances,
                    [index],
                )

                if item_id in received:
                    translated[str(item_id)] = received[item_id]
                    save_progress()

                    success = True
                    break

                print(
                    f"Retry {attempt}/2 failed "
                    f"for ID {item_id}."
                )

            if not success:
                raise RuntimeError(
                    f"Could not translate utterance ID {item_id}. "
                    "Successful translations were saved."
                )

    print("\nTranslation progress")
    print("-" * 45)
    print(f"Translated: {len(translated)}/{total}")

    print("\nFirst translated utterances:")

    for item in utterances[:target_total][:8]:
        text_ja = translated.get(str(item["id"]))

        if text_ja is None:
            continue

        speaker = item["speaker"] or "UNKNOWN"

        print(
            f"\n[{item['relative_start']:.2f} - "
            f"{item['relative_end']:.2f}] "
            f"{speaker}"
        )

        print("EN:", item["text_en"])
        print("JA:", text_ja)

    # Limited runs save progress only.
    if len(translated) < total:
        print(
            "\nPartial translation saved to: "
            f"{progress_path}"
        )

        return progress_path

    # Build complete output.
    results = []

    for item in utterances:
        results.append({
            "id": item["id"],
            "speaker": item["speaker"],
            "start": item["start"],
            "end": item["end"],
            "relative_start": item["relative_start"],
            "relative_end": item["relative_end"],
            "text_en": item["text_en"],
            "text_ja": translated[str(item["id"])],
            "review_required": item["review_required"],
            "review_reasons": item["review_reasons"],
            "end_reason": item["end_reason"],
        })

    output = {
        "video_id": video_id,
        "model": MODEL,
        "cache_key": cache_key,
        "clip": source["clip"],
        "speakers": source["speakers"],
        "utterances": results,
    }

    save_json(output_path, output)

    print("\nTranslation completed!")
    print(f"Saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Translate TransClip speaker utterances"
    )

    parser.add_argument("video_id")

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview without calling the API",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Translate only the first N utterances",
    )

    args = parser.parse_args()

    try:
        translate_utterances(
            args.video_id,
            preview=args.preview,
            limit=args.limit,
        )

    except Exception as error:
        print(f"Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
