import hashlib
import json
import os
import sys
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"

MODEL = "gpt-5.6-luna"


class Topic(BaseModel):
    title: str
    start: float
    end: float


class TopicList(BaseModel):
    topics: list[Topic]


def generate_topics(video_id):
    video_dir = INPUT_DIR / video_id

    transcript_path = video_dir / "transcript.json"
    metadata_path = video_dir / "source.info.json"
    topics_path = video_dir / "topics.json"

    transcript_bytes = transcript_path.read_bytes()
    transcript_hash = hashlib.sha256(transcript_bytes).hexdigest()

    # Reuse previously generated topics.
    if topics_path.exists():
        cached = json.loads(
            topics_path.read_text(encoding="utf-8")
        )

        if (
            cached.get("model") == MODEL
            and cached.get("transcript_hash") == transcript_hash
        ):
            print("Using cached AI topics.")
            return cached["topics"]

    transcript = json.loads(transcript_bytes)
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    duration = float(metadata["duration"])

    # Convert transcript segments into timestamped text.
    lines = []

    for segment in transcript["segments"]:
        start = float(segment["start"])
        end = float(segment["end"])
        text = segment["text"].strip()

        if text:
            lines.append(
                f"[{start:.2f} - {end:.2f}] {text}"
            )

    if not lines:
        raise ValueError("Transcript is empty.")

    transcript_text = "\n".join(lines)

    print(f"Generating AI topics with {MODEL}...")

    client = OpenAI()

    response = client.responses.parse(
        model=MODEL,
        reasoning={"effort": "none"},
        store=False,
        max_output_tokens=4000,
        input=[
            {
                "role": "system",
                "content": (
                    "Create a timestamped table of contents "
                    "for an English video transcript. "
                    "Identify meaningful changes of subject. "
                    "Use concise, specific English titles. "
                    "Aim for 10-18 topics for a long interview, "
                    "but prioritize actual topic boundaries. "
                    "Use timestamps in seconds. "
                    "Do not invent content. "
                    "Keep topics in chronological order "
                    "without overlapping ranges."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Video: {metadata.get('title', 'Unknown')}\n"
                    f"Duration: {duration} seconds\n\n"
                    f"Transcript:\n{transcript_text}"
                ),
            },
        ],
        text_format=TopicList,
    )

    result = response.output_parsed

    if result is None:
        raise RuntimeError("No structured topics returned.")

    topics = []

    for topic in result.topics:
        if (
            topic.title.strip()
            and 0 <= topic.start < topic.end <= duration
        ):
            topics.append(topic.model_dump())

    topics.sort(key=lambda item: item["start"])

    if not topics:
        raise RuntimeError("No valid topics generated.")

    data = {
        "model": MODEL,
        "transcript_hash": transcript_hash,
        "topics": topics,
    }

    temp_path = video_dir / "topics.json.tmp"

    temp_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temp_path.replace(topics_path)

    print(f"Generated {len(topics)} topics.")
    print(f"Saved to: {topics_path}")

    return topics


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ai_topics.py VIDEO_ID")
        sys.exit(1)

    try:
        topics = generate_topics(sys.argv[1])

        for index, topic in enumerate(topics, 1):
            print(
                f"[{index}] "
                f"{topic['start']:.1f} - "
                f"{topic['end']:.1f} "
                f"{topic['title']}"
            )

    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)
