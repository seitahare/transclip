import json
from datetime import timedelta
from pathlib import Path

from ai_topics import generate_topics


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"


def format_time(seconds):
    """Convert seconds to a readable time format."""
    return str(timedelta(seconds=int(seconds)))


def parse_time(value):
    """Convert a time string into seconds."""
    parts = value.strip().split(":")

    if not 1 <= len(parts) <= 3:
        raise ValueError("Invalid time format")

    if not all(part.isdigit() for part in parts):
        raise ValueError("Invalid time format")

    numbers = [int(part) for part in parts]

    if len(numbers) >= 2 and numbers[-1] >= 60:
        raise ValueError("Seconds must be less than 60")

    if len(numbers) == 3 and numbers[-2] >= 60:
        raise ValueError("Minutes must be less than 60")

    seconds = 0

    for number in numbers:
        seconds = seconds * 60 + number

    return seconds


def select_clip(video_id):
    """Interactively select a clip from a video."""

    video_dir = INPUT_DIR / video_id
    metadata_path = video_dir / "source.info.json"

    if not metadata_path.exists():
        raise FileNotFoundError("Video metadata not found.")

    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    title = metadata.get("title", "Unknown")
    duration = metadata.get("duration")
    chapters = metadata.get("chapters") or []

    if duration is None:
        raise ValueError("video duration is unknown.")

    while True:

        # Display the main menu.
        print("\n" + "=" * 40)
        print("               TransClip")
        print("=" * 40)

        print(f"\nVideo: {title}")
        print(f"Duration: {format_time(duration)}")

        print("\nHow would you like to select a clip?\n")

        print("[1] Choose from video chapters")
        print("[2] Choose from AI-generated topics")
        print("[3] Specify a custom time range")
        print("[q] Quit")

        choice = input("\nSelect an option: ").strip()

        # ----------------------------------------
        # Option 1: YouTube chapters
        # ----------------------------------------

        if choice == "1":

            if not chapters:
                print("\nNo chapters available.")
                continue

            print("\nVideo Chapters")
            print("-" * 40)

            for i, chapter in enumerate(chapters, 1):
                start_time = format_time(chapter["start_time"])
                end_time = format_time(chapter["end_time"])
                chapter_title = chapter["title"]

                print(
                    f"[{i}] {start_time} - {end_time}  "
                    f"{chapter_title}"
                )

            try:
                number = int(input("\nSelect a chapter: "))

                if not 1 <= number <= len(chapters):
                    raise ValueError

            except ValueError:
                print("Invalid chapter number.")
                continue

            chapter = chapters[number - 1]

            start = chapter["start_time"]
            end = chapter["end_time"]

        # ----------------------------------------
        # Option 2: AI-generated topics
        # ----------------------------------------

        elif choice == "2":

            try:
                topics = generate_topics(video_id)

            except Exception as error:
                print(
                    f"\nError: Could not generate AI topics: {error}"
                )
                continue

            if not topics:
                print("\nNo AI-generated topics available.")
                continue

            print("\nAI-Generated Topics")
            print("-" * 40)

            for i, topic in enumerate(topics, 1):
                start_time = format_time(topic["start"])
                end_time = format_time(topic["end"])
                topic_title = topic["title"]

                print(
                    f"[{i}] {start_time} - {end_time}"
                )
                print(f"    {topic_title}")

            try:
                number = int(input("\nSelect a topic: "))

                if not 1 <= number <= len(topics):
                    raise ValueError

            except ValueError:
                print("Invalid topic number.")
                continue

            topic = topics[number - 1]

            start = topic["start"]
            end = topic["end"]

        # ----------------------------------------
        # Option 3: Custom time range
        # ----------------------------------------

        elif choice == "3":

            try:
                start = parse_time(
                    input("\nStart time: ")
                )

                end = parse_time(
                    input("End time: ")
                )

            except ValueError:
                print("Invalid time format.")
                continue

        # ----------------------------------------
        # Quit
        # ----------------------------------------

        elif choice.lower() == "q":
            print("\nExiting TransClip.")
            return None

        else:
            print("Invalid option.")
            continue

        # ----------------------------------------
        # Validate the selected time range
        # ----------------------------------------

        if not 0 <= start < end <= duration:
            print("Invalid time range.")
            continue

        # ----------------------------------------
        # Confirm the selection
        # ----------------------------------------

        print("\nSelected Clip")
        print("-" * 40)

        print(f"Start: {format_time(start)}")
        print(f"End: {format_time(end)}")
        print(f"Duration: {format_time(end - start)}")

        confirm = input("\nConfirm selection? (y/n): ").strip()

        if confirm.lower() != "y":
            continue

        # ----------------------------------------
        # Save the selection
        # ----------------------------------------

        selection = {
            "start": start,
            "end": end,
        }

        selection_path = video_dir / "selection.json"

        selection_path.write_text(
            json.dumps(selection, indent=2),
            encoding="utf-8"
        )

        print("\nSelection saved!")
        print(f"Path: {selection_path}")

        return selection_path
