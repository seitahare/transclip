import json
from datetime import timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def parse_time(value):
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
    video_dir = INPUT_DIR / video_id
    metadata_path = video_dir / "source.info.json"

    if not metadata_path.exists():
        print("Error: Video metadata not found.")
        return 1

    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    title = metadata.get("title", "Unknown")
    duration = metadata.get("duration")
    chapters = metadata.get("chapters") or []

    if duration is None:
        print("Error: Video duration is unknown.")
        return 1

    while True:
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

        if choice == "1":
            if not chapters:
                print("\nNo chapters available.")
                continue

            print("\nVideo Chapters")
            print("-" * 40)

            for i, chapter in enumerate(chapters, 1):
                start = format_time(chapter["start_time"])
                end = format_time(chapter["end_time"])
                name = chapter["title"]

                print(f"[{i}] {start} - {end}  {name}")

            try:
                number = int(input("\nSelect a chapter: "))
                chapter = chapters[number - 1]

                if number < 1 or number > len(chapters):
                    raise ValueError

            except (ValueError, IndexError):
                print("Invalid chapter number.")
                continue

            start = chapter["start_time"]
            end = chapter["end_time"]

        elif choice == "2":
            print("\nAI-generated topics are not available yet.")
            continue

        elif choice == "3":
            try:
                start = parse_time(input("\nStart time: "))
                end = parse_time(input("End time: "))

            except ValueError:
                print("Invalid time format.")
                continue

        elif choice.lower() == "q":
            return 0

        else:
            print("Invalid option.")
            continue

        if not 0 <= start < end <= duration:
            print("Invalid time range.")
            continue

        print("\nSelected Clip")
        print("-" * 40)
        print(f"Start: {format_time(start)}")
        print(f"End: {format_time(end)}")
        print(f"Duration: {format_time(end - start)}")

        confirm = input("\nConfirm selection? (y/n): ")

        if confirm.lower() != "y":
            continue

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

        return 0
