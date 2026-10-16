import argparse
import hashlib
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

DEFAULT_MODEL = (
    "mlx-community/"
    "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
)

SPEAKER_VOICES = {
    "SPEAKER_00": "Ryan",
    "SPEAKER_01": "Aiden",
}

MAX_GROUP_DURATION = 13.0
MAX_MERGE_GAP = 0.5


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


def make_job(item, speaker):
    """Create one speech generation job."""

    return {
        "speaker": speaker,
        "voice": SPEAKER_VOICES[speaker],
        "start": float(item["relative_start"]),
        "end": float(item["relative_end"]),
        "text_ja": item["text_ja"].strip(),
        "utterance_ids": [item["id"]],
        "review_required": bool(
            item.get("review_required", False)
        ),
    }


def build_jobs(utterances, seconds):
    """
    Group adjacent utterances from the same speaker.

    Keep UNKNOWN utterances out of automatic generation.
    Do not merge across speaker changes.
    """

    jobs = []
    skipped = []
    current = None

    for item in utterances:
        start = float(item["relative_start"])
        end = float(item["relative_end"])

        if start >= seconds:
            break

        # Do not cut speech that crosses the preview boundary.
        if end > seconds:
            skipped.append({
                "id": item["id"],
                "reason": "crosses_preview_boundary",
                "text_ja": item["text_ja"],
            })
            continue

        speaker = item["speaker"]
        text = item["text_ja"].strip()

        if not text:
            skipped.append({
                "id": item["id"],
                "reason": "empty_text",
            })
            continue

        if speaker not in SPEAKER_VOICES:
            skipped.append({
                "id": item["id"],
                "reason": "unknown_speaker",
                "text_ja": text,
            })

            if current is not None:
                jobs.append(current)
                current = None

            continue

        next_job = make_job(item, speaker)

        if current is None:
            current = next_job
            continue

        gap = start - current["end"]

        combined_duration = (
            end - current["start"]
        )

        can_merge = (
            current["speaker"] == speaker
            and -0.15 <= gap <= MAX_MERGE_GAP
            and combined_duration <= MAX_GROUP_DURATION
        )

        if can_merge:
            current["text_ja"] += text
            current["end"] = max(current["end"], end)
            current["utterance_ids"].append(item["id"])

            current["review_required"] = (
                current["review_required"]
                or next_job["review_required"]
            )

        else:
            jobs.append(current)
            current = next_job

    if current is not None:
        jobs.append(current)

    return jobs, skipped


def generate_audio(video_id, seconds, max_jobs, dry_run, model_id):
    """Generate Japanese WAV files from translated utterances."""

    source_path = (
        INPUT_DIR
        / video_id
        / "utterance_translations.json"
    )

    if not source_path.exists():
        raise FileNotFoundError(
            "utterance_translations.json not found."
        )

    with source_path.open(encoding="utf-8") as f:
        data = json.load(f)

    jobs, skipped = build_jobs(
        data["utterances"],
        seconds,
    )

    if max_jobs is not None:
        if max_jobs <= 0:
            raise ValueError(
                "max_jobs must be positive."
            )

        jobs = jobs[:max_jobs]

    print("\n========================================")
    print("          TransClip TTS Preview")
    print("========================================")

    print(f"Video: {video_id}")
    print(f"Preview range: 0-{seconds:.2f} seconds")
    print(f"Speech jobs: {len(jobs)}")
    print(f"Skipped utterances: {len(skipped)}")
    print(f"Model: {model_id}")

    print("\nPlanned speech:")
    print("-" * 55)

    for index, job in enumerate(jobs, 1):
        print(
            f"\n[{index}] "
            f"{job['start']:.2f}-{job['end']:.2f}s "
            f"{job['speaker']} / {job['voice']}"
        )

        print(f"IDs: {job['utterance_ids']}")
        print(job["text_ja"])

        if job["review_required"]:
            print("[REVIEW]")

    if skipped:
        print("\nSkipped utterances:")
        print("-" * 55)

        for item in skipped:
            print(
                f"ID {item['id']}: {item['reason']}"
            )

    if dry_run:
        print("\nDry run completed. No audio generated.")
        return

    if not jobs:
        raise RuntimeError(
            "No valid speech jobs found."
        )

    # Import TTS dependencies only when generating audio.
    from qwen_tts_mlx import MLXQwen3TTS
    import soundfile as sf

    output_dir = (
        OUTPUT_DIR
        / video_id
        / "tts"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading Qwen3-TTS model...")

    tts = MLXQwen3TTS.from_pretrained(
        model_id
    )

    print("\nGenerating Japanese speech...")

    manifest = {
        "video_id": video_id,
        "model": model_id,
        "preview_seconds": seconds,
        "skipped": skipped,
        "jobs": [],
    }

    manifest_path = output_dir / "manifest.json"

    for index, job in enumerate(jobs, 1):
        cache_info = {
            "model": model_id,
            "voice": job["voice"],
            "text": job["text_ja"],
            "start": job["start"],
            "end": job["end"],
            "utterance_ids": job["utterance_ids"],
        }

        cache_key = hashlib.sha256(
            json.dumps(
                cache_info,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:12]

        filename = (
            f"speech_{index:03d}_"
            f"{job['speaker']}_"
            f"{cache_key}.wav"
        )

        wav_path = output_dir / filename

        print(
            f"\n[{index}/{len(jobs)}] "
            f"{job['speaker']} "
            f"({job['voice']})"
        )

        if wav_path.exists():
            info = sf.info(str(wav_path))

            if info.frames <= 0:
                raise RuntimeError(
                    f"Cached audio is empty: {wav_path}"
                )

            print("Using cached audio.")

        else:
            wavs, sample_rate = (
                tts.generate_custom_voice(
                    text=job["text_ja"],
                    language="Japanese",
                    speaker=job["voice"],
                )
            )

            if not wavs:
                raise RuntimeError(
                    "TTS returned no audio."
                )

            temporary_path = (
                wav_path.with_name(
                    wav_path.stem + ".tmp.wav"
                )
            )

            sf.write(
                str(temporary_path),
                wavs[0],
                sample_rate,
            )

            temporary_path.replace(wav_path)

            print(f"Generated: {wav_path.name}")

        info = sf.info(str(wav_path))

        manifest["jobs"].append({
            **job,
            "audio_file": filename,
            "audio_duration": (
                info.frames / info.samplerate
            ),
            "sample_rate": info.samplerate,
        })

        save_json(manifest_path, manifest)

        print(
            f"Audio duration: "
            f"{info.frames / info.samplerate:.2f}s"
        )

    print("\nTTS generation completed!")
    print(f"Audio directory: {output_dir}")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Japanese speech with Qwen3-TTS"
    )

    parser.add_argument("video_id")

    parser.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="Preview duration in seconds",
    )

    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Generate only the first N speech jobs",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview jobs without loading the TTS model",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Qwen3-TTS MLX CustomVoice model ID",
    )

    args = parser.parse_args()

    try:
        if args.seconds <= 0:
            raise ValueError(
                "seconds must be positive."
            )

        generate_audio(
            video_id=args.video_id,
            seconds=args.seconds,
            max_jobs=args.max_jobs,
            dry_run=args.dry_run,
            model_id=args.model,
        )

    except Exception as error:
        print(f"Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
