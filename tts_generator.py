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


class QwenPresetSynthesizer:
    """Offline, lazily loaded Qwen adapter implementing SpeechSynthesizer.

    All generation settings and the cached model revision enter the audio key.
    No package installation, download, voice cloning or speed control occurs.
    """
    supported_voice_modes = {"preset"}

    def __init__(self, model_id=DEFAULT_MODEL, *, seed=0, expected_model_type="custom_voice"):
        import os
        from importlib.metadata import version
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from huggingface_hub import snapshot_download
        from language.cache import fingerprint
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer.")
        # A missing cache fails here, before any attempt to load the model.
        self.snapshot = Path(snapshot_download(
            model_id, local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"]))
        required = [self.snapshot / "config.json", self.snapshot / "speech_tokenizer/config.json"]
        if not all(p.is_file() for p in required) or not list(self.snapshot.glob("*.safetensors")) or not list((self.snapshot / "speech_tokenizer").glob("*.safetensors")):
            raise ValueError("Incomplete cached model. Download requires separate approval.")
        config = json.loads(required[0].read_text())
        if config.get("tts_model_type") != expected_model_type:
            raise ValueError(f"Expected {expected_model_type} model, got {config.get('tts_model_type')!r}.")
        files = []
        for p in sorted(self.snapshot.rglob("*")):
            if p.is_file():
                # HF snapshots reference immutable content-addressed blobs.
                files.append({"name": str(p.relative_to(self.snapshot)), "blob": p.resolve().name,
                              "size": p.stat().st_size,
                              "small_file_sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size < 10_000_000 else None})
        self.settings = {"language": "Japanese", "seed": seed, "temperature": .9,
                         "max_tokens": 4096, "top_k": 50, "top_p": 1.0,
                         "repetition_penalty": 1.05, "instruct": None,
                         "stream": False, "verbose": False, "wav_subtype": "PCM_16"}
        self.identity = {"backend": "qwen3-tts-mlx", "adapter_version": 1, "model": model_id,
                         "model_revision": self.snapshot.name, "model_files_fingerprint": fingerprint(files),
                         "versions": {name: version(name) for name in
                                      ("qwen3-tts-mlx", "mlx-audio", "mlx", "numpy", "soundfile", "transformers")},
                         "settings": dict(self.settings)}
        # Editable backend code can change without a package-version bump.
        from importlib.util import find_spec
        package = Path(find_spec("qwen_tts_mlx").origin).parent
        self.identity["backend_source_fingerprint"] = fingerprint({
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob("*.py")})
        self._model = None

    def synthesize(self, *, text, profile=None, output_path=None, voice=None):
        import numpy as np
        import mlx.core as mx
        import soundfile as sf
        from qwen_tts_mlx import MLXQwen3TTS
        from language.voice import VoiceProfile

        if profile is None and voice is not None:
            profile = VoiceProfile(speaker="PRESET_SPEAKER", mode="preset", voice=str(voice))
        elif isinstance(profile, str) and voice is None:
            profile = VoiceProfile(speaker="PRESET_SPEAKER", mode="preset", voice=profile)
        if profile is None or getattr(profile, "mode", None) != "preset" or getattr(profile, "reference_audio", None) or not getattr(profile, "voice", None):
            raise ValueError("Only explicit preset voices are allowed in this adapter.")
        if self._model is None:
            print("Loading cached Qwen3-TTS model (offline)...", flush=True)
            self._model = MLXQwen3TTS.from_pretrained(str(self.snapshot))
        mx.random.seed(self.settings["seed"])
        options = {k: v for k, v in self.settings.items() if k not in ("seed", "wav_subtype")}
        # The facade discards token counts. Retain native results to detect the
        # generation limit, rather than reporting truncated speech as complete.
        results = list(self._model.model.generate_custom_voice(text=text, speaker=profile.voice, **options))
        if not results:
            raise RuntimeError("Qwen returned no audio.")
        rates = {r.sample_rate for r in results}
        if len(rates) != 1 or any(r.token_count >= self.settings["max_tokens"] for r in results):
            raise RuntimeError("Inconsistent sample rates or token limit reached; audio not published.")
        wav = np.concatenate([np.asarray(r.audio) for r in results])
        if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
            raise ValueError("Qwen returned empty, non-mono or non-finite samples.")
        rate = int(next(iter(rates)))
        if rate <= 0:
            raise ValueError("Qwen returned invalid sample rate.")
        sf.write(output_path, wav, rate, subtype="PCM_16", format="WAV")
        return {"sample_rate": rate, "sample_frames": int(wav.size),
                "token_count": sum(r.token_count for r in results),
                "termination": "eos_before_limit", "preset": profile.voice,
                "preset_voice": profile.voice,
                "processing_seconds": sum(getattr(r, "processing_time_seconds", 0.0) for r in results)}


def main(argv=None):
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

    parser.add_argument("--script", type=Path, help="Use dubbing_script.json (Phase 2)")
    parser.add_argument("--speaker-profiles", type=Path, help="Attach Phase 4 voice analysis to preset synthesis")
    parser.add_argument("--feedback-dir", type=Path, help="Separate audio cache and feedback directory")
    parser.add_argument("--baseline-manifest", type=Path, help="Read-only comparison with legacy WAVs")
    parser.add_argument("--voice-map", type=Path, help="JSON mapping known speakers to preset voices")
    parser.add_argument("--entry-id", action="append", help="Limit generation to explicit script entry IDs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        if args.seconds <= 0:
            raise ValueError(
                "seconds must be positive."
            )

        if args.speaker_profiles and not args.script:
            raise ValueError("--speaker-profiles requires --script.")

        if args.script:
            from tts_feedback import plan_script, run_feedback
            voices = json.loads(args.voice_map.read_text()) if args.voice_map else SPEAKER_VOICES
            script = json.loads(args.script.read_text())
            if script.get("video_id") != args.video_id:
                raise ValueError("Requested video_id does not match the script.")
            if Path(args.video_id).name != args.video_id or args.video_id in {".", ".."}:
                raise ValueError("video_id must be a directory name.")
            selection_path = args.script.parent / "selection.json"
            if selection_path.exists():
                selection = json.loads(selection_path.read_text())
                if any(abs(selection[k] - script["clip"][k]) > .01 for k in ("start", "end")):
                    raise ValueError("Script does not match current selection.json.")
            if args.max_jobs is not None and args.max_jobs <= 0:
                raise ValueError("max_jobs must be positive.")
            plan = plan_script(script, voices, seconds=args.seconds, entry_ids=args.entry_id)
            profile_document = None
            if args.speaker_profiles:
                from language.voice import ProfiledPresetSynthesizer, profile_for_synthesis
                profile_document = json.loads(args.speaker_profiles.read_text())
                # Validate before the dry-run return or loading any Qwen assets.
                if profile_document.get("video_id") != args.video_id or profile_document.get("clip") != script["clip"]:
                    raise ValueError("Voice profiles do not match synthesis video/clip.")
                for row in plan:
                    if row["generation_status"] != "skipped":
                        profile_for_synthesis(profile_document, row["speaker"], preset=row["voice"])

            if args.dry_run:
                for row in plan:
                    print(row["id"], row["speaker"] or "UNKNOWN", row["voice"],
                          row["generation_status"], row["skip_reason"] or "")
                print("Dry run: no model loaded, downloaded, or synthesized.")
                return 0
            backend = QwenPresetSynthesizer(args.model, seed=args.seed)
            if profile_document is not None:
                backend = ProfiledPresetSynthesizer(backend, profile_document, video_id=args.video_id, clip=script["clip"])
            run_feedback(args.script, args.feedback_dir or OUTPUT_DIR / args.video_id / "tts_feedback",
                         backend, voices, seconds=args.seconds, max_jobs=args.max_jobs,
                         entry_ids=args.entry_id, baseline_manifest=args.baseline_manifest)
            return 0
        if args.feedback_dir or args.baseline_manifest or args.voice_map or args.entry_id:
            raise ValueError("Feedback options require --script.")
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
