"""Transparent duration estimates and explicit, voice-scoped WAV calibration."""
from dataclasses import dataclass, asdict
import math
import wave


@dataclass(frozen=True)
class TimingConfig:
    mora_per_second: float = 7.0
    comma_pause: float = 0.12
    sentence_pause: float = 0.25
    ellipsis_pause: float = 0.3
    tolerance_seconds: float = 0.25

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid timing option: {key}")
        if self.mora_per_second <= 0:
            raise ValueError("mora_per_second must be positive.")


class TimingEngine:
    def __init__(self, config=None):
        self.config = config or TimingConfig()

    def estimate(self, analysis, available):
        if not math.isfinite(available) or available <= 0:
            raise ValueError("Available duration must be positive and finite.")
        c = self.config
        pauses = sum({"comma": c.comma_pause, "sentence": c.sentence_pause,
                      "ellipsis": c.ellipsis_pause}[p["kind"]] for p in analysis["pauses"])
        duration = analysis["mora_count"] / c.mora_per_second + pauses
        delta = duration - available
        return {"estimated_duration_seconds": round(duration, 3),
                "source_duration_seconds": round(available, 3),
                "duration_delta_seconds": round(delta, 3),
                "duration_ratio": round(duration / available, 3),
                "pause_seconds": round(pauses, 3),
                "timing_status": "overlong" if delta > c.tolerance_seconds else "estimated_fit",
                "actual_tts_duration_seconds": None,
                "estimate_reliable": not analysis["warnings"]}


def wav_duration(path):
    """Read PCM WAV header without generating audio or loading TTS."""
    with wave.open(str(path), "rb") as audio:
        if audio.getnframes() <= 0 or audio.getframerate() <= 0:
            raise ValueError("Empty WAV.")
        return audio.getnframes() / audio.getframerate()


def calibrate(samples, *, model, voice):
    """Samples contain model/voice, mora_count, pause_seconds, actual_seconds.

    Caller must use readings of the exact synthesized text. Never apply another
    speaker/model's speed silently; return a profile for explicit future use.
    """
    mora, spoken = 0, 0.0
    for sample in samples:
        if sample["model"] != model or sample["voice"] != voice:
            raise ValueError("Calibration cannot mix models or voices.")
        n, seconds, pause = sample["mora_count"], sample["actual_seconds"], sample["pause_seconds"]
        if not all(math.isfinite(v) for v in (n, seconds, pause)) or n <= 0 or pause < 0 or seconds <= pause:
            raise ValueError("Invalid calibration sample.")
        mora += n
        spoken += seconds - pause
    if not samples:
        raise ValueError("Calibration requires samples.")
    return {"format_version": 1, "model": model, "voice": voice,
            "mora_per_second": mora / spoken, "sample_count": len(samples)}


def measure_wav(path):
    """Validate the entire PCM payload and count frames, not RIFF duration tags.

    A truncated/corrupt/empty/silent file must never become a successful TTS cache.
    Floating-point WAV is intentionally unsupported; the adapter writes PCM16.
    """
    import hashlib
    import struct
    from pathlib import Path
    try:
        with wave.open(str(path), "rb") as audio:
            frames, rate = audio.getnframes(), audio.getframerate()
            channels, width = audio.getnchannels(), audio.getsampwidth()
            if frames <= 0 or rate <= 0 or channels <= 0 or width not in (1, 2, 3, 4):
                raise ValueError("Invalid WAV dimensions.")
            frame_bytes = channels * width
            seen, energy, peak, clipped = 0, 0, 0, 0
            scale = 2 ** (8 * width - 1)
            while True:
                block = audio.readframes(65536)
                if not block:
                    break
                if len(block) % frame_bytes:
                    raise ValueError("WAV ends inside a sample frame.")
                seen += len(block) // frame_bytes
                if width == 2:
                    values = (v[0] for v in struct.iter_unpack("<h", block))
                elif width == 1:
                    values = (v - 128 for v in block)
                else:
                    values = (int.from_bytes(block[i:i + width], "little", signed=True)
                              for i in range(0, len(block), width))
                for value in values:
                    amplitude = abs(value)
                    peak = max(peak, amplitude)
                    energy += value * value
                    clipped += amplitude >= scale - 1
            if seen != frames:
                raise ValueError(f"Truncated WAV: expected {frames} frames, read {seen}.")
            if peak == 0:
                raise ValueError("WAV contains only silence.")
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return {"sample_rate": rate, "sample_frames": seen, "channels": channels,
                "sample_width_bytes": width, "audio_duration": seen / rate,
                "sha256": digest, "peak_amplitude": peak / scale,
                "rms_amplitude": math.sqrt(energy / (seen * channels)) / scale,
                "clipped_sample_fraction": clipped / (seen * channels)}
    except (wave.Error, EOFError) as error:
        raise ValueError(f"Invalid PCM WAV: {error}") from error


def feedback_profiles(records, *, minimum_samples=3):
    """Return scoped diagnostic profiles. No automatic global speed changes.

    Deduplicate identical audio/text. Unresolved speakers, unreliable readings,
    short samples and unverified/failed generations are not training evidence.
    """
    from .cache import fingerprint
    groups = {}
    for record in records:
        if (record.get("generation_status") != "complete" or
                record.get("speaker") in (None, "UNKNOWN", "REVIEW") or
                not record.get("estimate_reliable") or
                record.get("actual_tts_duration_seconds", 0) < 2 or
                record.get("mora_count", 0) <= 0 or
                record["actual_tts_duration_seconds"] <= record.get("pause_seconds", 0) or
                record.get("audio_quality_warnings")):
            continue
        scope = {"speaker": record["speaker"], "voice": record["voice"],
                 "tts": record["tts"], "analyzer": record.get("analyzer_identity"),
                 "pause_config": record.get("timing_config")}
        key = fingerprint(scope)
        group = groups.setdefault(key, {"scope": scope, "records": {}})
        group["records"][record["audio_cache_key"]] = record
    profiles = []
    for key, group in sorted(groups.items()):
        rows = list(group["records"].values())
        scope = group["scope"]
        samples = [{"model": scope["tts"]["model"], "voice": scope["voice"],
                    "mora_count": row["mora_count"], "pause_seconds": row["pause_seconds"],
                    "actual_seconds": row["actual_tts_duration_seconds"]} for row in rows]
        profile = calibrate(samples, model=scope["tts"]["model"], voice=scope["voice"])
        profiles.append({**profile, "scope_key": key, "scope": scope,
                         "sample_ids": [r["id"] for r in rows],
                         "audio_cache_keys": [r["audio_cache_key"] for r in rows],
                         "minimum_samples": minimum_samples,
                         "eligible_for_explicit_use": len(rows) >= minimum_samples,
                         "auto_applied": False,
                         "note": "Diagnostic fit on training samples, not held-out accuracy."})
    return profiles


def estimate_with_profile(analysis, available, profile, *, scope, config=None):
    """Explicit opt-in only; mismatched speaker/model/settings are rejected."""
    from .cache import fingerprint
    if fingerprint(scope) != profile["scope_key"] or scope != profile["scope"]:
        raise ValueError("Calibration scope does not match this speaker/model/settings.")
    if profile["sample_count"] < profile["minimum_samples"]:
        raise ValueError("Too few samples for this calibration profile.")
    current = config or TimingConfig()
    if asdict(current) != scope["pause_config"]:
        raise ValueError("Calibration timing configuration mismatch.")
    calibrated = TimingConfig(**{**asdict(current), "mora_per_second": profile["mora_per_second"]})
    return {**TimingEngine(calibrated).estimate(analysis, available),
            "calibration_sample_count": profile["sample_count"],
            "calibration_scope_key": profile["scope_key"]}
