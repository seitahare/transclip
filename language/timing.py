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
