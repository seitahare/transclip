"""Future TTS boundary. This module generates no audio."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VoiceProfile:
    speaker: str
    mode: str = "preset"
    voice: str | None = None
    reference_audio: str | None = None
    consent_reference: str | None = None

    def __post_init__(self):
        if self.mode not in {"preset", "approved_reference"}:
            raise ValueError("Unsupported voice mode.")
        if self.mode == "approved_reference" and not (self.reference_audio and self.consent_reference):
            raise ValueError("Reference voice requires audio and documented consent.")
        if self.mode == "preset" and self.reference_audio:
            raise ValueError("Reference audio cannot be supplied as a preset.")


class SpeechSynthesizer(Protocol):
    def synthesize(self, *, text: str, profile: VoiceProfile, output_path: str) -> dict:
        """Return path and actual duration. Never truncate to another turn's start."""
        ...
