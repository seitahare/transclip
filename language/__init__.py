"""Replaceable, local language analysis; no API or TTS imports."""
from typing import Protocol


class Analyzer(Protocol):
    @property
    def identity(self) -> dict: ...

    def analyze(self, text: str) -> dict: ...
