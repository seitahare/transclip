"""Sudachi adapter; text is never replaced with pronunciation in the script."""
from importlib.metadata import version
import re
from .pronunciation import PronunciationDictionary, count_mora, katakana, PAUSES


class JapaneseAnalyzer:
    def __init__(self, pronunciations=None):
        from sudachipy import dictionary, tokenizer
        self.tokenizer = dictionary.Dictionary(dict="core").create()
        self.mode = tokenizer.Tokenizer.SplitMode.C
        self.dictionary = PronunciationDictionary(pronunciations)

    @property
    def identity(self):
        return {"engine": "sudachi", "version": version("SudachiPy"),
                "dictionary_version": version("SudachiDict-core"),
                "pronunciations": self.dictionary.entries}

    def analyze(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Japanese text must be non-empty.")
        normalized, changes = self.dictionary.normalize(text)
        tokens, warnings, readings = [], [], []
        for m in self.tokenizer.tokenize(normalized, self.mode):
            surface = m.surface()
            reading = katakana(m.reading_form())
            spoken = any(c.isalnum() for c in surface)
            if not spoken:
                reading = ""
            if spoken and any(not ("ァ" <= c <= "ヺ" or c == "ー" or c.isspace()) for c in reading):
                warnings.append("unresolved_reading:" + surface)
            readings.append(reading)
            tokens.append({"surface": surface, "pos": list(m.part_of_speech()),
                           "reading": reading, "oov": m.is_oov(),
                           "start": m.begin(), "end": m.end()})
        # Preserve source morphology as well as pronunciation-normalized tokens.
        morphology = [{"surface": m.surface(), "pos": list(m.part_of_speech()),
                       "reading": m.reading_form()} for m in self.tokenizer.tokenize(text, self.mode)]
        pauses = [{"text": m.group(), "kind": PAUSES[m.group()[0]], "start": m.start()}
                  for m in re.finditer("[" + re.escape("".join(PAUSES)) + "]+", normalized)]
        if re.search(r"\d+\s*(?:人|本|匹|日|月|歳|才|個|回|時|分)", text):
            warnings.append("counter_or_date_reading_needs_review")
        reading = "".join(readings)
        return {"morphology": morphology, "tokens": tokens, "reading": reading,
                "mora_count": count_mora(reading), "pauses": pauses,
                "normalizations": changes, "warnings": sorted(set(warnings))}
