"""Reading normalization, not a morphological tokenizer.

Counters/dates and ambiguous homographs need dictionary overrides and review.
"""
import re
import unicodedata

LETTERS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", [
    "エー", "ビー", "シー", "ディー", "イー", "エフ", "ジー", "エイチ", "アイ",
    "ジェー", "ケー", "エル", "エム", "エヌ", "オー", "ピー", "キュー", "アール",
    "エス", "ティー", "ユー", "ブイ", "ダブリュー", "エックス", "ワイ", "ゼット"]))
DIGITS = ["ゼロ", "イチ", "ニ", "サン", "ヨン", "ゴ", "ロク", "ナナ", "ハチ", "キュウ"]
SMALL = set("ァィゥェォャュョヮ")
PAUSES = {"、": "comma", ",": "comma", "。": "sentence", ".": "sentence",
          "！": "sentence", "!": "sentence", "？": "sentence", "?": "sentence",
          "…": "ellipsis", ";": "comma", "：": "comma", ":": "comma", "\n": "sentence"}


def katakana(text):
    text = unicodedata.normalize("NFKC", text)
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


def count_mora(reading):
    """Small kana combine; sokuon, nasal and long vowels each count as one."""
    count = 0
    previous_kana = False
    for char in katakana(reading):
        kana = "ァ" <= char <= "ヺ" or char == "ー"
        if kana and not (char in SMALL and previous_kana):
            count += 1
        previous_kana = kana
    return count


def integer_reading(value):
    n = int(value)
    if n == 0:
        return DIGITS[0]
    if n >= 10**16:
        return "".join(DIGITS[int(c)] for c in str(n))
    for unit, name in [(10**12, "チョウ"), (10**8, "オク"), (10**4, "マン")]:
        if n >= unit:
            q, r = divmod(n, unit)
            return integer_reading(q) + name + (integer_reading(r) if r else "")
    result = ""
    for unit, name, irregular in [
        (1000, "セン", {3: "サンゼン", 8: "ハッセン"}),
        (100, "ヒャク", {3: "サンビャク", 6: "ロッピャク", 8: "ハッピャク"}),
        (10, "ジュウ", {}),
    ]:
        q, n = divmod(n, unit)
        if q:
            result += irregular.get(q, (DIGITS[q] if q != 1 else "") + name)
    return result + (DIGITS[n] if n else "")


def number_reading(text):
    text = unicodedata.normalize("NFKC", text).replace(",", "")
    sign = ""
    if text.startswith(("-", "+")):
        sign, text = ("マイナス" if text[0] == "-" else "プラス"), text[1:]
    whole, dot, fraction = text.partition(".")
    whole_reading = ("".join(DIGITS[int(c)] for c in whole)
                     if len(whole) > 1 and whole.startswith("0") else integer_reading(whole))
    return sign + whole_reading + ("テン" + "".join(DIGITS[int(c)] for c in fraction) if dot else "")


class PronunciationDictionary:
    def __init__(self, entries=None):
        self.entries = {}
        for surface, reading in (entries or {}).items():
            self.register(surface, reading)

    def register(self, surface, reading):
        if not isinstance(surface, str) or not surface or not isinstance(reading, str):
            raise ValueError("Pronunciation dictionary requires surface and kana reading.")
        reading = katakana(reading)
        if not reading or any(not ("ァ" <= c <= "ヺ" or c == "ー") for c in reading):
            raise ValueError("Dictionary readings must contain kana only.")
        self.entries[unicodedata.normalize("NFKC", surface)] = reading

    def normalize(self, text):
        text = unicodedata.normalize("NFKC", text)
        keys = sorted(self.entries, key=len, reverse=True)
        parts = ([re.escape(k) for k in keys] +
                 [r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", r"[A-Z]{2,}", "%"])
        changes = []

        def replace(match):
            surface = match.group()
            if surface in self.entries:
                reading, kind = self.entries[surface], "dictionary"
            elif surface == "%":
                reading, kind = "パーセント", "percent"
            elif re.fullmatch(r"[A-Z]{2,}", surface):
                reading, kind = "".join(LETTERS[c] for c in surface), "acronym"
            else:
                reading, kind = number_reading(surface), "number"
            changes.append({"surface": surface, "reading": reading, "kind": kind,
                            "start": match.start(), "end": match.end()})
            return reading

        return re.sub("|".join(parts), replace, text), changes
