"""Conservative evidence checks, never a claim of semantic equivalence."""
from decimal import Decimal
import json
from pathlib import Path
import re
import unicodedata

NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*(?:billion|million|thousand|億|万|千))?\s*(?:%|percent|パーセント)?", re.I)


def numeric_facts(text):
    text = unicodedata.normalize("NFKC", text)
    # Japanese preceding characters are word characters too; delimit digits.
    text = re.sub(r"(?<=[^\x00-\x7f])(?=[+\-\d])", " ", text)
    result = set()
    for m in NUMBER.finditer(text):
        raw = m.group().strip()
        value = Decimal(re.match(r"[+-]?[\d,.]+", raw).group().replace(",", ""))
        for unit, scale in [("billion", 10**9), ("million", 10**6), ("thousand", 1000), ("億", 10**8), ("万", 10000), ("千", 1000)]:
            if unit in raw.lower():
                value *= scale
                break
        unit = "percent" if re.search(r"%|percent|パーセント", raw, re.I) else "number"
        result.add((str(value.normalize()), unit))
    return result


class Rulebook:
    def __init__(self, data=None):
        self.data = data if data is not None else json.loads(Path(__file__).with_name("rules.json").read_text())
        if not isinstance(self.data, dict) or self.data.get("format_version") != 1:
            raise ValueError("Unsupported rulebook format.")
        required = {"instructions", "checks", "name_aliases", "pronunciations", "local_rewrites",
                    "japanese_negation_pattern", "japanese_uncertainty_pattern"}
        if not required <= self.data.keys():
            raise ValueError("Incomplete rulebook.")
        if set(self.data["checks"]) - {"numbers", "names", "negation", "uncertainty"}:
            raise ValueError("Unknown rulebook check.")
        self.negation = re.compile(self.data["japanese_negation_pattern"])
        self.uncertainty = re.compile(self.data["japanese_uncertainty_pattern"])
        for rule in self.data["local_rewrites"]:
            if not rule.get("id") or not rule.get("from") or not rule.get("to"):
                raise ValueError("Invalid local rewrite rule.")

    def rewrite(self, text):
        applied = []
        for rule in self.data["local_rewrites"]:
            if rule["from"] in text:
                text = text.replace(rule["from"], rule["to"])
                applied.append(rule["id"])
        return text, applied

    def has_negation(self, text):
        # These negative-looking forms express uncertainty, not negative polarity.
        return bool(self.negation.search(re.sub(r"のではないか|かもしれない", "", text)))

    def validate(self, text_en, text_ja, english, original_ja=None):
        findings = []
        checks = self.data["checks"]
        if "numbers" in checks:
            source, target = numeric_facts(text_en), numeric_facts(text_ja)
            if source - target:
                findings.append("number_or_percentage_missing_or_changed")
            # A Japanese counter written with kanji or an English number word
            # cannot be proven equivalent by this numeric check.
            if target - source:
                findings.append("number_added_or_not_verified")
            if original_ja is not None and numeric_facts(original_ja) != target:
                findings.append("baseline_numeric_facts_changed")
            if any(not re.search(r"\d", word) for word in english["numbers"]):
                findings.append("number_words_need_review")
        if "names" in checks:
            known = self.data["name_aliases"]
            names = set(english["proper_names"])
            names.update(name for name in known if re.search(r"\b" + re.escape(name) + r"\b", text_en, re.I))
            for name in sorted(names):
                aliases = next((a for n, a in known.items() if n.lower() == name.lower()), [name])
                if not any(alias.casefold() in text_ja.casefold() for alias in aliases):
                    findings.append("name_missing_or_unverified:" + name)
        if "negation" in checks:
            if english["negations"] and not self.has_negation(text_ja):
                findings.append("negation_missing_or_unverified")
            if original_ja and self.has_negation(original_ja) and not self.has_negation(text_ja):
                findings.append("baseline_negation_changed")
        if "uncertainty" in checks:
            if english["uncertainty"] and not self.uncertainty.search(text_ja):
                findings.append("uncertainty_missing_or_unverified")
            if original_ja and self.uncertainty.search(original_ja) and not self.uncertainty.search(text_ja):
                findings.append("baseline_uncertainty_changed")
        return sorted(set(findings))
