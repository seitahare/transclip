"""spaCy adapter. An explicitly selected blank pipeline reports missing features."""
import re
from importlib.metadata import version


class EnglishAnalyzer:
    def __init__(self, model="en_core_web_sm"):
        import spacy
        if model == "blank":
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")
        else:
            try:
                self.nlp = spacy.load(model)
            except OSError as error:
                raise ValueError(
                    f"Missing English model {model!r}. Install it explicitly, "
                    "or use --english-model blank (no POS/dependency/NER)."
                ) from error
        self.identity = {"engine": "spacy", "version": version("spacy"),
                         "model": model, "model_version": self.nlp.meta.get("version")}

    def analyze(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("English text must be non-empty.")
        doc = self.nlp(text)
        has_pos = doc.has_annotation("POS")
        has_dep = doc.has_annotation("DEP")
        names = [e.text for e in doc.ents if e.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "LOC"}]
        names += [t.text for t in doc if t.pos_ == "PROPN"]
        lower = text.lower().replace("’", "'")
        negations = re.findall(r"\b(?:no|not|never|neither|without|cannot)\b|n't\b", lower)
        uncertainty = re.findall(r"\b(?:may|might|perhaps|probably|possibly|apparently|seems?|think|believe|guess|could)\b", lower)
        content = [t for t in doc if not t.is_punct and not t.is_space]
        reasons = []
        if not re.search(r'[.!?][\s\"\u201d\u2019]*$', text):
            reasons.append("no_terminal_punctuation")
        if content and content[-1].pos_ in {"ADP", "DET", "CCONJ", "SCONJ", "AUX", "PRON"}:
            reasons.append("trailing_incomplete_phrase")
        if has_pos and not any(t.pos_ in {"VERB", "AUX"} for t in content) and len(content) > 1:
            reasons.append("no_finite_predicate_candidate")
        return {
            "tokens": [{"text": t.text, "start": t.idx, "end": t.idx + len(t.text),
                        "pos": t.pos_ or None, "dependency": t.dep_ or None,
                        "head": t.head.i if has_dep else None} for t in doc],
            "sentences": [s.text for s in doc.sents], "proper_names": sorted(set(names)),
            "numbers": [t.text for t in doc if t.like_num],
            "negations": negations, "uncertainty": uncertainty,
            "fragment_reasons": reasons, "possibly_fragmented": bool(reasons),
            "capabilities": {"pos": has_pos, "dependency": has_dep, "ner": doc.has_annotation("ENT_IOB")},
        }
