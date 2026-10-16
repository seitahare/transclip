import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
import wave

from language.cache import CachedProvider, atomic_json, fingerprint
from language.pronunciation import PronunciationDictionary, count_mora, number_reading
from language.rulebook import Rulebook, numeric_facts
from language.timing import TimingConfig, TimingEngine, calibrate, wav_duration
from language.voice import VoiceProfile


def language_dependencies():
    return all(importlib.util.find_spec(n) for n in ("spacy", "en_core_web_sm", "sudachipy", "sudachidict_core"))


class PronunciationTests(unittest.TestCase):
    def test_mora_small_kana_sokuon_nasal_long_vowel(self):
        for text, expected in [("きゃ", 1), ("きって", 3), ("キャットーン", 5),
                               ("コーヒー", 4), ("ティッシュ", 3), ("ん", 1), ("。", 0)]:
            with self.subTest(text=text):
                self.assertEqual(count_mora(text), expected)

    def test_numbers_and_acronyms(self):
        normalized, changes = PronunciationDictionary().normalize("AIは７５％、3.5です。")
        self.assertEqual(normalized, "エーアイはナナジュウゴパーセント、サンテンゴです。")
        self.assertEqual([x["kind"] for x in changes], ["acronym", "number", "percent", "number"])

    def test_numeric_reading_irregular_and_zero(self):
        for number, reading in [("300", "サンビャク"), ("8000", "ハッセン"), ("0", "ゼロ"),
                                ("-3.5", "マイナスサンテンゴ"), ("001", "ゼロゼロイチ"),
                                ("10000", "イチマン"), ("1,000", "セン")]:
            self.assertEqual(number_reading(number), reading)

    def test_dictionary_longest_match_and_registration(self):
        d = PronunciationDictionary({"OpenAI": "オープンエーアイ", "AI": "エーアイ"})
        d.register("Dario", "だりお")
        self.assertEqual(d.normalize("OpenAIとDario")[0], "オープンエーアイとダリオ")
        with self.assertRaises(ValueError):
            d.register("name", "unreadable")


class TimingTests(unittest.TestCase):
    def test_estimate_mora_pauses_and_delta(self):
        timing = TimingEngine(TimingConfig(mora_per_second=5, comma_pause=.1, sentence_pause=.2))
        result = timing.estimate({"mora_count": 10, "pauses": [{"kind": "comma"}, {"kind": "sentence"}], "warnings": []}, 1)
        self.assertAlmostEqual(result["estimated_duration_seconds"], 2.3)
        self.assertAlmostEqual(result["duration_delta_seconds"], 1.3)
        self.assertEqual(result["timing_status"], "overlong")
        self.assertIsNone(result["actual_tts_duration_seconds"])

    def test_invalid_timing_settings(self):
        for value in [0, -1, math.nan, math.inf, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                TimingConfig(mora_per_second=value)
        with self.assertRaises(ValueError):
            TimingConfig(comma_pause=-1)

    def test_pcm_wav_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\0\0" * 4000)
            self.assertEqual(wav_duration(path), .5)

    def test_voice_scoped_calibration(self):
        sample = dict(model="qwen", voice="Ryan", mora_count=70, actual_seconds=11, pause_seconds=1)
        profile = calibrate([sample], model="qwen", voice="Ryan")
        self.assertEqual(profile["mora_per_second"], 7)
        with self.assertRaises(ValueError):
            calibrate([sample], model="qwen", voice="Aiden")
        with self.assertRaises(ValueError):
            calibrate([], model="qwen", voice="Ryan")

    def test_reference_voice_requires_consent(self):
        with self.assertRaises(ValueError):
            VoiceProfile("A", mode="approved_reference", reference_audio="voice.wav")
        self.assertEqual(VoiceProfile("A", voice="Ryan").mode, "preset")


class RulebookTests(unittest.TestCase):
    def setUp(self):
        self.rules = Rulebook()
        self.english = {"numbers": ["75"], "proper_names": ["Dario"], "negations": ["not"], "uncertainty": ["might"]}

    def test_preserve_number_name_negation_uncertainty(self):
        self.assertEqual(self.rules.validate("Dario might not reach 75%.", "ダリオは75％に届かないかもしれない。", self.english), [])

    def test_dropped_protected_facts(self):
        issues = self.rules.validate("Dario might not reach 75%.", "到達する。", self.english)
        for issue in ["number_or_percentage_missing_or_changed", "name_missing_or_unverified:Dario", "negation_missing_or_unverified", "uncertainty_missing_or_unverified"]:
            self.assertIn(issue, issues)

    def test_percent_not_plain_number(self):
        self.assertNotEqual(numeric_facts("75 percent"), numeric_facts("75"))
        self.assertEqual(numeric_facts("75 percent"), numeric_facts("75％"))
        self.assertEqual(numeric_facts("1.5 million"), numeric_facts("150万"))
        self.assertEqual(numeric_facts("-3.5%"), numeric_facts("−3.5%".replace("−", "-")))

    def test_baseline_bad_numbers_cannot_be_changed_again(self):
        issues = self.rules.validate("75%", "60％", self.english, "70％")
        self.assertIn("baseline_numeric_facts_changed", issues)

    def test_hedge_not_treated_as_negative_polarity(self):
        self.assertFalse(self.rules.has_negation("仕事がなくなるのではないかと思います。".replace("なくなる", "変わる")))
        self.assertTrue(self.rules.has_negation("来ないかもしれない。"))

    def test_unknown_rule_and_version_rejected(self):
        for change in [{"format_version": 2}, {"checks": ["made_up_guard"]}]:
            data = copy.deepcopy(self.rules.data)
            data.update(change)
            with self.assertRaises(ValueError):
                Rulebook(data)


class Provider:
    def __init__(self):
        self.identity = {"model": "fake", "prompt_version": 1}
        self.calls = 0

    def propose(self, request):
        self.calls += 1
        return "短い案です。"


class CacheTests(unittest.TestCase):
    def test_cache_and_resume_with_zero_call_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            provider = Provider()
            cache = CachedProvider(provider, temp, max_calls=1)
            request = {"text_en": "Example", "rules": {"revision": 1}}
            self.assertEqual(cache.propose(request), cache.propose(request))
            self.assertEqual(provider.calls, 1)
            resumed = CachedProvider(provider, temp, max_calls=0)
            self.assertEqual(resumed.propose(request), "短い案です。")

    def test_rules_context_model_dictionary_and_source_invalidate(self):
        with tempfile.TemporaryDirectory() as temp:
            provider = Provider()
            cache = CachedProvider(provider, temp, max_calls=10)
            base = {"source": "a", "rules": 1, "context": "a", "dictionary": 1, "timing": 7}
            cache.propose(base)
            for key in base:
                changed = dict(base)
                changed[key] = "different"
                cache.propose(changed)
            provider.identity["model"] = "different"
            cache.propose(base)
            self.assertEqual(provider.calls, 7)

    def test_failed_call_not_cached_and_no_automatic_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            class Failure(Provider):
                def propose(self, request):
                    raise RuntimeError("failure")
            cache = CachedProvider(Failure(), temp, max_calls=1)
            with self.assertRaisesRegex(RuntimeError, "failure"):
                cache.propose({})
            self.assertEqual(list(Path(temp).iterdir()), [])
            with self.assertRaisesRegex(RuntimeError, "budget"):
                cache.propose({})

    def test_atomic_no_clobber(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "out.json"
            atomic_json(path, {"a": 1}, overwrite=False)
            with self.assertRaises(FileExistsError):
                atomic_json(path, {"a": 2}, overwrite=False)
            self.assertEqual(json.loads(path.read_text()), {"a": 1})
            self.assertEqual(len(list(Path(temp).iterdir())), 1)

    def test_corrupt_cache_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = CachedProvider(Provider(), temp, max_calls=1)
            cache.propose({})
            next(Path(temp).iterdir()).write_text('{"key":"wrong"}')
            with self.assertRaises(ValueError):
                cache.propose({})


@unittest.skipUnless(language_dependencies(), "Install requirements-language.txt for real analyzer tests")
class AnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from language.english import EnglishAnalyzer
        from language.japanese import JapaneseAnalyzer
        cls.en = EnglishAnalyzer()
        cls.ja = JapaneseAnalyzer({"Dario": "ダリオ", "AI": "エーアイ"})

    def test_sentence_boundaries_abbreviation_and_decimal(self):
        result = self.en.analyze("Dr. Smith paid 3.5 dollars. Is that correct?")
        self.assertEqual(result["sentences"], ["Dr. Smith paid 3.5 dollars.", "Is that correct?"])
        self.assertTrue(result["capabilities"]["dependency"])
        self.assertTrue(all(t["pos"] for t in result["tokens"]))

    def test_negation_entities_and_fragment(self):
        result = self.en.analyze("Dario might not agree with 75%.")
        self.assertIn("Dario", result["proper_names"])
        self.assertIn("not", result["negations"])
        self.assertIn("might", result["uncertainty"])
        self.assertTrue(self.en.analyze("if you look at,")["possibly_fragmented"])
        self.assertFalse(self.en.analyze("We agreed.")["possibly_fragmented"])

    def test_japanese_morphology_reading_and_pauses(self):
        result = self.ja.analyze("東京都で話します。AIは75％です。")
        self.assertIn("トウキョウト", result["reading"])
        self.assertIn("ナナジュウゴパーセント", result["reading"])
        self.assertTrue(all(t["pos"] for t in result["morphology"]))
        self.assertEqual(len(result["pauses"]), 2)
        self.assertFalse(result["warnings"])

    def test_dictionary_and_unresolved_reading(self):
        self.assertEqual(self.ja.analyze("Dario")["reading"], "ダリオ")
        self.assertTrue(self.ja.analyze("xyzqwerty")["warnings"])
        self.assertIn("counter_or_date_reading_needs_review", self.ja.analyze("3人")["warnings"])

    def test_blank_pipeline_explicit_limitations(self):
        from language.english import EnglishAnalyzer
        result = EnglishAnalyzer("blank").analyze("One sentence. Another one.")
        self.assertEqual(len(result["sentences"]), 2)
        self.assertFalse(result["capabilities"]["pos"])

    def test_empty_input_rejected(self):
        for analyzer in [self.en, self.ja]:
            with self.assertRaises(ValueError):
                analyzer.analyze("")


if __name__ == "__main__":
    unittest.main()
