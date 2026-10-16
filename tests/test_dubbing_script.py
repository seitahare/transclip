import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import math
from pathlib import Path
import tempfile
import unittest

from dubbing_script import optimize, validate_inputs, main, read_json, save_result, comparison_markdown
from language.cache import fingerprint, CachedProvider
from test_language import language_dependencies

FIXTURE = Path(__file__).parent / "fixtures" / "language_mvp"


def inputs():
    return read_json(FIXTURE / "utterances.json"), read_json(FIXTURE / "utterance_translations.json")


class ContractTests(unittest.TestCase):
    def test_existing_v3_schema(self):
        clip, items, mapping = validate_inputs(*inputs())
        self.assertEqual(clip, {"start": 100, "end": 145})
        self.assertEqual(len(items), 8)
        self.assertEqual(mapping[1]["text_ja"], "ダリオが言っていたように、")

    def test_invalid_or_stale_inputs(self):
        mutations = [
            lambda s,t: s["utterances"].append(copy.deepcopy(s["utterances"][0])),
            lambda s,t: t["utterances"].pop(),
            lambda s,t: t["clip"].update(start=101),
            lambda s,t: t.update(video_id="wrong"),
            lambda s,t: s.update(format_version=99),
            lambda s,t: t["utterances"][0].update(text_en="Changed source"),
            lambda s,t: t["utterances"][0].update(speaker="B"),
            lambda s,t: t["utterances"][0].update(text_ja=""),
            lambda s,t: s["utterances"][0].update(start=math.nan),
            lambda s,t: s["utterances"][0].update(start=True),
            lambda s,t: s["utterances"][0].update(relative_start=-1),
            lambda s,t: s["utterances"][0].update(relative_end=0),
            lambda s,t: s["utterances"][0].update(start=110),
            lambda s,t: s["utterances"][0].update(speaker=123),
            lambda s,t: s["utterances"][0].update(id=True),
            lambda s,t: s["utterances"][0].update(review_reasons="not a list"),
            lambda s,t: s["utterances"][0].update(words=["not an object"]),
            lambda s,t: s.update(utterances=[]),
            lambda s,t: s.update(video_id=None),
            lambda s,t: t.update(format_version=99),
            lambda s,t: s["utterances"][0].pop("speaker"),
            lambda s,t: s["utterances"][0].update(review_required="false"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                s,t = inputs()
                mutate(s,t)
                with self.assertRaises(ValueError):
                    validate_inputs(s,t)

    def test_selection_range_must_match(self):
        with self.assertRaisesRegex(ValueError, "selection"):
            validate_inputs(*inputs(), selection={"start": 101, "end": 145})

    def test_cli_bad_json_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "utterances.json").write_text("{bad}")
            with redirect_stderr(io.StringIO()) as error:
                code = main(["--input-dir", temp])
            self.assertEqual(code, 1)
            self.assertIn("Error:", error.getvalue())


@unittest.skipUnless(language_dependencies(), "Install requirements-language.txt for integration tests")
class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from language.english import EnglishAnalyzer
        from language.japanese import JapaneseAnalyzer
        from language.rulebook import Rulebook
        cls.en, cls.rules = EnglishAnalyzer(), Rulebook()
        cls.ja = JapaneseAnalyzer(cls.rules.data["pronunciations"])

    def run_optimizer(self, source=None, translated=None, **kwargs):
        if source is None:
            source, translated = inputs()
        return optimize(source, translated, english=self.en, japanese=self.ja, rulebook=self.rules, **kwargs)

    def candidate(self, ids, text):
        s,t = inputs()
        return {"format_version": 1, "input_fingerprint": fingerprint({"source": s,"translations": t}),
                "entries": [{"source_utterance_ids": ids, "text_ja": text}]}

    def test_group_same_speaker_keep_different_and_unknown(self):
        result = self.run_optimizer()
        self.assertEqual(result["entries"][0]["source_utterance_ids"], [1,2])
        self.assertEqual(result["entries"][1]["source_utterance_ids"], [3])
        unknowns = [e for e in result["entries"] if e["speaker"] is None]
        self.assertEqual([e["source_utterance_ids"] for e in unknowns], [[4],[5]])
        self.assertNotEqual(unknowns[0]["timeline_id"], unknowns[1]["timeline_id"])
        self.assertTrue(all(e["review_required"] for e in unknowns))

    def test_overlapping_speakers_not_serialized_or_truncated(self):
        first, second = self.run_optimizer()["entries"][:2]
        self.assertEqual(first["relative_end"],5)
        self.assertEqual(second["relative_start"],3)
        self.assertIn(second["id"], first["overlapping_entry_ids"])
        self.assertNotEqual(first["timeline_id"],second["timeline_id"])

    def test_coverage_and_full_source_evidence(self):
        source, translated = inputs()
        result = self.run_optimizer(source, translated)
        covered = [i for e in result["entries"] for i in e["source_utterance_ids"]]
        self.assertEqual(sorted(covered), [1,2,3,4,5,6])
        self.assertEqual(result["entries"][0]["provenance"][0]["source"], source["utterances"][0])
        self.assertEqual(result["entries"][0]["provenance"][0]["translation"], translated["utterances"][0])

    def test_30_second_boundary_explicitly_excluded(self):
        self.assertEqual(self.run_optimizer()["excluded"], [
            {"source_utterance_id":7,"reason":"crosses_preview_boundary","relative_start":28,"relative_end":32},
            {"source_utterance_id":8,"reason":"outside_preview","relative_start":33,"relative_end":35}])

    def test_exact_end_boundary_included(self):
        result = self.run_optimizer(seconds=32)
        self.assertIn(7, [i for e in result["entries"] for i in e["source_utterance_ids"]])

    def test_invalid_preview_and_empty_preview(self):
        for seconds in [-1, 0, math.nan, math.inf, .01]:
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                self.run_optimizer(seconds=seconds)

    def test_offline_rewrite_shortens_preserves_hedge_and_review(self):
        entry = next(e for e in self.run_optimizer()["entries"] if e["source_utterance_ids"] == [6])
        self.assertEqual(entry["text_ja"], "機能すると思います。")
        self.assertLess(entry["estimated_duration_seconds"],entry["original_timing"]["estimated_duration_seconds"])
        self.assertIn("rewrite_semantics_need_review",entry["review_reasons"])

    def test_dropped_number_negation_or_name_rejected(self):
        proposals = ["ダリオによるとAIは仕事を置き換えないかもしれない。",
                     "ダリオによるとAIは75％の仕事を置き換えるかもしれない。",
                     "AIは75％の仕事を置き換えないかもしれない。"]
        for text in proposals:
            with self.subTest(text=text):
                entry = self.run_optimizer(candidates=self.candidate([1,2],text))["entries"][0]
                self.assertEqual(entry["text_ja"],entry["original_text_ja"])
                self.assertIn("candidate_rejected",entry["review_reasons"])
                self.assertFalse(entry["proposal"]["selected"])

    def test_hedge_loss_rejected(self):
        entry = self.run_optimizer(candidates=self.candidate([6],"機能します。"))["entries"][-1]
        self.assertIn("uncertainty_missing_or_unverified",entry["proposal"]["rejected_reasons"])

    def test_stale_or_cross_speaker_candidates_rejected(self):
        for candidate in [self.candidate([1,2,3],"混ぜない。"), self.candidate([99],"なし。"),
                          {**self.candidate([6],"機能すると思います。"),"input_fingerprint":"stale"}]:
            with self.assertRaises(ValueError):
                self.run_optimizer(candidates=candidate)

    def test_self_overlap_detected_without_merging(self):
        s,t=inputs()
        for data in [s,t]: data["utterances"][2]["speaker"]="A"
        a,b=self.run_optimizer(s,t)["entries"][:2]
        self.assertEqual(b["source_utterance_ids"],[3])
        self.assertIn(b["id"],a["same_speaker_conflict_ids"])
        self.assertEqual(a["relative_end"],5)

    def test_literal_unknown_is_not_a_shared_identity(self):
        s,t=inputs()
        for data in [s,t]:
            for item in data["utterances"]:
                if item["speaker"] is None: item["speaker"]="UNKNOWN"
        unknown=[e for e in self.run_optimizer(s,t)["entries"] if e["speaker"]=="UNKNOWN"]
        self.assertEqual(len(unknown),2)
        self.assertNotEqual(unknown[0]["timeline_id"],unknown[1]["timeline_id"])

    def test_long_gap_not_merged(self):
        s,t=inputs()
        for data in [s,t]:
            data["utterances"][1].update(relative_start=2.9,start=102.9)
        self.assertEqual(self.run_optimizer(s,t)["entries"][0]["source_utterance_ids"],[1])

    def test_rejected_candidates_preserve_numeric_values_even_if_baseline_wrong(self):
        s,t=inputs()
        t["utterances"][1]["text_ja"]=t["utterances"][1]["text_ja"].replace("75", "70")
        candidate=self.candidate([1,2],"ダリオによるとAIは60％の仕事を置き換えないかもしれない。")
        candidate["input_fingerprint"]=fingerprint({"source":s,"translations":t})
        e=self.run_optimizer(s,t,candidates=candidate)["entries"][0]
        self.assertIn("baseline_numeric_facts_changed",e["proposal"]["rejected_reasons"])
        self.assertEqual(e["text_ja"],e["original_text_ja"])

    def test_unknown_barrier_not_bridged(self):
        s,t=inputs()
        for data in [s,t]: data["utterances"][1]["speaker"]=None
        result=self.run_optimizer(s,t)
        self.assertEqual(result["entries"][0]["source_utterance_ids"],[1])
        self.assertIsNone(result["entries"][1]["speaker"])

    def test_input_not_mutated_and_order_normalized(self):
        s,t=inputs()
        s["utterances"].reverse()
        before=copy.deepcopy((s,t))
        result=self.run_optimizer(s,t)
        self.assertEqual((s,t),before)
        self.assertEqual(result["entries"][0]["source_utterance_ids"],[1,2])

    def test_cache_key_dictionary_and_rate_sensitive(self):
        from language.japanese import JapaneseAnalyzer
        from language.timing import TimingEngine, TimingConfig
        base=self.run_optimizer()
        changed=self.run_optimizer(timing=TimingEngine(TimingConfig(mora_per_second=8)))
        self.assertNotEqual(base["cache_key"],changed["cache_key"])
        result=optimize(*inputs(),english=self.en,japanese=JapaneseAnalyzer({"Dario":"ダーリオ"}))
        self.assertNotEqual(base["cache_key"],result["cache_key"])

    def test_saved_result_idempotent_but_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"out.json"
            result=self.run_optimizer()
            self.assertTrue(save_result(path,result))
            self.assertFalse(save_result(path,result))
            result["summary"]["entry_count"]=99
            with self.assertRaises(FileExistsError): save_result(path,result)

    def test_provider_context_rules_and_resumption(self):
        class FakeProvider:
            identity={"model":"fake","prompt":1}
            def __init__(self): self.requests=[]
            def propose(self, request):
                self.requests.append(request)
                return request["original_text_ja"]
        with tempfile.TemporaryDirectory() as temp:
            provider=FakeProvider()
            first=self.run_optimizer(provider=CachedProvider(provider,temp,max_calls=20))
            count=len(provider.requests)
            second=self.run_optimizer(provider=CachedProvider(provider,temp,max_calls=0))
            self.assertEqual(first,second)
            self.assertEqual(len(provider.requests),count)
            self.assertTrue(provider.requests[0]["context_reference_only"])
            self.assertIn("instructions",provider.requests[0]["rules"])
            self.assertEqual(provider.requests[0]["source_utterance_ids"],[1,2])

    def test_cli_fixture_and_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            out,report=Path(temp)/"out.json",Path(temp)/"report.md"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code=main(["--input-dir",str(FIXTURE),"--output",str(out),"--report",str(report)])
            self.assertEqual(code,0)
            self.assertEqual(read_json(out)["format_version"],1)
            self.assertEqual(report.read_text(),comparison_markdown(read_json(out)))
            self.assertIn("既存訳",report.read_text())


if __name__ == "__main__":
    unittest.main()
