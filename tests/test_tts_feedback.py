import copy
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave
from unittest.mock import patch
from dataclasses import asdict
from language.cache import atomic_json, fingerprint
from language.timing import TimingConfig, measure_wav, feedback_profiles, estimate_with_profile
from tts_feedback import run_feedback, plan_script, validate_script
import tts_generator


def wav(path, seconds=1, rate=8000, channels=1):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        samples = [int(8000 * math.sin(2*math.pi*220*i/rate)) for i in range(round(seconds*rate))]
        stream.writeframes(b"".join(struct.pack("<h", x)*channels for x in samples))


def script():
    result = {"format_version": 1, "video_id": "fixture", "clip": {"start": 100, "end": 140},
              "preview": {"relative_start": 0, "relative_end": 30},
              "configuration": {"japanese": {"engine": "fake", "version": 1}, "timing": asdict(TimingConfig())},
              "excluded": [{"source_utterance_id": 5, "reason": "crosses_preview_boundary"}], "entries": []}
    for uid, speaker, start, end, text in [(1,"A",0,1,"最初です。"),(2,"B",.5,1.5,"はい。"),
                                          (3,"A",1.5,2.5,"次です。"),(4,None,3,4,"不明です。")]:
        result["entries"].append({"id": f"dub-{uid}", "speaker": speaker, "source_utterance_ids": [uid],
            "start": 100+start, "end": 100+end, "relative_start": start, "relative_end": end,
            "text_en": "Example.", "text_ja": text, "original_text_ja": text,
            "source_duration_seconds": end-start, "estimated_duration_seconds": .8,
            "pause_seconds": .25, "estimate_reliable": True,
            "japanese_analysis": {"reading": "テスト", "mora_count": 10, "pauses": [{"kind": "sentence"}]},
            "review_required": speaker is None, "review_reasons": ["unknown_speaker"] if speaker is None else [],
            "actual_tts_duration_seconds": None})
    return result


class FakeBackend:
    def __init__(self, duration=2.5, fail_at=None):
        self.identity = {"model": "fake", "backend": "mock", "settings": {"seed": 0}}
        self.calls = []
        self.duration, self.fail_at = duration, fail_at

    def synthesize(self, *, text, profile, output_path):
        self.calls.append((text,profile))
        if len(self.calls) == self.fail_at: raise RuntimeError("Injected TTS failure")
        wav(output_path,self.duration)
        return {"termination": "eos_before_limit", "token_count": 31}


class WavMeasurementTests(unittest.TestCase):
    def test_exact_frames_rate_and_stereo_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"audio.wav";wav(path,.125,16000,2)
            m=measure_wav(path)
            self.assertEqual((m["sample_frames"],m["sample_rate"],m["channels"],m["audio_duration"]),(2000,16000,2,.125))
            self.assertGreater(m["rms_amplitude"],0)

    def test_truncated_invalid_empty_and_silent_wav(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"audio.wav"
            for variant in ("truncated","not_wav","empty","silent"):
                with self.subTest(variant=variant):
                    wav(path)
                    if variant == "truncated": path.write_bytes(path.read_bytes()[:-100])
                    if variant == "not_wav": path.write_bytes(b"this is not audio")
                    if variant in ("empty","silent"):
                        with wave.open(str(path),"wb") as f:
                            f.setparams((1,2,8000,0,"NONE","not compressed"))
                            f.writeframes(b"" if variant=="empty" else b"\0\0"*8000)
                    with self.assertRaises(ValueError): measure_wav(path)


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/"script.json"
        self.data=script();atomic_json(self.source,self.data)
        self.directory=self.root/"feedback";self.voices={"A":"Ryan","B":"Aiden"}

    def run_engine(self, backend=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            path=run_feedback(self.source,self.directory,backend or FakeBackend(),self.voices,**kwargs)
        return path,json.loads(path.read_text())

    def test_measured_error_source_delta_and_independent_timeline(self):
        _,r=self.run_engine();a,b,c,unknown=r["entries"]
        self.assertEqual(a["actual_tts_duration_seconds"],2.5)
        self.assertAlmostEqual(a["prediction_error_seconds"],1.7)
        self.assertAlmostEqual(a["actual_minus_source_seconds"],1.5)
        self.assertEqual((a["actual_playback_end"],a["relative_end"],b["relative_start"]),(2.5,1,.5))
        self.assertEqual(a["actual_same_speaker_conflicts"][0]["entry_id"],c["id"])
        self.assertEqual(a["actual_other_speaker_overlaps"][0]["entry_id"],b["id"])
        self.assertEqual(a["actual_other_speaker_overlaps"][0]["source_overlap_seconds"],.5)
        self.assertTrue(a["review_required"])

    def test_unknown_kept_and_source_bytes_unchanged(self):
        before=self.source.read_bytes();backend=FakeBackend();_,r=self.run_engine(backend)
        self.assertEqual(len(backend.calls),3)
        unknown=r["entries"][-1]
        self.assertIsNone(unknown["speaker"]);self.assertIsNone(unknown["actual_tts_duration_seconds"])
        self.assertEqual(unknown["skip_reason"],"unknown_speaker")
        self.assertEqual(before,self.source.read_bytes());self.assertEqual(r["source_excluded"],self.data["excluded"])

    def test_cache_reused_without_backend_calls(self):
        self.run_engine();second=FakeBackend(fail_at=1);_,r=self.run_engine(second)
        self.assertFalse(second.calls)
        self.assertTrue(all(e["cache_reused"] for e in r["entries"][:3]))

    def test_resume_after_second_job_failure(self):
        with self.assertRaisesRegex(RuntimeError,"Injected"):self.run_engine(FakeBackend(fail_at=2))
        path=next((self.directory/"runs").glob("*/feedback.json"));partial=json.loads(path.read_text())
        self.assertEqual(partial["status"],"failed")
        self.assertEqual(partial["entries"][0]["generation_status"],"complete")
        self.assertEqual(partial["entries"][1]["generation_status"],"failed")
        later=FakeBackend();_,done=self.run_engine(later)
        self.assertEqual(len(later.calls),2);self.assertEqual(done["status"],"complete_with_skips")
        self.assertTrue(done["entries"][0]["cache_reused"])
        self.assertFalse(list(self.directory.glob("audio/*/.pending-*")))

    def test_budgeted_run_resumes(self):
        _,partial=self.run_engine(max_jobs=1)
        self.assertEqual(partial["summary"]["complete"],1);self.assertEqual(partial["status"],"partial")
        later=FakeBackend();_,complete=self.run_engine(later)
        self.assertEqual(len(later.calls),2);self.assertEqual(complete["summary"]["complete"],3)

    def test_changed_text_settings_or_voice_invalidates_relevant_audio(self):
        self.run_engine();self.data["entries"][0]["text_ja"]="変更です。";atomic_json(self.source,self.data)
        backend=FakeBackend();self.run_engine(backend);self.assertEqual(len(backend.calls),1)
        backend=FakeBackend();backend.identity["settings"]["seed"]=123
        self.run_engine(backend);self.assertEqual(len(backend.calls),3)
        self.voices["A"]="Ono_Anna";backend=FakeBackend()
        self.run_engine(backend);self.assertEqual(len(backend.calls),2)

    def test_cache_reused_after_timeline_change(self):
        self.run_engine();self.data["entries"][0].update(start=100.1,relative_start=.1,end=101.1,relative_end=1.1)
        atomic_json(self.source,self.data);backend=FakeBackend();_,r=self.run_engine(backend)
        self.assertFalse(backend.calls);self.assertAlmostEqual(r["entries"][0]["actual_playback_end"],2.6)

    def test_corrupt_cache_not_overwritten(self):
        _,r=self.run_engine();path=Path(r["entries"][0]["audio_file"]);path.write_bytes(b"bad")
        backend=FakeBackend()
        with self.assertRaises(ValueError):self.run_engine(backend)
        self.assertEqual(path.read_bytes(),b"bad");self.assertFalse(backend.calls)

    def test_valid_but_modified_cache_rejected(self):
        _,r=self.run_engine();wav(Path(r["entries"][0]["audio_file"]),3)
        with self.assertRaisesRegex(ValueError,"changed"):self.run_engine()

    def test_bad_generated_audio_not_published(self):
        class Bad(FakeBackend):
            def synthesize(self, **kw):
                Path(kw["output_path"]).write_bytes(b"bad");return {}
        with self.assertRaises(ValueError):self.run_engine(Bad())
        self.assertFalse(list(self.directory.glob("audio/*/*/speech.wav")))

    def test_invalid_schema_and_boundary(self):
        for key,value in [("estimated_duration_seconds",math.nan),("relative_start",-1),
                          ("source_duration_seconds",2),("text_ja",""),("review_required","yes")]:
            data=copy.deepcopy(self.data);data["entries"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_script(data)
        rows=plan_script(self.data,self.voices,seconds=1.2)
        self.assertEqual([r["skip_reason"] for r in rows[:3]],[None,"crosses_preview_boundary","outside_preview"])
        with self.assertRaises(ValueError):plan_script(self.data,{"UNKNOWN":"Ryan"})
        with self.assertRaises(ValueError):plan_script(self.data,self.voices,entry_ids=["no-such-id"])

    def test_baseline_exact_matching_and_new_measurement(self):
        old_dir=self.root/"old";old_dir.mkdir();wav(old_dir/"old.wav",3)
        baseline={"video_id":"fixture","model":"fake","jobs":[{"speaker":"A","voice":"Ryan",
                  "utterance_ids":[1],"text_ja":"最初です。","start":0,"end":1,"audio_file":"old.wav"}]}
        atomic_json(old_dir/"manifest.json",baseline)
        _,r=self.run_engine(baseline_manifest=old_dir/"manifest.json");c=r["baseline_comparisons"][0]
        self.assertEqual(c["baseline_measurement"]["audio_duration"],3);self.assertEqual(c["reduction_seconds"],.5)
        self.assertFalse(c["settings_comparable"]);self.assertEqual(c["provenance"],"legacy_exact_text_ids_speaker_relative_times")

    def test_cli_dry_run_no_backend(self):
        voices=self.root/"voices.json";atomic_json(voices,self.voices)
        with patch.object(tts_generator,"QwenPresetSynthesizer") as backend,redirect_stdout(io.StringIO()):
            code=tts_generator.main(["fixture","--script",str(self.source),"--voice-map",str(voices),"--dry-run"])
        self.assertEqual(code,0);backend.assert_not_called()

    def test_script_hash_rejects_stale_readings(self):
        self.data["input_fingerprint"]="source"
        self.data["cache_key"]=fingerprint({"input":"source", "config":self.data["configuration"], "entries":self.data["entries"]})
        validate_script(self.data)
        self.data["entries"][0]["text_ja"]="変更です。"
        with self.assertRaisesRegex(ValueError,"hash mismatch"):validate_script(self.data)

    def test_receipt_survives_failure_saving_progress(self):
        import tts_feedback
        original=tts_feedback.atomic_json
        calls=[]
        def flaky(path,data,**kw):
            if Path(path).name=="feedback.json":
                calls.append(1)
                if len(calls)==2:raise OSError("progress write failure")
            return original(path,data,**kw)
        with patch.object(tts_feedback,"atomic_json",side_effect=flaky):
            with self.assertRaisesRegex(OSError,"progress write"):self.run_engine()
        backend=FakeBackend();_,r=self.run_engine(backend)
        self.assertEqual(len(backend.calls),2)
        self.assertTrue(r["entries"][0]["cache_reused"])

    def test_offline_adapter_limits_cached_files_without_download(self):
        import types
        from unittest.mock import Mock
        snapshot=self.root/"snapshot";snapshot.mkdir();(snapshot/"speech_tokenizer").mkdir()
        (snapshot/"config.json").write_text('{"tts_model_type":"custom_voice"}')
        (snapshot/"speech_tokenizer/config.json").write_text('{}')
        (snapshot/"model.safetensors").write_bytes(b"fake")
        (snapshot/"speech_tokenizer/model.safetensors").write_bytes(b"fake")
        download=Mock(return_value=str(snapshot))
        with patch.dict("sys.modules",{"huggingface_hub":types.SimpleNamespace(snapshot_download=download)}), \
             patch("importlib.metadata.version",return_value="test"), \
             patch("importlib.util.find_spec",return_value=types.SimpleNamespace(origin=str(snapshot/"__init__.py"))):
            backend=tts_generator.QwenPresetSynthesizer()
        self.assertTrue(download.call_args.kwargs["local_files_only"])
        self.assertIn("*.safetensors",download.call_args.kwargs["allow_patterns"])
        self.assertNotIn("*.md",download.call_args.kwargs["allow_patterns"])
        self.assertIsNone(backend._model)

    def test_legacy_cli_preserved(self):
        with patch.object(tts_generator,"generate_audio") as legacy:
            self.assertEqual(tts_generator.main(["fixture","--dry-run"]),0)
        legacy.assert_called_once_with(video_id="fixture",seconds=30,max_jobs=None,dry_run=True,model_id=tts_generator.DEFAULT_MODEL)

    def test_calibration_scoped_not_auto_applied(self):
        _,report=self.run_engine();profiles=report["calibration_profiles"]
        self.assertEqual(sorted(p["sample_count"] for p in profiles),[1,2])
        self.assertEqual({p["scope"]["speaker"] for p in profiles},{"A","B"})
        self.assertTrue(all(not p["auto_applied"] and not p["eligible_for_explicit_use"] for p in profiles))
        p=profiles[0]
        with self.assertRaisesRegex(ValueError,"Too few"):
            estimate_with_profile({"mora_count":10,"pauses":[],"warnings":[]},1,p,scope=p["scope"])
        wrong=copy.deepcopy(p["scope"]);wrong["speaker"]="C"
        with self.assertRaisesRegex(ValueError,"scope"):estimate_with_profile({},1,p,scope=wrong)

    def test_sufficient_scoped_samples_explicit_use(self):
        _,report=self.run_engine();samples=[]
        for i in range(3):
            e=copy.deepcopy(report["entries"][0]);e["id"]=str(i);e["audio_cache_key"]=str(i);samples.append(e)
        profiles=feedback_profiles(samples+samples);self.assertEqual(len(profiles),1);p=profiles[0]
        self.assertEqual(p["sample_count"],3)
        estimate=estimate_with_profile({"mora_count":10,"pauses":[{"kind":"sentence"}],"warnings":[]},1,p,scope=p["scope"])
        self.assertAlmostEqual(estimate["estimated_duration_seconds"],2.5)
        self.assertEqual(estimate["calibration_sample_count"],3)


if __name__ == "__main__": unittest.main()
