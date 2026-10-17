import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

import audio_mixer as mixer
from language.cache import atomic_json, fingerprint
from tts_feedback import run_feedback


def tone(path, seconds=.8, rate=8000, amplitude=20000):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(b"".join(struct.pack("<h", round(amplitude * math.sin(2*math.pi*200*i/rate)))
                                   for i in range(round(seconds*rate))))


class Backend:
    identity = {"backend": "fixture", "model": "sine", "settings": {"rate": 8000}}

    def synthesize(self, *, text, profile, output_path):
        tone(output_path)
        return {"termination": "fixture"}


def fixture_script():
    script = {"format_version": 1, "video_id": "fixture", "clip": {"start": 1, "end": 7},
              "preview": {"relative_start": 0, "relative_end": 4}, "entries": [], "excluded": []}
    for uid, speaker, start in [(1,"A",.5), (2,"B",.7), (3,"A",1), (4,None,2)]:
        script["entries"].append({"id": f"dub-{uid}", "speaker": speaker, "source_utterance_ids": [uid],
            "start": 1+start, "end": 1+start+.25, "relative_start": start, "relative_end": start+.25,
            "text_en": f"Example {uid}.", "text_ja": f"例{uid}です。", "source_duration_seconds": .25,
            "estimated_duration_seconds": .8, "pause_seconds": 0,
            "japanese_analysis": {"mora_count": 5, "reading": "レイ", "pauses": []},
            "review_required": speaker is None, "review_reasons": ["unknown_speaker"] if speaker is None else []})
    return script


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe not installed")
class MixerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.video = Path(cls.shared.name) / "source.mp4"
        mixer.ffmpeg(["-f", "lavfi", "-i", "color=c=blue:s=64x48:r=25:d=8", "-f", "lavfi", "-i",
                      "sine=frequency=440:sample_rate=48000:duration=8", "-c:v", "libx264", "-preset", "ultrafast",
                      "-c:a", "aac", "-shortest", cls.video])

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mp4"; shutil.copy2(self.video, self.source)
        self.selection = self.root / "selection.json"; atomic_json(self.selection, {"start": 1, "end": 7})
        self.info = self.root / "source.info.json"; atomic_json(self.info, {"id": "fixture"})
        self.script_path = self.root / "script.json"
        self.script = fixture_script()
        self.generate()

    def generate(self):
        atomic_json(self.script_path, self.script)
        with redirect_stdout(io.StringIO()):
            self.feedback = run_feedback(self.script_path, self.root / "tts", Backend(), {"A": "one", "B": "two"}, seconds=4)
        self.data = mixer.read_json(self.feedback)

    def save(self):
        atomic_json(self.feedback, self.data)

    def plan(self, **kwargs):
        return mixer.build_plan(self.feedback, self.selection, self.source, self.info, "fixture",
                                **{"seconds": 4, "diagnostic": True, **kwargs})

    def render(self, plan=None):
        path, reused = mixer.render_plan(plan or self.plan(), self.root / "mixes")
        return path, mixer.read_json(path), reused

    def test_absolute_relative_and_sample_positions(self):
        p = self.plan(); row = p["placements"][0]
        self.assertEqual((p["absolute_video_start"], row["placement_sample"], row["absolute_placement_seconds"]), (1,24000,1.5))
        self.assertAlmostEqual(row["actual_tts_duration_seconds"], .8)
        self.assertAlmostEqual(row["end_sample"]/48000,1.3)

    def test_same_speaker_collision_and_other_speaker_overlap(self):
        p = self.plan()
        self.assertEqual(p["same_speaker_conflicts"][0]["entry_ids"], ["dub-1", "dub-3"])
        self.assertAlmostEqual(p["same_speaker_conflicts"][0]["overlap_seconds"], .3)
        self.assertEqual(len(p["other_speaker_overlaps"]),2)
        self.assertEqual([r["relative_start"] for r in p["placements"]], [.5,.7,1])

    def test_unknown_preserved_and_incomplete_mode_blocked(self):
        p = self.plan(diagnostic=False)
        row = p["undubbed_entries"][0]
        self.assertIsNone(row["speaker"])
        self.assertEqual(row["text_en"], "Example 4.")
        self.assertIn("unknown_speaker", row["review_reasons"])
        self.assertIn("unresolved_same_speaker_collisions",p["blockers"])
        self.assertIn("undubbed_source_intervals",p["blockers"])
        with self.assertRaisesRegex(ValueError,"blocked"): self.render(p)
        self.assertFalse((self.root/"mixes").exists())

    def test_missing_and_corrupt_wav(self):
        wav = Path(self.data["entries"][0]["audio_file"])
        original = wav.read_bytes()
        for contents in (None, b"broken", original[:-50]):
            with self.subTest(contents="missing" if contents is None else len(contents)):
                if contents is None: wav.unlink()
                else: wav.write_bytes(contents)
                with self.assertRaisesRegex(ValueError,"dub-1"): self.plan()
                wav.write_bytes(original)

    def test_wrong_video_and_selection(self):
        for kind in ("video", "selection", "metadata"):
            with self.subTest(kind=kind):
                if kind=="video": self.data["video_id"]="wrong"; self.save()
                if kind=="selection": atomic_json(self.selection,{"start":2,"end":7})
                if kind=="metadata": atomic_json(self.info,{"id":"wrong"})
                with self.assertRaises(ValueError): self.plan()
                self.data["video_id"]="fixture"; self.save()
                atomic_json(self.selection,{"start":1,"end":7}); atomic_json(self.info,{"id":"fixture"})

    def test_audio_provenance_and_status_validation(self):
        original=copy.deepcopy(self.data)
        for key,value in [("actual_tts_duration_seconds",1.1),("audio_cache_key","f"*64),
                          ("relative_start",.6),("id","duplicate"),("generation_status","mystery"),
                          ("speaker","UNKNOWN"),("tts",{}),("voice","other")]:
            with self.subTest(key=key):
                self.data=copy.deepcopy(original); self.data["entries"][0][key]=value; self.save()
                with self.assertRaises(ValueError): self.plan()

    def test_source_script_and_receipt_validation(self):
        receipt=Path(self.data["entries"][0]["audio_file"]).with_name("receipt.json")
        content=receipt.read_bytes(); d=mixer.read_json(receipt);d["request"]["text_ja"]="別の台本"
        atomic_json(receipt,d)
        with self.assertRaisesRegex(ValueError,"receipt"): self.plan()
        receipt.write_bytes(content)
        self.script["entries"][0]["text_ja"]="変更";atomic_json(self.script_path,self.script)
        with self.assertRaisesRegex(ValueError,"fingerprint"): self.plan()

    def test_preview_end_stops_or_explicitly_extends(self):
        p=self.plan(seconds=1.1)
        self.assertTrue(p["boundary_crossing_ids"])
        with self.assertRaisesRegex(ValueError,"preview"): self.render(p)
        p=self.plan(seconds=1.1,extend_to_fit=True)
        self.assertAlmostEqual(p["duration_seconds"],1.8)
        self.assertFalse(p["blockers"])
        self.assertTrue(p["extended"])
        path,r,_=self.render(p)
        self.assertAlmostEqual(r["wav_measurements"][r["japanese_audio_file"]]["audio_duration"],1.8)

    def test_extension_never_exceeds_selection(self):
        self.script["clip"]["end"]=2.6
        self.script["preview"]["relative_end"]=1.6
        self.script["entries"]=self.script["entries"][:3]
        atomic_json(self.selection,{"start":1,"end":2.6});self.generate()
        p=self.plan(seconds=1.5,extend_to_fit=True)
        self.assertIn("audio_exceeds_selected_source_range",p["blockers"])
        with self.assertRaises(ValueError): self.render(p)

    def test_new_entries_inside_extension_require_new_plan(self):
        with self.assertRaisesRegex(ValueError,"additional entries"):
            self.plan(seconds=.6,extend_to_fit=True)

    def test_exact_boundary_accepted(self):
        p=self.plan(seconds=1.8)
        self.assertFalse(p["boundary_crossing_ids"])
        self.assertEqual(p["sample_frames"],86400)

    def test_recomputed_conflicts_ignore_stale_lists(self):
        for row in self.data["entries"]: row["actual_same_speaker_conflicts"]=[]
        self.save()
        self.assertEqual(len(self.plan()["same_speaker_conflicts"]),1)

    def test_render_silence_overlap_peak_and_streams(self):
        path,r,_=self.render(); folder=path.parent
        self.assertIn("diagnostic_conflicts_partial",r["video_file"])
        self.assertTrue(r["verification"]["full_decode_passed"])
        self.assertLess(r["verification"]["decoded_audio_peak"],1)
        with wave.open(str(folder/r["japanese_audio_file"]),"rb") as stream:
            self.assertEqual(stream.getnframes(),192000)
            self.assertEqual(set(stream.readframes(24000)),{0})
            self.assertTrue(any(stream.readframes(1000)))
            stream.setpos(90000); self.assertEqual(set(stream.readframes(1000)),{0})
        def values(name):
            with wave.open(str(folder/name),"rb") as stream:
                return struct.unpack("<"+"h"*(stream.getnframes()*2), stream.readframes(stream.getnframes()))
        a,b,m=values("speaker_A.wav"),values("speaker_B.wav"),values(r["japanese_audio_file"])
        self.assertTrue(any(b[70000:80000]))
        self.assertTrue(any(a[70000:80000]))
        self.assertLessEqual(max(abs(x+y-z) for x,y,z in zip(a,b,m)),1)
        for measure in r["wav_measurements"].values():
            self.assertEqual(measure["clipped_sample_fraction"],0)
            self.assertLessEqual(measure["peak_amplitude"],mixer.PEAK_TARGET+.0001)

    def test_static_gain_prevents_overlap_clipping(self):
        with patch.object(mixer,"speaker_audio",return_value=mixer.array("f",[.9]*192000*2)):
            _,r,_=self.render()
        self.assertAlmostEqual(r["gain"]["pre_gain_peak"],1.8,places=5)
        self.assertLess(r["gain"]["linear"],.5)
        self.assertEqual(r["verification"]["decoded_clipped_samples"],0)

    def test_reference_mix_keeps_quiet_source_audio(self):
        p=self.plan(mode="reference-mix")
        path,r,_=self.render(p)
        self.assertIn("no speech separation",r["source_audio_policy"])
        with wave.open(str(path.parent/r["mixed_audio_file"]),"rb") as stream:
            self.assertTrue(any(stream.readframes(1000)))
        with wave.open(str(path.parent/r["japanese_audio_file"]),"rb") as stream:
            self.assertFalse(any(stream.readframes(1000)))

    def test_identical_rerun_reuses_and_inputs_unchanged(self):
        inputs=[self.feedback,self.selection,self.script_path,self.source,self.info]+[Path(e["audio_file"]) for e in self.data["entries"] if e["audio_file"]]
        before={str(p):mixer.sha256_file(p) for p in inputs}
        path,r,_=self.render();mtime=(path.parent/r["video_file"]).stat().st_mtime_ns
        with patch.object(mixer,"speaker_audio",side_effect=AssertionError("must reuse")):
            again,_,reused=self.render()
        self.assertEqual(path,again);self.assertTrue(reused)
        self.assertEqual(mtime,(path.parent/r["video_file"]).stat().st_mtime_ns)
        self.assertEqual(before,{str(p):mixer.sha256_file(p) for p in inputs})

    def test_existing_corrupt_output_not_overwritten(self):
        path,r,_=self.render();video=path.parent/r["video_file"];video.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError,"corrupt"): self.render()
        self.assertEqual(video.read_bytes(),b"corrupt")

    def test_ffmpeg_failure_publishes_nothing_and_retry_works(self):
        plan=self.plan()
        with patch.object(mixer,"ffmpeg",side_effect=RuntimeError("FFmpeg injected failure")):
            with self.assertRaisesRegex(RuntimeError,"injected"): self.render(plan)
        self.assertEqual([p.name for p in (self.root/"mixes").iterdir()],[".mixer.lock"])
        _,r,_=self.render(plan);self.assertTrue(r["verification"]["full_decode_passed"])

    def test_cli_dry_run_has_no_outputs(self):
        args=["fixture","--feedback",str(self.feedback),"--selection",str(self.selection),
              "--source-video",str(self.source),"--source-info",str(self.info),"--seconds","4",
              "--output-dir",str(self.root/"mixes"),"--dry-run"]
        out=io.StringIO()
        with redirect_stdout(out): code=mixer.main(args)
        self.assertEqual(code,0);self.assertIn("UNDUBBED",out.getvalue());self.assertFalse((self.root/"mixes").exists())
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            self.assertEqual(mixer.main(args[:-1]),1)

    def test_bad_numbers_and_out_of_coverage(self):
        for kwargs in ({"seconds":float("nan")},{"seconds":0},{"seconds":7},{"reference_gain":1},{"mode":"unknown"}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): self.plan(**kwargs)

    def test_clean_normal_mode(self):
        self.script["entries"]=self.script["entries"][:1]
        self.script["entries"][0]["relative_end"]=1.3
        self.script["entries"][0]["end"]=2.3
        self.script["entries"][0]["source_duration_seconds"]=.8
        self.generate()
        p=self.plan(diagnostic=False)
        self.assertFalse(p["blockers"])
        _,r,_=self.render(p);self.assertEqual(r["status"],"rendered")

    def test_upstream_boundary_exclusion_reported(self):
        self.script["excluded"]=[{"source_utterance_id":5,"relative_start":3.5,"relative_end":5,"reason":"crosses_preview_boundary"}]
        self.generate();p=self.plan()
        self.assertEqual(p["excluded_intersecting_preview"][0]["source_utterance_id"],5)
        self.assertIn("partial",p["quality_label"])


if __name__ == "__main__": unittest.main()
