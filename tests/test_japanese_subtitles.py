import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import audio_mixer as mixer
from japanese_subtitles import (
    SubtitleStyleConfig,
    build_subtitle_plan,
    export_subtitles,
    format_ass_timestamp,
    format_srt_timestamp,
    render_ass,
    render_srt,
    split_japanese_clauses,
    wrap_clause_units,
)
from language.cache import atomic_json
from test_audio_mixer import Backend, fixture_script, tone
from tts_feedback import run_feedback


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe required")
class JapaneseSubtitleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.video = Path(cls.shared.name) / "source.mp4"
        mixer.ffmpeg([
            "-f", "lavfi", "-i", "color=c=navy:s=320x180:r=25:d=8",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=8",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-shortest", cls.video,
        ])

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mp4"
        shutil.copy2(self.video, self.source)
        self.selection = self.root / "selection.json"
        atomic_json(self.selection, {"start": 1, "end": 7})
        self.info = self.root / "source.info.json"
        atomic_json(self.info, {"id": "fixture", "duration": 8.0})
        self.script_path = self.root / "script.json"
        self.script = fixture_script()
        # Make dub-1 and dub-2 overlap across different speakers (A and B), and dub-3 non-colliding for speaker A
        self.script["entries"] = [
            {
                "id": "dub-1",
                "speaker": "A",
                "source_utterance_ids": [1],
                "start": 1.2,
                "end": 2.2,
                "relative_start": 0.2,
                "relative_end": 1.2,
                "text_en": "New technology is gradually changing our daily lives.",
                "original_text_ja": "旧翻訳テキスト（TTSには未使用）",
                "text_ja": "新しい技術が、私たちの暮らしを少しずつ変えています。",
                "source_duration_seconds": 1.0,
                "estimated_duration_seconds": 0.8,
                "pause_seconds": 0.12,
                "japanese_analysis": {
                    "mora_count": 24,
                    "reading": "アタラシイギジュツガワタシタチノクラシヲスコシズツカエテイマス",
                    "pauses": [{"kind": "comma"}, {"kind": "sentence"}],
                    "tokens": [
                        {"surface": "新しい", "pos": ["形容詞"]},
                        {"surface": "技術", "pos": ["名詞"]},
                        {"surface": "が", "pos": ["助詞"]},
                        {"surface": "、", "pos": ["補助記号"]},
                        {"surface": "私たち", "pos": ["代名詞"]},
                        {"surface": "の", "pos": ["助詞"]},
                        {"surface": "暮らし", "pos": ["名詞"]},
                        {"surface": "を", "pos": ["助詞"]},
                        {"surface": "少しずつ", "pos": ["副詞"]},
                        {"surface": "変えています", "pos": ["動詞"]},
                        {"surface": "。", "pos": ["補助記号"]},
                    ],
                },
                "review_required": False,
                "review_reasons": [],
            },
            {
                "id": "dub-2",
                "speaker": "B",
                "source_utterance_ids": [2],
                "start": 1.5,
                "end": 2.4,
                "relative_start": 0.5,
                "relative_end": 1.4,
                "text_en": "Right, indeed.",
                "original_text_ja": "そうですね。",
                "text_ja": "本当にそうですね。",
                "source_duration_seconds": 0.9,
                "estimated_duration_seconds": 0.8,
                "pause_seconds": 0.0,
                "japanese_analysis": {"mora_count": 9, "reading": "ホントウニソウデスネ", "pauses": []},
                "review_required": False,
                "review_reasons": [],
            },
            {
                "id": "dub-3",
                "speaker": "A",
                "source_utterance_ids": [3],
                "start": 2.6,
                "end": 3.6,
                "relative_start": 1.6,
                "relative_end": 2.6,
                "text_en": "This computer performs 144 trillion calculations per second.",
                "original_text_ja": "旧翻訳144兆回",
                "text_ja": "このコンピューターは、1秒間に144兆回の計算を行います。",
                "source_duration_seconds": 1.0,
                "estimated_duration_seconds": 0.8,
                "pause_seconds": 0.12,
                "japanese_analysis": {"mora_count": 26, "reading": "コノコンピューターハイチビョウカンニヒャクヨンジュウヨンチョウカイノケイサンヲオコナイマス", "pauses": [{"kind": "comma"}, {"kind": "sentence"}]},
                "review_required": False,
                "review_reasons": [],
            },
        ]
        atomic_json(self.script_path, self.script)
        with redirect_stdout(io.StringIO()):
            self.feedback = run_feedback(
                self.script_path, self.root / "tts", Backend(), {"A": "Ryan", "B": "Aiden"}, seconds=4.0
            )

    def test_timestamps_and_clause_wrapping(self):
        self.assertEqual(format_srt_timestamp(61.234), "00:01:01,234")
        self.assertEqual(format_ass_timestamp(61.234), "0:01:01.23")
        units = split_japanese_clauses("このコンピューターは、1秒間に144兆回の計算を行います。")
        self.assertEqual("".join(units), "このコンピューターは、1秒間に144兆回の計算を行います。")
        cues = wrap_clause_units(units, max_chars_per_line=16, max_lines=2)
        self.assertEqual(len(cues), 1)
        self.assertLessEqual(len(cues[0]["lines"]), 2)
        self.assertIn("144兆回", cues[0]["text"])

    def test_uses_final_tts_script_not_pre_tts_translation_and_handles_multi_speaker_overlap(self):
        plan = build_subtitle_plan(
            self.feedback, self.selection, "fixture", seconds=4.0, style=SubtitleStyleConfig(max_chars_per_line=15, max_lines=2)
        )
        self.assertEqual(plan["blockers"], [])
        texts = [c["text_ja"] for c in plan["cues"]]
        self.assertNotIn("旧翻訳テキスト（TTSには未使用）", "".join(texts))
        self.assertIn("新しい技術が、私たちの暮らしを少しずつ変えています。", "".join(texts))
        # Verify simultaneous overlap between speaker A (0.2..1.0) and speaker B (0.5..1.3) gets distinct ASS lanes
        overlapping = [c for c in plan["cues"] if c["simultaneous_overlap"]]
        self.assertGreaterEqual(len(overlapping), 2)
        lanes = {c["speaker"]: c["lane"] for c in overlapping}
        self.assertNotEqual(lanes["A"], lanes["B"])

        srt_text = render_srt(plan)
        ass_text = render_ass(plan)
        self.assertIn("[A]", srt_text)
        self.assertIn("[B]", srt_text)
        self.assertIn("Style: Speaker_A", ass_text)
        self.assertIn("Style: Speaker_B", ass_text)

    def test_alignment_verification_flags_asr_discrepancy_without_corrupting_script(self):
        alignment = {
            "dub-3": {
                "recognized_text": "このコンピューターは、1秒間に144調解の計算を行います。",
                "segments": [{"relative_start": 1.62, "relative_end": 2.38}],
            }
        }
        plan = build_subtitle_plan(
            self.feedback, self.selection, "fixture", seconds=4.0, alignment_data=alignment
        )
        self.assertIn("dub-3", plan["alignment_warnings"])
        self.assertIn("asr_transcript_differs_from_tts_script", plan["alignment_warnings"]["dub-3"])
        self.assertTrue(plan["review_required"])
        cue3 = next(c for c in plan["cues"] if c["entry_id"] == "dub-3")
        # Authoritative TTS script text ("144兆回") is preserved, while aligned timestamps are applied
        self.assertIn("144兆回", cue3["text_ja"])
        self.assertAlmostEqual(cue3["start"], 1.62, places=2)
        self.assertAlmostEqual(cue3["end"], 2.38, places=2)

    def test_export_srt_ass_and_burned_mp4_non_destructively(self):
        mix_plan = mixer.build_plan(
            self.feedback, self.selection, self.source, self.info, "fixture", seconds=4.0, diagnostic=True
        )
        mix_report_path, _ = mixer.render_plan(mix_plan, self.root / "mixes")
        mix_report = mixer.read_json(mix_report_path)
        dubbed_mp4 = mix_report_path.parent / mix_report["video_file"]

        sub_plan = build_subtitle_plan(mix_report_path, self.selection, "fixture", seconds=4.0)
        report_path, reused = export_subtitles(sub_plan, self.root / "subtitles", dubbed_video_path=dubbed_mp4)
        self.assertFalse(reused)
        out_dir = report_path.parent
        self.assertTrue((out_dir / "final_ja.srt").is_file())
        self.assertTrue((out_dir / "final_ja.ass").is_file())
        self.assertTrue((out_dir / "final_ja_no_subtitles.mp4").is_file())
        self.assertTrue((out_dir / "final_ja.mp4").is_file())

        report_path_2, reused_2 = export_subtitles(sub_plan, self.root / "subtitles", dubbed_video_path=dubbed_mp4)
        self.assertEqual(report_path, report_path_2)
        self.assertTrue(reused_2)


if __name__ == "__main__":
    unittest.main()
