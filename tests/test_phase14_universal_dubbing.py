import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import wave

import numpy as np

import dubbing_pipeline
from phase14_universal import (
    PHASE14_PROVISIONAL_PRESET_VOICE,
    PHASE14_PROVISIONAL_SPEAKER_ID,
    QwenModelUnavailableError,
    adjust_overlong_script_entries,
    detect_unnatural_english_split_pair,
    load_phase14_video_project,
    propose_semantic_preserving_shortening,
    reconcile_utterance_and_word_timestamps_with_selection,
    resolve_phase14_splits_and_unverified_speakers,
    run_phase14_universal_dubbing,
    validate_shortened_candidate,
)


def _create_precut_clip_mp4(mp4_path: Path, duration_seconds: float = 180.0) -> Path:
    """Create a real 0..180s pre-cut MP4 clip on disk (representing `output/wJasRLuu_LA/clip_300_480.mp4`)."""
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    sr = 24000
    n_total = int(round(duration_seconds * sr))
    t = np.arange(n_total, dtype=np.float32) / float(sr)
    audio = (0.04 * np.sin(2.0 * np.pi * 160.0 * t)).astype(np.float32)
    wav_tmp = mp4_path.with_suffix(".tmp.wav")
    pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(wav_tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=0x18222C:s=640x360:r=24:d={duration_seconds:.3f}",
            "-i", str(wav_tmp),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest",
            str(mp4_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if wav_tmp.exists():
        wav_tmp.unlink()
    return mp4_path


def _build_wjas_like_fixture(root_dir: Path, video_id: str = "wJasRLuu_LA", *, create_mp4: bool = False) -> dict:
    """Create a realistic 3-minute (300..480s) 55-utterance / 42-entry / 41-review / 33-overlong dataset."""
    in_dir = root_dir / "input" / video_id
    out_vid_dir = root_dir / "output" / video_id
    in_dir.mkdir(parents=True, exist_ok=True)
    out_vid_dir.mkdir(parents=True, exist_ok=True)

    selection = {"video_id": video_id, "start": 300.0, "end": 480.0, "duration": 180.0}
    (in_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (in_dir / "source.info.json").write_text(json.dumps({"id": video_id, "duration": 920.0}, indent=2), encoding="utf-8")

    utterances = []
    translations = []
    words = []
    cur_t = 300.2
    for idx in range(1, 56):
        dur = 2.6 if idx % 3 else 1.8
        u_start = round(cur_t, 2)
        u_end = round(min(479.8, u_start + dur), 2)
        cur_t = round(u_end + 0.42, 2)
        if idx in (10, 11):
            spk = "SPEAKER_00"
            en_txt = "I'm not telling" if idx == 10 else "you."
            ja_txt = "私は絶対に伝えません" if idx == 10 else "あなたには。"
        elif idx in (20, 21):
            spk = "SPEAKER_00" if idx == 20 else "UNKNOWN"
            en_txt = "I'm not telling" if idx == 20 else "you that secret."
            ja_txt = "それは言えません" if idx == 20 else "あなたにその秘密は。"
        elif idx % 9 == 0:
            spk = "UNKNOWN"
            en_txt = f"Unverified comment number {idx} in this segment."
            ja_txt = f"えっと、この区間の第{idx}番目の未確認コメントということになります。"
        else:
            spk = "SPEAKER_00" if idx % 2 == 1 else "SPEAKER_01"
            en_txt = f"Statement {idx} about OpenAI model 2026 performance."
            ja_txt = f"つまりですね、OpenAIの2026年モデル第{idx}項目は非常に重要なポイントとして説明することができます。"

        u_words = [
            {"word": w, "start": round(u_start + i * 0.25, 3), "end": round(u_start + (i + 1) * 0.22, 3)}
            for i, w in enumerate(en_txt.split())
        ]
        words.extend(u_words)
        utterances.append({"id": idx, "start": u_start, "end": u_end, "speaker": spk, "text": en_txt, "words": u_words})
        translations.append(
            {
                "id": idx,
                "start": u_start,
                "end": u_end,
                "relative_start": round(u_start - 300.0, 2),
                "relative_end": round(u_end - 300.0, 2),
                "speaker": spk,
                "text": en_txt,
                "text_ja": ja_txt,
                "review_required": idx != 1,
            }
        )

    entries = []
    for e_idx in range(1, 43):
        u_src = translations[e_idx - 1]
        is_overlong = e_idx <= 33
        entries.append(
            {
                "id": f"entry_{e_idx:03d}",
                "utterance_ids": [u_src["id"]],
                "speaker": u_src["speaker"],
                "start": u_src["start"],
                "end": u_src["end"],
                "relative_start": u_src["relative_start"],
                "relative_end": u_src["relative_end"],
                "text_en": u_src["text"],
                "text_ja": u_src["text_ja"],
                "review_required": e_idx <= 41,
                "overlong": is_overlong,
                "status": "overlong" if is_overlong else "ok",
            }
        )

    # Make the last entry near the 480.0s boundary so we can also test post-TTS boundary extension
    entries[-1]["start"] = 477.5
    entries[-1]["end"] = 479.9
    entries[-1]["relative_start"] = 177.5
    entries[-1]["relative_end"] = 179.9

    (in_dir / "utterances.json").write_text(
        json.dumps({"video_id": video_id, "utterances": utterances}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (in_dir / "utterance_translations.json").write_text(
        json.dumps(
            {"video_id": video_id, "clip": {"start": 300.0, "end": 480.0}, "utterances": translations},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (in_dir / "word_timestamps.json").write_text(
        json.dumps({"video_id": video_id, "words": words}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (in_dir / "dubbing_script.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "clip": {"start": 300.0, "end": 480.0, "duration": 180.0},
                "entries": entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    clip_mp4 = out_vid_dir / "clip_300_480.mp4"
    if create_mp4:
        _create_precut_clip_mp4(clip_mp4, duration_seconds=180.0)
    return {"root_dir": root_dir, "in_dir": in_dir, "out_vid_dir": out_vid_dir, "clip_mp4": clip_mp4}


class TestPhase14UniversalDubbing(unittest.TestCase):
    def test_cli_registration_and_help(self):
        """1. Verify `run-phase14` is registered on `args.command` in `dubbing_pipeline.py` and `--help` works."""
        repo_root = Path(__file__).resolve().parent.parent
        res_top = subprocess.run(
            [sys.executable, str(repo_root / "dubbing_pipeline.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_top.returncode, 0, msg=res_top.stderr)
        self.assertIn("run-phase14", res_top.stdout)

        res_sub = subprocess.run(
            [sys.executable, str(repo_root / "dubbing_pipeline.py"), "run-phase14", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_sub.returncode, 0, msg=res_sub.stderr)
        self.assertIn("--video-id", res_sub.stdout)
        self.assertIn("--voice-mode", res_sub.stdout)
        self.assertIn("--test-synthetic-tts", res_sub.stdout)

    def test_missing_qwen_model_raises_error_without_silent_fallback(self):
        """2. Verify that when Qwen3-TTS-MLX is unavailable, default production mode stops with an error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_wjas_like_fixture(root, video_id="wJasRLuu_LA", create_mp4=True)
            out_dir = root / "output" / "phase14_fail_check"

            with mock.patch(
                "phase14_universal.verify_qwen3_tts_mlx_available",
                return_value={"available": False, "backend_identity": None, "error": "ModuleNotFoundError: No module named 'qwen_tts_mlx'"},
            ):
                with self.assertRaises(QwenModelUnavailableError) as ctx:
                    run_phase14_universal_dubbing(
                        video_id="wJasRLuu_LA",
                        output_dir=out_dir,
                        voice_mode="preset",
                        test_synthetic_tts=False,
                        base_dir=root,
                    )
                self.assertIn("Automatic fallback to synthetic audio is disabled", str(ctx.exception))

    def test_universal_loader_and_300_480_vs_0_180_timeline_reconciliation(self):
        """3. Verify 300..480s source selection and 0..180s pre-cut clip timeline reconciliation."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_wjas_like_fixture(root, video_id="wJasRLuu_LA", create_mp4=False)
            proj = load_phase14_video_project(video_id="wJasRLuu_LA", base_dir=root)
            self.assertEqual(proj["video_id"], "wJasRLuu_LA")
            self.assertAlmostEqual(proj["selection_start"], 300.0)
            self.assertAlmostEqual(proj["selection_end"], 480.0)
            self.assertAlmostEqual(proj["clip_duration"], 180.0)
            self.assertEqual(proj["raw_utterance_count"], 55)
            self.assertEqual(proj["translation_utterance_count"], 55)
            self.assertEqual(proj["script_entry_count"], 42)

            # Also verify that if an input JSON stores clip-relative 0..180 timestamps,
            # reconcile_utterance_and_word_timestamps_with_selection maps them consistently
            rel_entries = [
                {"id": "e1", "speaker": "SPEAKER_00", "start": 2.5, "end": 5.0, "text_en": "Hello", "text_ja": "こんにちは"},
                {"id": "e2", "speaker": "SPEAKER_01", "start": 175.0, "end": 179.5, "text_en": "Bye", "text_ja": "さようなら"},
            ]
            rel_words = [{"word": "Hello", "start": 2.5, "end": 3.2}]
            rec = reconcile_utterance_and_word_timestamps_with_selection(
                rel_entries, rel_words, selection_start=300.0, selection_end=480.0
            )
            self.assertAlmostEqual(rec["entries"][0]["start"], 302.5)
            self.assertAlmostEqual(rec["entries"][0]["clip_relative_start"], 2.5)
            self.assertAlmostEqual(rec["entries"][1]["start"], 475.0)
            self.assertAlmostEqual(rec["entries"][1]["clip_relative_end"], 179.5)
            self.assertAlmostEqual(rec["words"][0]["start"], 302.5)
            self.assertAlmostEqual(rec["words"][0]["clip_relative_start"], 2.5)

    def test_split_utterance_detection_and_unknown_speaker_safety(self):
        """Verify split detection ('I'm not telling' + 'you.') and that UNKNOWN is never turned into a confirmed speaker."""
        u1 = {"id": "e1", "speaker": "SPEAKER_00", "start": 310.0, "end": 311.2, "text_en": "I'm not telling", "text_ja": "私は言いません"}
        u2 = {"id": "e2", "speaker": "SPEAKER_00", "start": 311.3, "end": 311.8, "text_en": "you.", "text_ja": "あなたには。"}
        u3 = {"id": "e3", "speaker": "SPEAKER_01", "start": 320.0, "end": 321.1, "text_en": "I'm not telling", "text_ja": "伝えません"}
        u4 = {"id": "e4", "speaker": "UNKNOWN", "start": 321.2, "end": 321.9, "text_en": "you.", "text_ja": "君には。"}

        res = resolve_phase14_splits_and_unverified_speakers([u1, u2, u3, u4])
        self.assertEqual(len(res["merged_groups"]), 1)
        self.assertEqual(res["merged_groups"][0]["merged_from_ids"], ["e1", "e2"])
        self.assertEqual(len(res["blocked_unverified_splits"]), 1)
        self.assertEqual(res["blocked_unverified_splits"][0]["decision"], "hold_unverified_speaker_review_required")
        self.assertEqual(len(res["unverified_speaker_targets"]), 1)
        self.assertEqual(res["unverified_speaker_targets"][0]["id"], "e4")
        # Confirm e4 speaker is None (unassigned), never SPEAKER_00 or SPEAKER_01
        e4_after = [x for x in res["entries"] if x["id"] == "e4"][0]
        self.assertIsNone(e4_after["speaker"])
        self.assertTrue(e4_after["review_required"])

    def test_shortening_records_unverified_semantic_items_in_report(self):
        """Verify shortened candidates validate surface tokens AND record semantic_review_required in the report."""
        orig = "えっと、つまりですね、OpenAIの2026年モデルは33%高速ではないでしょうか？"
        short_res = propose_semantic_preserving_shortening(orig, text_en="Isn't OpenAI's 2026 model 33% faster?", target_mora=20)
        self.assertTrue(short_res["applied"])
        self.assertTrue(short_res["semantic_review_required"])
        self.assertFalse(short_res["semantic_validation"]["mechanically_guaranteed"])
        self.assertIn("2026", short_res["shortened_text_ja"])
        self.assertIn("33%", short_res["shortened_text_ja"])
        self.assertIn("OpenAI", short_res["shortened_text_ja"])

        bad_val = validate_shortened_candidate("これは33%ではありません。", "これは33%です。")
        self.assertFalse(bad_val["valid"])
        self.assertIn("negation_polarity_changed", bad_val["reasons"])

    def test_end_to_end_cli_execution_with_precut_clip_and_boundary_extension(self):
        """Verify E2E CLI execution on a 0..180s pre-cut clip for 300..480s selection, including clip end extension."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = _build_wjas_like_fixture(root, video_id="wJasRLuu_LA", create_mp4=True)
            out_dir = root / "output" / "phase14_new_video_01"

            # Force the final entry (477.5..479.9s -> clip-relative 177.5..179.9s) to have a long text
            # so its WAV exceeds 180.0s and triggers post-TTS clip extension on the 0..180s pre-cut MP4!
            script_doc = json.loads((fx["in_dir"] / "dubbing_script.json").read_text(encoding="utf-8"))
            script_doc["entries"][-1]["speaker"] = "SPEAKER_00"
            script_doc["entries"][-1]["text_ja"] = (
                "最後のまとめとして、2026年の新しい音声合成システムでは、"
                "すべての発話が途切れることなく最後まで自然に再生されます。"
            )
            (fx["in_dir"] / "dubbing_script.json").write_text(
                json.dumps(script_doc, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            rc = dubbing_pipeline.main(
                [
                    "run-phase14",
                    "--video-id", "wJasRLuu_LA",
                    "--input-dir", str(fx["in_dir"]),
                    "--video", str(fx["clip_mp4"]),
                    "--voice-mode", "preset",
                    "--test-synthetic-tts",
                    "--output", str(out_dir),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "final_ja.mp4").is_file())
            self.assertTrue((out_dir / "pipeline_report.json").is_file())
            self.assertTrue((out_dir / "script_adjustment_report.json").is_file())
            self.assertTrue((out_dir / "speaker_verification_targets.json").is_file())

            report = json.loads((out_dir / "pipeline_report.json").read_text(encoding="utf-8"))
            summary = report["phase14_universal_summary"]
            self.assertEqual(summary["video_id"], "wJasRLuu_LA")
            self.assertEqual(summary["tts_execution_mode"], "explicit_test_synthetic_tts")
            self.assertFalse(summary["is_real_qwen_model_generation"])
            self.assertTrue(summary["is_test_synthetic_generation"])
            self.assertFalse(summary["unknown_converted_to_confirmed_speaker"])
            self.assertTrue(summary["timeline_consistency_passed"])
            self.assertTrue(summary["input_files_unchanged"])

            # Verify timeline consistency: pre-cut clip was 180.0s, extended cleanly to >= 180.0s without offset drift
            t_ver = report["timeline_consistency_verification"]
            self.assertTrue(t_ver["input_clip_is_precut_0_to_duration"])
            self.assertTrue(t_ver["all_placements_mapped_to_clip_relative_0_to_duration"])
            self.assertTrue(t_ver["duration_matches_effective_clip"])
            self.assertGreaterEqual(t_ver["effective_clip_duration_seconds"], 180.0)

            # Verify unverified items (UNKNOWN speakers + semantic shortening) are explicitly recorded in report
            unver = report["unverified_items_requiring_human_review"]
            self.assertGreater(len(unver["unverified_speaker_targets"]), 0)
            self.assertGreater(len(unver["unverified_semantic_shortening_items"]), 0)

            # Verify subtitles are within [0.0, effective_clip_duration] and never offset by +300s
            srt_text = (out_dir / "final_ja.srt").read_text(encoding="utf-8")
            self.assertIn("00:00:00,", srt_text)
            self.assertNotIn("00:05:", srt_text.splitlines()[1])


if __name__ == "__main__":
    unittest.main()
