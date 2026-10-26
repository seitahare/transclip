import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from audio_mixer import (
    apply_boundary_audio_fade,
    build_ducking_envelope,
    build_plan,
    render_plan,
    verify_audio_loudness_and_clipping,
)
from dubbing_pipeline import (
    analyze_timeline_collisions_and_overlaps,
    detect_and_merge_split_utterances,
    run_phase13_1_dubbing_pipeline,
    run_phase13_1_new_video_e2e,
)
from timing_adjustment import (
    classify_english_words_and_pauses,
     compute_word_anchored_placement,
    concise_rewrite_for_timing_budget,
    detect_and_resolve_clip_boundaries,
    extract_or_resolve_word_timestamps,
    split_utterance_on_long_pauses_and_align_clauses,
)


def _write_test_wav(path: Path, duration_sec: float = 2.0, freq_hz: float = 220.0, amp: float = 0.35, sr: int = 24000):
    n = int(round(duration_sec * sr))
    t = np.arange(n, dtype=np.float32) / float(sr)
    sig = amp * np.sin(2.0 * np.pi * freq_hz * t)
    pad = min(int(sr * 0.04), n // 10)
    if pad > 0:
        sig[:pad] = 0.0
        sig[-pad:] = 0.0
    pcm = np.clip(np.round(sig * 32767.0), -32768, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


class TestPhase131WordLevelSync(unittest.TestCase):
    def test_filler_and_silence_classification_skips_leading_fillers(self):
        # Pattern 1: Leading fillers ("Um, uh, so") + 0.6s pause before content ("we built the system")
        words = [
            {"word": "Um,", "start": 1.00, "end": 1.22},
            {"word": "uh,", "start": 1.30, "end": 1.50},
            {"word": "so", "start": 1.55, "end": 1.75},
            {"word": "we", "start": 2.40, "end": 2.55},
            {"word": "built", "start": 2.55, "end": 2.85},
            {"word": "the", "start": 2.85, "end": 3.00},
            {"word": "system.", "start": 3.00, "end": 3.50},
        ]
        res = classify_english_words_and_pauses(words, utterance_start=1.00, utterance_end=3.50)
        self.assertAlmostEqual(res["effective_content_start"], 2.40, places=3)
        self.assertAlmostEqual(res["leading_filler_skip_seconds"], 1.40, places=3)
        self.assertEqual(len(res["leading_fillers"]), 3)
        self.assertEqual(res["content_words"][0]["word"], "we")

    def test_real_dataset_hesitation_starter_before_long_pause(self):
        # Pattern 2: Real XuoqKYxDHVc IDs 57-59 ("But what about" + 1.14s pause + "for physical jobs...")
        words = [
            {"word": "But", "start": 13.36, "end": 13.42},
            {"word": "what", "start": 13.42, "end": 13.68},
            {"word": "about", "start": 13.68, "end": 13.88},
            {"word": "for", "start": 15.02, "end": 15.32},
            {"word": "physical", "start": 15.32, "end": 15.72},
            {"word": "jobs,", "start": 15.72, "end": 16.20},
        ]
        res = classify_english_words_and_pauses(
            words,
            utterance_start=13.36,
            utterance_end=16.20,
            long_pause_threshold=0.55,
        )
        # "But what about" before a 1.14s pause is classified as hesitation_starter_before_long_pause
        self.assertAlmostEqual(res["effective_content_start"], 15.02, places=3)
        self.assertAlmostEqual(res["leading_filler_skip_seconds"], 1.66, places=3)
        self.assertEqual(len(res["long_pauses"]), 1)
        self.assertAlmostEqual(res["long_pauses"][0]["duration"], 1.14, places=2)

    def test_real_dataset_right_so_discourse_filler(self):
        # Pattern 3: Real XuoqKYxDHVc ID 60 ("Right, [0.30s] so that's why I said digital AI")
        words = [
            {"word": "Right,", "start": 24.86, "end": 24.94},
            {"word": "so", "start": 25.24, "end": 25.34},
            {"word": "that's", "start": 25.34, "end": 25.60},
            {"word": "why", "start": 25.60, "end": 25.74},
            {"word": "I", "start": 25.74, "end": 25.86},
            {"word": "said", "start": 25.86, "end": 26.06},
            {"word": "digital", "start": 26.06, "end": 26.42},
            {"word": "AI,", "start": 26.42, "end": 26.90},
        ]
        res = classify_english_words_and_pauses(words, utterance_start=24.86, utterance_end=26.90)
        self.assertAlmostEqual(res["effective_content_start"], 25.34, places=3)
        self.assertAlmostEqual(res["leading_filler_skip_seconds"], 0.48, places=3)

    def test_split_utterance_on_long_pauses_and_align_clauses(self):
        utt = {
            "utterance_id": 10,
            "speaker": "SPEAKER_01",
            "start": 5.0,
            "end": 12.0,
            "original_text": "We tried the first approach, and then we switched to robotics.",
            "translated": "最初のアプローチを試して、それからロボティクスに切り替えた。",
            "words": [
                {"word": "We", "start": 5.0, "end": 5.2},
                {"word": "tried", "start": 5.2, "end": 5.6},
                {"word": "the", "start": 5.6, "end": 5.8},
                {"word": "first", "start": 5.8, "end": 6.2},
                {"word": "approach,", "start": 6.2, "end": 6.8},
                # 1.4s internal pause between 6.8 and 8.2
                {"word": "and", "start": 8.2, "end": 8.4},
                {"word": "then", "start": 8.4, "end": 8.7},
                {"word": "we", "start": 8.7, "end": 8.9},
                {"word": "switched", "start": 8.9, "end": 9.5},
                {"word": "to", "start": 9.5, "end": 9.8},
                {"word": "robotics.", "start": 9.8, "end": 11.8},
            ],
        }
        res = split_utterance_on_long_pauses_and_align_clauses(utt, long_pause_threshold=0.55)
        clusters = res["word_timing_analysis"]["speech_clusters"]
        self.assertEqual(len(clusters), 2)
        self.assertAlmostEqual(clusters[0]["start"], 5.0, places=2)
        self.assertAlmostEqual(clusters[0]["end"], 6.8, places=2)
        self.assertAlmostEqual(clusters[1]["start"], 8.2, places=2)
        self.assertAlmostEqual(clusters[1]["end"], 11.8, places=2)
        units = res["timed_semantic_units"]
        self.assertGreaterEqual(len(units), 2)
        c0_units = [u for u in units if u["cluster_index"] == 0]
        c1_units = [u for u in units if u["cluster_index"] == 1]
        self.assertAlmostEqual(c0_units[0]["start"], 5.0, places=2)
        self.assertAlmostEqual(c0_units[-1]["end"], 6.8, places=2)
        self.assertAlmostEqual(c1_units[0]["start"], 8.2, places=2)
        self.assertAlmostEqual(c1_units[-1]["end"], 11.8, places=2)

    def test_merged_utterances_preserve_words_and_prevent_flying(self):
        raw_utts = [
            {
                "id": 1,
                "raw_id": 1,
                "numeric_id": 1,
                "speaker": "SPEAKER_01",
                "start": 2.0,
                "end": 2.5,
                "source_start": 2.0,
                "source_end": 2.5,
                "clip_relative_start": 2.0,
                "clip_relative_end": 2.5,
                "text_en": "Well, um,",
                "text_ja": "ええと、",
                "words": [
                    {"word": "Well,", "start": 2.0, "end": 2.2},
                    {"word": "um,", "start": 2.25, "end": 2.5},
                ],
            },
            {
                "id": 2,
                "raw_id": 2,
                "numeric_id": 2,
                "speaker": "SPEAKER_01",
                "start": 3.4,
                "end": 6.5,
                "source_start": 3.4,
                "source_end": 6.5,
                "clip_relative_start": 3.4,
                "clip_relative_end": 6.5,
                "text_en": "gardening is a good example.",
                "text_ja": "たぶん、ガーデニングがいい例になると思う。",
                "words": [
                    {"word": "gardening", "start": 3.4, "end": 4.1},
                    {"word": "is", "start": 4.1, "end": 4.3},
                    {"word": "a", "start": 4.3, "end": 4.4},
                    {"word": "good", "start": 4.4, "end": 4.8},
                    {"word": "example.", "start": 4.8, "end": 6.5},
                ],
            },
        ]
        merge_res = detect_and_merge_split_utterances(
            raw_utts,
            explicit_merge_groups=[[1, 2]],
            max_merge_gap_seconds=1.5,
        )
        merged = merge_res["utterances"]
        self.assertEqual(len(merged), 1)
        m = merged[0]
        self.assertEqual(len(m["words"]), 7)
        self.assertEqual(len(m["source_segment_spans"]), 2)
        self.assertAlmostEqual(m["start"], 2.0, places=2)
        # Substantive content starts at "gardening" (3.4s), NOT at "Well, um," (2.0s)!
        self.assertAlmostEqual(m["effective_content_start"], 3.4, places=2)

    def test_placement_respects_start_time_and_supports_end_anchor(self):
        # Previous utterance ended early at 2.1s, next utterance content starts at 4.5s.
        # Placement MUST NOT pull the next utterance earlier than 4.5s in content_start mode.
        p_start = compute_word_anchored_placement(
            raw_start=4.0,
            raw_end=7.5,
            effective_content_start=4.5,
            effective_content_end=7.4,
            tts_duration=2.2,
            prev_placed_end=2.1,
            next_barrier=9.0,
            sync_anchor="content_start",
        )
        self.assertGreaterEqual(p_start["placed_start"], 4.5)
        self.assertTrue(p_start["premature_start_prevented"])

        # Back-calculated end-anchor mode aligns end of short TTS with effective_content_end
        p_end = compute_word_anchored_placement(
            raw_start=4.0,
            raw_end=7.5,
            effective_content_start=4.5,
            effective_content_end=7.4,
            tts_duration=2.0,
            prev_placed_end=2.1,
            next_barrier=9.0,
            sync_anchor="end",
        )
        self.assertAlmostEqual(p_end["placed_end"], 7.4, places=2)
        self.assertAlmostEqual(p_end["placed_start"], 5.4, places=2)
        self.assertGreaterEqual(p_end["placed_start"], 4.5)

    def test_concise_rewrite_for_timing_budget(self):
        long_ja = "つまり、基本的には、そのおかげで一種の防壁になっていて、肉体労働は守られると思うということになります。"
        rewritten = concise_rewrite_for_timing_budget(long_ja, target_mora_budget=25)
        self.assertTrue(rewritten["changed"])
        self.assertLess(rewritten["rewritten_mora_count"], rewritten["original_mora_count"])
        self.assertNotIn("つまり、", rewritten["rewritten_text"])
        self.assertNotIn("ということになります", rewritten["rewritten_text"])


class TestPhase131DuckingStabilization(unittest.TestCase):
    def test_continuous_ducking_keeps_constant_18_percent_and_never_jumps_to_100(self):
        sr = 24000
        env, meta = build_ducking_envelope(
            total_samples=sr * 10,
            sample_rate=sr,
            speech_intervals=[(1.5, 4.0), (6.0, 8.5)],
            duck_gain=0.18,
            unspoken_gain=1.0,
            ducking_style="continuous",
        )
        self.assertAlmostEqual(float(np.min(env)), 0.18, places=4)
        self.assertAlmostEqual(float(np.max(env)), 0.18, places=4)
        self.assertTrue(meta["prevents_100pct_return"])

        # Verify loudness & clipping analyzer on continuous-ducked mix
        t = np.arange(sr * 6, dtype=np.float32) / float(sr)
        src_samples = 0.40 * np.sin(2.0 * np.pi * 140.0 * t)
        bg_samples = src_samples * env[: sr * 6]
        tts_samples = np.zeros_like(src_samples)
        tts_samples[int(1.5 * sr) : int(3.5 * sr)] = 0.50 * np.sin(2.0 * np.pi * 240.0 * t[: int(2.0 * sr)])
        mixed = bg_samples + tts_samples

        loud = verify_audio_loudness_and_clipping(
            mixed,
            japanese_samples=tts_samples,
            background_samples=bg_samples,
            source_samples=src_samples,
            sample_rate=sr,
            speech_intervals=[(1.5, 3.5)],
        )
        self.assertTrue(loud["continuous_ducking_verified"])
        self.assertEqual(loud["clipped_sample_count"], 0)
        self.assertLessEqual(loud["max_background_gain_jump_db"], 0.1)
        self.assertAlmostEqual(loud["measured_duck_ratio_during_speech"], 0.18, delta=0.02)
        self.assertAlmostEqual(loud["measured_duck_ratio_unspoken"], 0.18, delta=0.02)

    def test_long_release_ducking_extends_release_beyond_1_5_seconds(self):
        sr = 24000
        env, meta = build_ducking_envelope(
            total_samples=sr * 8,
            sample_rate=sr,
            speech_intervals=[(1.0, 3.0)],
            duck_gain=0.15,
            unspoken_gain=0.80,
            fade_out_seconds=0.20,
            fade_in_seconds=0.25,
            ducking_style="long-release",
            release_seconds=1.8,
        )
        self.assertGreaterEqual(meta["release_seconds"], 1.8)
        # At t=3.08s (end of post_duck), envelope is still at duck_gain (0.15)
        self.assertAlmostEqual(float(env[int(3.05 * sr)]), 0.15, places=2)
        # Halfway through 1.8s release (t=3.98s), envelope is near midpoint (~0.475), NOT at 0.80!
        mid_val = float(env[int(3.98 * sr)])
        self.assertGreater(mid_val, 0.25)
        self.assertLess(mid_val, 0.70)
        # After 1.8s release (t=5.1s), envelope reaches unspoken_gain (0.80)
        self.assertAlmostEqual(float(env[int(5.1 * sr)]), 0.80, places=2)


class TestPhase131ClipBoundaryHandling(unittest.TestCase):
    def test_detect_and_resolve_clip_boundaries_auto_extends_mid_sentence_cut(self):
        utts = [
            {"id": 60, "speaker": "SPEAKER_01", "start": 1860.86, "end": 1866.30, "text": "Right, so that's why I said digital AI."},
            {
                "id": 61,
                "speaker": "SPEAKER_01",
                "start": 1866.50,
                "end": 1874.30,
                "text": "Because of that, it acts as a barrier. I think our physical work is protected from AI, at least for a little bit longer.",
            },
        ]
        # 1871.0s cuts right through Utterance 61!
        res_extend = detect_and_resolve_clip_boundaries(
            utts,
            clip_start=1836.0,
            clip_end=1871.0,
            auto_extend_clip=True,
            max_extension_seconds=12.0,
            boundary_padding_seconds=0.35,
        )
        self.assertTrue(res_extend["end_boundary"]["cut_detected"])
        self.assertTrue(res_extend["clip_was_extended"])
        self.assertAlmostEqual(res_extend["effective_clip_end"], 1874.65, places=2)

        # When auto_extend_clip=False, records cut in report without extending
        res_no_extend = detect_and_resolve_clip_boundaries(
            utts,
            clip_start=1836.0,
            clip_end=1871.0,
            auto_extend_clip=False,
        )
        self.assertTrue(res_no_extend["end_boundary"]["cut_detected"])
        self.assertFalse(res_no_extend["clip_was_extended"])
        self.assertEqual(len(res_no_extend["untranslated_boundary_utterances"]), 1)
        self.assertEqual(res_no_extend["untranslated_boundary_utterances"][0]["utterance_id"], 61)

    def test_boundary_audio_fade_eliminates_abrupt_end_cutoff(self):
        sr = 24000
        samples = np.ones(sr * 2, dtype=np.float32) * 0.5
        faded = apply_boundary_audio_fade(samples, sr, fade_in_seconds=0.1, fade_out_seconds=0.4)
        self.assertAlmostEqual(float(faded[0]), 0.0, places=4)
        self.assertAlmostEqual(float(faded[-1]), 0.0, places=4)
        self.assertAlmostEqual(float(faded[sr]), 0.5, places=3)


class TestPhase131NewVideoE2E(unittest.TestCase):
    def test_run_phase13_1_new_video_e2e_all_13_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "phase13_1_e2e"
            report_path, report = run_phase13_1_new_video_e2e(
                input_dir="input/XuoqKYxDHVc",
                output_dir=out_dir,
                clip_start=1836.45,
                clip_end=1871.0,
                auto_extend_clip=True,
                duck_gain=0.18,
                ducking_style="continuous",
                mix_mode="continuous-voiceover",
            )
            self.assertTrue(report_path.exists())
            self.assertEqual(report["status"], "completed_diagnostic")
            self.assertEqual(report["completion"]["video_generation"]["status"], "succeeded")
            self.assertEqual(report["completion"]["utterance_coverage"]["status"], "complete")
            self.assertEqual(report["completion"]["utterance_coverage"]["missing_source_ids"], [])
            self.assertTrue((out_dir / "source.mp4").exists())
            self.assertTrue((out_dir / "final_ja_clone.mp4").exists())
            self.assertTrue((out_dir / "final_ja_preset.mp4").exists())
            self.assertTrue((out_dir / "reference.wav").exists())
            self.assertTrue((out_dir / "cloned_ja.wav").exists())
            self.assertTrue((out_dir / "preset_ja.wav").exists())
            self.assertTrue((out_dir / "pipeline_report.json").exists())

            # Verify all 13 stage JSONs exist
            stages_dir = out_dir / "stages"
            stage_files = sorted(p.name for p in stages_dir.glob("*.json"))
            self.assertEqual(len(stage_files), 13)

            # Verify clip auto-extension from 1871.0s to 1874.65s
            e2e_meta = report["new_video_e2e_verification"]
            self.assertEqual(e2e_meta["requested_clip_range_seconds"], [1836.45, 1871.0])
            self.assertAlmostEqual(e2e_meta["effective_clip_range_seconds"][1], 1874.65, places=2)

            # Verify word-level filler/pause skipping on IDs 57-59 and ID 60
            utts = {u["utterance_id"]: u for u in report["utterances"]}
            # ID 57 ("But what about" + 1.14s pause -> source content starts at 1851.02s, clip-relative 14.57s instead of 12.91s)
            self.assertAlmostEqual(utts[57]["source_effective_content_start"], 1851.02, places=2)
            self.assertAlmostEqual(utts[57]["effective_content_start"], 14.57, places=2)
            self.assertGreaterEqual(utts[57]["placed_start"], 14.57)
            # ID 60 ("Right, so" -> source content starts at 1861.34s, clip-relative 24.89s instead of 24.41s)
            self.assertAlmostEqual(utts[60]["source_effective_content_start"], 1861.34, places=2)
            self.assertAlmostEqual(utts[60]["effective_content_start"], 24.89, places=2)
            self.assertGreaterEqual(utts[60]["placed_start"], 24.89)

            # Verify continuous 18% ducking
            amd = report["audio_mixing_and_ducking"]
            self.assertEqual(amd["ducking_style"], "continuous")
            self.assertAlmostEqual(amd["duck_gain"], 0.18, places=2)
            self.assertTrue(amd["loudness_and_clipping"]["continuous_ducking_verified"])


if __name__ == "__main__":
    unittest.main()
