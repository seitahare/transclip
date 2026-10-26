"""Phase 13.3 Regression Test Suite (`tests/test_phase13_3_improvements.py`).

Covers all required Phase 13.3 improvements and non-regression invariants:
1. Subtitle split timing synchronization to actual Japanese TTS audio (Method B WAV acoustic boundary + Kanji-aware phonetic mora & Method A chunked TTS)
2. Long sentence subtitle splitting preserves script integrity, <= 2 lines, and natural bunsetsu breaks
3. Inter-utterance ducking hold control (same-speaker continuous hold & <= 3-4s gap hold, suppressing 0:12-0:14 English prominence)
4. Multi-speaker ITU-R BS.1770-4 LUFS loudness normalization (-16 LUFS / -18 LUFS, FFmpeg loudnorm & BS.1770 linear modes, clipping prevention)
5. Full backward compatibility with Phase 13.2 speaker consistency, natural speaking rate, and clip boundary auto-extension.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audio_mixer import (
    build_ducking_envelope,
    build_lufs_normalization_report,
    normalize_tts_wav_lufs,
)
from dubbing_pipeline import (
    _fallback_generate_subtitles_and_mp4,
    _fallback_mix_audio_and_video,
    _synthesize_mora_conditioned_speech_wav,
    ensure_30s_verification_video,
)
from japanese_subtitles import (
    SubtitleStyleConfig,
    align_entry_cues,
    build_subtitle_sync_report,
    check_subtitle_overflow,
    split_japanese_clauses,
    synthesize_or_measure_chunked_subtitle_cues,
    wrap_clause_units,
)


class TestPhase133Improvements(unittest.TestCase):
    def test_01_subtitle_split_sync_matches_tts_clause_boundary_not_early_kana_only(self):
        """1. 字幕の分割開始位置が音声の読み上げ境界（0:19.8付近）と一致し、0:18.5のように早く切り替わらないこと."""
        text_57 = "でも肉体労働や組み立て作業では人型ロボットの能力も必要になりますが、まだそこまでは至っていませんよね？"
        style = SubtitleStyleConfig.for_video_resolution(1920, 1080)

        with tempfile.TemporaryDirectory() as tmp:
            wav_57 = Path(tmp) / "57.wav"
            _synthesize_mora_conditioned_speech_wav(
                wav_57,
                text_57,
                target_duration_seconds=6.80,
                f0_hz=195.0,
                seed=202,
            )
            entry = {
                "id": "57",
                "speaker": "SPEAKER_00",
                "text_ja": text_57,
                "placement_seconds": 15.00,
                "actual_tts_duration_seconds": 6.80,
                "audio_file": str(wav_57),
                "generation_status": "complete",
            }
            cues, _ = align_entry_cues(entry, style)
            self.assertEqual(len(cues), 2)
            second_cue_start = cues[1]["start"]
            self.assertGreaterEqual(second_cue_start, 19.45)
            self.assertLessEqual(second_cue_start, 20.45)
            self.assertEqual(cues[1]["timing_source"], "tts_wav_acoustic_boundary")

    def test_02_method_a_chunked_tts_and_method_b_wav_boundary_maintain_integrity_and_2_lines(self):
        """2. 長文を分割しても字幕とTTSの整合性・最大2行・文節改行が維持されること（方法A・方法B両方検証）."""
        text_61 = "ロボットはいわばエンドエフェクターで、局所的な知能を持ちつつ大規模AIモデルに管理される形になる。"
        style = SubtitleStyleConfig.for_video_resolution(1920, 1080)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            units = split_japanese_clauses(text_61)
            specs = wrap_clause_units(units, max_chars_per_line=style.max_chars_per_line, max_lines=style.max_lines)
            self.assertEqual(len(specs), 2)

            c0_wav = tmp_dir / "c0.wav"
            c1_wav = tmp_dir / "c1.wav"
            concat_wav = tmp_dir / "61_concat.wav"
            _synthesize_mora_conditioned_speech_wav(c0_wav, specs[0]["text"], target_duration_seconds=2.45, f0_hz=120.0)
            _synthesize_mora_conditioned_speech_wav(c1_wav, specs[1]["text"], target_duration_seconds=5.55, f0_hz=120.0)

            res_a = synthesize_or_measure_chunked_subtitle_cues(
                {"id": "61", "text_ja": text_61},
                style,
                chunk_wav_paths=[c0_wav, c1_wav],
                inter_chunk_pause_seconds=0.16,
                output_concat_wav_path=concat_wav,
            )
            self.assertEqual(res_a["chunk_count"], 2)
            self.assertAlmostEqual(res_a["cue_boundary_offsets_seconds"][0], 2.61, places=2)

            entry_a = {
                "id": "61",
                "speaker": "SPEAKER_01",
                "text_ja": text_61,
                "placement_seconds": 30.05,
                "actual_tts_duration_seconds": res_a["total_duration_seconds"],
                "cue_boundary_offsets_seconds": res_a["cue_boundary_offsets_seconds"],
                "audio_file": str(concat_wav),
                "generation_status": "complete",
            }
            cues_a, _ = align_entry_cues(entry_a, style)
            self.assertEqual(len(cues_a), 2)
            self.assertEqual(cues_a[1]["timing_source"], "chunked_tts_exact")
            self.assertAlmostEqual(cues_a[1]["start"], 30.05 + 2.61, places=2)

            wav_61 = tmp_dir / "61_single.wav"
            _synthesize_mora_conditioned_speech_wav(wav_61, text_61, target_duration_seconds=8.16, f0_hz=120.0)
            entry_b = {
                "id": "61",
                "speaker": "SPEAKER_01",
                "text_ja": text_61,
                "placement_seconds": 30.05,
                "actual_tts_duration_seconds": 8.16,
                "audio_file": str(wav_61),
                "generation_status": "complete",
            }
            cues_b, _ = align_entry_cues(entry_b, style)
            self.assertEqual(len(cues_b), 2)
            self.assertEqual("".join(c["text_ja"] for c in cues_b), text_61)
            overflow = check_subtitle_overflow(cues_b, style)
            self.assertEqual(overflow["overflow_count"], 0)
            self.assertEqual(overflow["unnatural_break_count"], 0)
            self.assertLessEqual(overflow["max_line_count_observed"], 2)

            sync_rep = build_subtitle_sync_report([entry_b], style=style)
            self.assertEqual(sync_rep["split_utterance_count"], 1)
            self.assertTrue(sync_rep["all_split_cues_synchronized"])

    def test_03_ducking_hold_control_bridges_inter_utterance_and_same_speaker_gaps(self):
        """3. 発話間のダッキングホールド（初期値3秒程度・同一話者連続発話維持）が正しく機能し、0:12〜0:14の英語突出を抑えること."""
        sr = 24000
        total_sec = 25.0
        total_frames = int(total_sec * sr)
        speech_intervals = [
            {"id": "54", "speaker": "SPEAKER_01", "start": 7.43, "end": 9.12},
            {"id": "56", "speaker": "SPEAKER_00", "start": 10.45, "end": 11.35},
            {"id": "57", "speaker": "SPEAKER_00", "start": 14.57, "end": 21.37},
        ]

        env_no_hold, _ = build_ducking_envelope(
            total_frames,
            sample_rate=sr,
            speech_intervals=[(7.43, 9.12), (10.45, 11.35), (14.57, 21.37)],
            duck_gain=0.15,
            unspoken_gain=1.0,
            ducking_style="dynamic",
            hold_seconds=None,
        )
        env_with_hold, meta_hold = build_ducking_envelope(
            total_frames,
            sample_rate=sr,
            speech_intervals=speech_intervals,
            duck_gain=0.15,
            unspoken_gain=1.0,
            ducking_style="dynamic",
            hold_seconds=3.2,
            same_speaker_hold_seconds=4.0,
            hold_gap_gain=0.11,
        )

        slice_12_14_no_hold = env_no_hold[int(12.0 * sr) : int(14.0 * sr)]
        slice_12_14_with_hold = env_with_hold[int(12.0 * sr) : int(14.0 * sr)]
        self.assertGreater(float(np.mean(slice_12_14_no_hold)), 0.85)
        self.assertLessEqual(float(np.max(slice_12_14_with_hold)), 0.15 + 1e-4)
        self.assertGreaterEqual(meta_hold["held_gap_count"], 1)
        reasons = {g["reason"] for g in meta_hold["held_gaps"]}
        self.assertIn("same_speaker_continuous_hold", reasons)

    def test_04_lufs_normalization_reduces_male_female_loudness_disparity_without_clipping(self):
        """4. LUFS正規化（-16 LUFS / -18 LUFS / FFmpeg loudnorm対応）によって男女話者間の音量差が減り、音割れしないこと."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            male_wav = tmp_dir / "male_54.wav"
            female_wav = tmp_dir / "female_56.wav"

            _synthesize_mora_conditioned_speech_wav(
                male_wav,
                "それはあらゆることに当てはまる。",
                target_duration_seconds=1.70,
                f0_hz=118.0,
                spectral_centroid_hz=640.0,
                peak_amplitude=0.42,
                seed=101,
            )
            _synthesize_mora_conditioned_speech_wav(
                female_wav,
                "あらゆることに？",
                target_duration_seconds=0.92,
                f0_hz=215.0,
                spectral_centroid_hz=1120.0,
                peak_amplitude=0.78,
                seed=202,
            )

            _, m_rep = normalize_tts_wav_lufs(male_wav, tmp_dir / "male_norm.wav", target_lufs=-16.0)
            _, f_rep = normalize_tts_wav_lufs(female_wav, tmp_dir / "female_norm.wav", target_lufs=-16.0)

            m_rep["utterance_id"] = "54"
            m_rep["speaker"] = "SPEAKER_01"
            f_rep["utterance_id"] = "56"
            f_rep["speaker"] = "SPEAKER_00"

            summary = build_lufs_normalization_report([m_rep, f_rep], target_lufs=-16.0)
            self.assertGreater(summary["inter_speaker_lufs_spread_before_lu"], 4.0)
            self.assertLess(
                summary["inter_speaker_lufs_spread_after_lu"],
                summary["inter_speaker_lufs_spread_before_lu"] * 0.45,
            )
            self.assertTrue(summary["all_clipping_free"])
            self.assertTrue(summary["loudness_disparity_reduced"])

            _, f_rep_18 = normalize_tts_wav_lufs(
                female_wav,
                tmp_dir / "female_norm_18.wav",
                target_lufs=-18.0,
                method="ffmpeg_loudnorm",
            )
            self.assertTrue(f_rep_18["clipping_free"])
            self.assertLess(f_rep_18["after"]["integrated_lufs"], f_rep_18["before"]["integrated_lufs"])

    def test_05_integrated_mixer_and_subtitles_preserve_phase13_2_guarantees(self):
        """5. ミキサーと字幕生成の統合実行でLUFS正規化・ダッキングホールド・字幕同期が同時に機能し、Phase 13.2に回帰がないこと."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            vid = ensure_30s_verification_video(tmp_dir / "src.mp4", duration_seconds=24.0)
            w54 = tmp_dir / "54.wav"
            w56 = tmp_dir / "56.wav"
            w57 = tmp_dir / "57.wav"
            _synthesize_mora_conditioned_speech_wav(
                w54, "それはあらゆることに当てはまる。", target_duration_seconds=1.68, f0_hz=118.0, peak_amplitude=0.44
            )
            _synthesize_mora_conditioned_speech_wav(
                w56, "あらゆることに？", target_duration_seconds=0.92, f0_hz=215.0, peak_amplitude=0.76
            )
            _synthesize_mora_conditioned_speech_wav(
                w57,
                "でも肉体労働や組み立て作業では人型ロボットの能力も必要になりますが、まだそこまでは至っていませんよね？",
                target_duration_seconds=6.80,
                f0_hz=205.0,
                peak_amplitude=0.74,
            )
            entries = [
                {
                    "id": "54",
                    "speaker": "SPEAKER_01",
                    "text_ja": "それはあらゆることに当てはまる。",
                    "placement_seconds": 7.43,
                    "actual_tts_duration_seconds": 1.68,
                    "audio_file": str(w54),
                    "generation_status": "complete",
                },
                {
                    "id": "56",
                    "speaker": "SPEAKER_00",
                    "text_ja": "あらゆることに？",
                    "placement_seconds": 10.45,
                    "actual_tts_duration_seconds": 0.92,
                    "audio_file": str(w56),
                    "generation_status": "complete",
                },
                {
                    "id": "57",
                    "speaker": "SPEAKER_00",
                    "text_ja": "でも肉体労働や組み立て作業では人型ロボットの能力も必要になりますが、まだそこまでは至っていませんよね？",
                    "placement_seconds": 14.57,
                    "actual_tts_duration_seconds": 6.80,
                    "audio_file": str(w57),
                    "generation_status": "complete",
                },
            ]
            placements = {
                e["id"]: {
                    "source_start_seconds": e["placement_seconds"],
                    "clip_relative_start_seconds": e["placement_seconds"],
                    "clip_relative_end_seconds": round(e["placement_seconds"] + e["actual_tts_duration_seconds"], 4),
                    "active_audio_file": e["audio_file"],
                }
                for e in entries
            }
            _, mix_rep = _fallback_mix_audio_and_video(
                video_path=vid,
                feedback_doc={"entries": entries},
                placements_map=placements,
                selection_start=0.0,
                clip_duration=24.0,
                output_dir=tmp_dir / "mix",
                mix_mode="continuous-voiceover",
                ducking_style="continuous",
                duck_gain=0.18,
                unspoken_gain=0.18,
                normalize_tts_lufs=True,
                target_lufs=-16.0,
                duck_hold_seconds=3.2,
                same_speaker_hold_seconds=4.0,
                hold_gap_gain=0.11,
            )
            self.assertTrue(mix_rep["lufs_normalization"]["all_clipping_free"])
            self.assertTrue(mix_rep["lufs_normalization"]["loudness_disparity_reduced"])
            self.assertGreaterEqual(mix_rep["ducking_verification"]["held_gap_count"], 1)
            self.assertTrue(
                mix_rep["ducking_verification"]["focus_window_0_12_to_0_14_verification"]["english_prominence_suppressed"]
            )

            _, sub_rep = _fallback_generate_subtitles_and_mp4(
                rel_feedback_doc={"entries": entries},
                dubbed_video_path=mix_rep["dubbed_video_mp4"],
                output_dir=tmp_dir / "subs",
                clip_duration=24.0,
            )
            self.assertEqual(sub_rep["overflow_check"]["overflow_count"], 0)
            self.assertEqual(sub_rep["overflow_check"]["unnatural_break_count"], 0)
            self.assertTrue(sub_rep["subtitle_sync_report"]["all_split_cues_synchronized"])


if __name__ == "__main__":
    unittest.main()
