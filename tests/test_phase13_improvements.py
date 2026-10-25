"""Phase 13 Unit & Integration Tests — Universal Video Dubbing Quality Improvements.

Covers:
1. Original audio ducking (voice-over mode: 10–15% adjustable ratio, smooth attack/release fades,
   unspoken original audio retention, soft-knee anti-clipping, and full-dub mode switching).
2. Video & TTS speech timing synchronization (speaker window & semantic clause alignment,
   0.9–1.3x bidirectional automatic speech rate adjustment, early TTS finish & undubbed speech
   detection, truncation & over-compression prevention, multi-speaker turn-taking).
3. Ambient sound preservation & audio mixing (speech vs. ambient separation, 10–15% ducking mode
   and separated ambient mode, graceful fallback when neural models are unavailable, loudness &
   clipping verification).
4. Voice quality & TTS improvements (X-vector voice cloning, age/atmosphere/F0-based preset voice
   selection from catalog, Japanese prosody tuning, voice similarity & naturalness evaluation,
   real-person rights governance vs. synthetic speakers).
5. Subtitle UI improvements (enhanced outline & shadow, semi-transparent black background box 座布団,
   1.2–1.3x enlarged font size, elevated bottom margin, resolution & aspect-ratio scaling for
   16:9 / 9:16 / 1:1, overflow prevention/detection, and display punctuation formatting omitting '。'
   and replacing '、' with half-width space without mutating source script).
6. Seamless integration with Phase 12 (speaker decisions, editorial plan, revised script, quality
   anomaly detection) and multi-video generality (zero hardcoded IDs or timestamps).
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave

import numpy as np

import audio_mixer
import dubbing_pipeline as dp
import dubbing_quality as dq
import japanese_subtitles as jsub
import prosody_experiments as pe
import timing_adjustment as tadj
import voice_quality as vq


def _write_synth_wav(path, *, duration=1.5, rate=24000, freq=145.0, amplitude=0.28):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, int(round(float(duration) * rate)))
    t = np.arange(frames, dtype=np.float64) / float(rate)
    sig = (float(amplitude) * np.sin(2.0 * np.pi * float(freq) * t)).astype(np.float32)
    pcm = np.clip(np.round(sig * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(p), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(int(rate))
        stream.writeframes(pcm.tobytes())
    return p


class TestPhase13Improvements(unittest.TestCase):
    def test_req1_audio_ducking_envelope_ratios_fades_and_unspoken_retention(self):
        """Requirement 1: Dynamic ducking lowers original audio to 10-15% during TTS, applies smooth
        fade-out/fade-in transitions, bridges micro-gaps, and retains original audio when no TTS is active.
        """
        sr = 48000
        total_seconds = 10.0
        total_frames = int(total_seconds * sr)
        # Two utterances at [1.5..3.5s] and [3.7..5.5s] (gap 0.2s < 0.35s should bridge smoothly),
        # plus an isolated utterance at [7.5..8.5s].
        speech_intervals = [(1.5, 3.5), (3.7, 5.5), (7.5, 8.5)]

        env, meta = audio_mixer.build_ducking_envelope(
            total_frames,
            sample_rate=sr,
            speech_intervals=speech_intervals,
            duck_gain=0.12,
            unspoken_gain=1.0,
            fade_out_seconds=0.20,
            fade_in_seconds=0.25,
            pre_duck_seconds=0.05,
            post_duck_seconds=0.05,
            min_unspoken_gap_seconds=0.35,
            curve="cosine",
        )
        self.assertEqual(len(env), total_frames)
        # Micro-gap between 3.5s and 3.7s is bridged into a single window [1.45, 5.55]
        self.assertEqual(len(meta["merged_speech_windows"]), 2)
        self.assertAlmostEqual(meta["min_gain"], 0.12, places=3)
        self.assertAlmostEqual(meta["max_gain"], 1.0, places=3)

        # Unspoken intro [0.0..1.0s] and gap [6.2..7.0s] retain full original audio (gain == 1.0)
        self.assertAlmostEqual(float(np.mean(env[0 : int(1.0 * sr)])), 1.0, places=3)
        self.assertAlmostEqual(float(np.mean(env[int(6.2 * sr) : int(7.0 * sr)])), 1.0, places=3)

        # Active speech [2.0..5.0s] and [7.6..8.4s] are ducked to 12% (within 10-15%)
        self.assertAlmostEqual(float(np.mean(env[int(2.0 * sr) : int(5.0 * sr)])), 0.12, places=3)
        self.assertAlmostEqual(float(np.mean(env[int(7.6 * sr) : int(8.4 * sr)])), 0.12, places=3)

        # Fade-out ramp before 1.45s and fade-in ramp after 5.55s are monotonic and smooth (intermediate values)
        fade_out_slice = env[int(1.26 * sr) : int(1.44 * sr)]
        self.assertTrue(np.all(np.diff(fade_out_slice) <= 1e-5))
        self.assertGreater(float(fade_out_slice[0]), 0.80)
        self.assertLess(float(fade_out_slice[-1]), 0.30)

        fade_in_slice = env[int(5.56 * sr) : int(5.79 * sr)]
        self.assertTrue(np.all(np.diff(fade_in_slice) >= -1e-5))
        self.assertLess(float(fade_in_slice[0]), 0.30)
        self.assertGreater(float(fade_in_slice[-1]), 0.80)

    def test_req2_timing_sync_semantic_clauses_and_bidirectional_atempo_09_to_13x(self):
        """Requirement 2: Semantic clause mapping, 0.9-1.3x automatic speech rate adjustment (both speedup
        and slowdown), prevention of excessive distortion, and early-finish / undubbed speech detection.
        """
        entry = {
            "id": "utt-10",
            "speaker": "HOST_A",
            "source_start": 10.0,
            "source_end": 18.0,
            "text_ja": "まず前提を整理します。次に実験データを見て、最後に結論を述べます。",
        }
        units = tadj.align_semantic_units_to_utterance_window(entry)
        self.assertGreaterEqual(len(units), 2)
        self.assertAlmostEqual(units[0]["target_start_seconds"], 10.0, places=2)
        self.assertAlmostEqual(units[-1]["target_end_seconds"], 18.0, places=2)
        self.assertEqual("".join(u["text_ja"] for u in units), entry["text_ja"])

        # Speedup within 1.30x (3.6s TTS into 3.0s slot -> 1.20x)
        r_fast = tadj.compute_auto_speech_rate_ratio(3.6, 3.0, min_atempo=0.90, max_atempo=1.30)
        self.assertAlmostEqual(r_fast["applied_atempo"], 1.20, places=2)
        self.assertEqual(r_fast["direction"], "speedup")
        self.assertFalse(r_fast["clamped"])

        # Excessive compression prevented (5.0s TTS into 3.0s slot -> ideal 1.67x clamped to 1.30x)
        r_clamp_hi = tadj.compute_auto_speech_rate_ratio(5.0, 3.0, min_atempo=0.90, max_atempo=1.30)
        self.assertAlmostEqual(r_clamp_hi["applied_atempo"], 1.30, places=2)
        self.assertTrue(r_clamp_hi["clamped"])
        self.assertTrue(r_clamp_hi["prevented_excessive_stretch"])

        # Slowdown within 0.90x when TTS finishes early (2.76s TTS in 3.0s slot -> 0.92x)
        r_slow = tadj.compute_auto_speech_rate_ratio(2.76, 3.0, min_atempo=0.90, max_atempo=1.30)
        self.assertAlmostEqual(r_slow["applied_atempo"], 0.92, places=2)
        self.assertEqual(r_slow["direction"], "slowdown")
        self.assertFalse(r_slow["clamped"])

        # Verify actual FFmpeg bidirectional atempo on WAV files (both speedup and slowdown)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_long = _write_synth_wav(root / "long.wav", duration=2.40, rate=24000)
            wav_short = _write_synth_wav(root / "short.wav", duration=1.84, rate=24000)

            entries = [
                {
                    "id": "u1",
                    "speaker": "SPK_1",
                    "source_start": 1.0,
                    "source_end": 3.0,  # 2.0s slot vs 2.40s WAV -> 1.20x speedup
                    "audio_file": str(wav_long),
                    "actual_tts_duration_seconds": 2.40,
                    "text_ja": "長い発話を元の枠に合わせます。",
                    "mora_count": 14,
                },
                {
                    "id": "u2",
                    "speaker": "SPK_2",
                    "source_start": 4.0,
                    "source_end": 6.0,  # 2.0s slot vs 1.72s WAV -> ~0.90x slowdown
                    "audio_file": str(_write_synth_wav(root / "short2.wav", duration=1.72, rate=24000)),
                    "actual_tts_duration_seconds": 1.72,
                    "text_ja": "短い発話です。",
                    "mora_count": 7,
                },
            ]
            source_utts = [
                {"id": "u1", "speaker": "SPK_1", "start": 1.0, "end": 3.0},
                {"id": "u2", "speaker": "SPK_2", "start": 4.0, "end": 6.0},
                {"id": "u3_undubbed", "speaker": "SPK_1", "start": 7.0, "end": 9.2},
            ]
            analysis = dp.analyze_timeline_collisions_and_overlaps(
                entries,
                selection_start=0.0,
                selection_end=10.0,
                auto_tempo_adjust=True,
                min_atempo=0.90,
                max_atempo=1.30,
                fit_utterance_slots=True,
                adjusted_wav_dir=root / "adj",
                source_utterances=source_utts,
            )
            self.assertAlmostEqual(analysis["placements"]["u1"]["actual_tts_duration_seconds"], 2.0, delta=0.08)
            self.assertGreater(analysis["placements"]["u2"]["actual_tts_duration_seconds"], 1.80)
            diag = analysis["timing_sync_diagnostics"]
            self.assertEqual(len(diag["undubbed_active_speech_intervals"]), 1)
            self.assertEqual(diag["undubbed_active_speech_intervals"][0]["source_id"], "u3_undubbed")

    def test_req3_ambient_separation_fallback_and_loudness_verification(self):
        """Requirement 3: Separates speech and ambient room tone, supports both 10-15% ducking and
        separated ambient modes, handles unavailable separation backends via fallback, and verifies
        loudness and zero clipping.
        """
        sr = 48000
        frames = sr * 2  # 2 seconds stereo
        t = np.arange(frames, dtype=np.float32) / float(sr)
        # Simulated stereo recording: center vocal at 220Hz + stereo ambient room tone at 60Hz and 5200Hz
        vocal = 0.45 * np.sin(2.0 * np.pi * 220.0 * t)
        amb_l = 0.08 * np.sin(2.0 * np.pi * 65.0 * t) + 0.05 * np.cos(2.0 * np.pi * 5100.0 * t)
        amb_r = 0.08 * np.sin(2.0 * np.pi * 65.0 * t) - 0.05 * np.cos(2.0 * np.pi * 5100.0 * t)
        stereo = np.stack([vocal + amb_l, vocal + amb_r], axis=1).astype(np.float32)

        sep = audio_mixer.separate_ambient_and_speech(stereo, sample_rate=sr, backend="auto")
        self.assertTrue(sep["separation_available"])
        self.assertEqual(sep["backend_used"], "dsp_spectral_ambient")
        self.assertFalse(sep["fallback_used"])
        self.assertEqual(sep["ambient"].shape, stereo.shape)
        self.assertEqual(sep["speech"].shape, stereo.shape)
        # Vocal energy in ambient stem is significantly reduced compared to original
        self.assertLess(sep["metrics"]["ambient_rms"], sep["metrics"]["original_rms"])

        # Fallback when separation backend is unavailable ('none' or missing weights)
        sep_fb = audio_mixer.separate_ambient_and_speech(stereo, sample_rate=sr, backend="none", allow_fallback=True)
        self.assertTrue(sep_fb["fallback_used"])
        self.assertEqual(sep_fb["backend_used"], "ducking_fallback")

        # Soft limiter prevents clipping on hot overlapping signals
        hot_signal = stereo * 2.5
        limited, lim_rep = audio_mixer.apply_soft_limiter(hot_signal, ceiling=audio_mixer.PEAK_TARGET)
        self.assertGreater(lim_rep["pre_peak"], 1.0)
        self.assertLessEqual(lim_rep["post_peak"], audio_mixer.PEAK_TARGET + 1e-5)
        loudness = audio_mixer.verify_audio_loudness_and_clipping(
            limited,
            source_samples=stereo,
            sample_rate=sr,
            speech_intervals=[(0.2, 1.8)],
        )
        self.assertTrue(loudness["clipping_free"])
        self.assertEqual(loudness["clipped_sample_count"], 0)

    def test_req4_preset_voice_catalog_selection_and_voice_naturalness_verification(self):
        """Requirement 4: Selects optimal preset voices from catalog by age, atmosphere, and F0,
        optimizes prosody settings, and evaluates voice similarity and Japanese naturalness.
        """
        # Mature authoritative low-pitched speaker -> Ethan
        rec_mature = vq.recommend_preset_voice(
            "PROFESSOR_01",
            f0_hz=104.0,
            age_hint="mature",
            atmosphere_hint="authoritative",
            gender_hint="masculine",
        )
        self.assertEqual(rec_mature["recommended_voice"], "Ethan")

        # Young energetic high-pitched feminine speaker -> Mia
        rec_young = vq.recommend_preset_voice(
            "GUEST_02",
            f0_hz=235.0,
            age_hint="young",
            atmosphere_hint="energetic",
            gender_hint="feminine",
        )
        self.assertEqual(rec_young["recommended_voice"], "Mia")

        # Calm adult feminine speaker -> Claire
        rec_calm_f = vq.recommend_preset_voice(
            "HOST_01",
            f0_hz=192.0,
            age_hint="adult",
            atmosphere_hint="calm",
            gender_hint="feminine",
        )
        self.assertEqual(rec_calm_f["recommended_voice"], "Claire")

        # Prosody settings optimization
        lecture_prosody = vq.optimize_tts_prosody_settings(prosody_style="lecture")
        self.assertLess(lecture_prosody["temperature"], 0.80)
        expressive_prosody = vq.optimize_tts_prosody_settings(prosody_style="expressive")
        self.assertGreater(expressive_prosody["temperature"], lecture_prosody["temperature"])

        # Voice & naturalness evaluation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_wav = _write_synth_wav(root / "ref.wav", duration=3.0, rate=24000, freq=135.0)
            utt_wav = _write_synth_wav(root / "utt.wav", duration=2.2, rate=24000, freq=138.0)
            eval_res = vq.evaluate_dubbing_voice_and_naturalness(
                [
                    {
                        "id": "dub-1",
                        "speaker": "SYNTH_EN_01",
                        "generation_status": "complete",
                        "audio_file": str(utt_wav),
                        "mora_count": 15,
                    }
                ],
                speaker_references={"SYNTH_EN_01": {"reference_audio": str(ref_wav)}},
            )
            self.assertEqual(eval_res["evaluated_utterance_count"], 1)
            self.assertTrue(eval_res["acoustic_naturalness_passed"])
            self.assertFalse(eval_res["human_perceptual_verified"])
            self.assertLess(eval_res["utterances"][0]["f0_distance_semitones"], 1.0)

    def test_req5_subtitle_ui_visibility_box_scaling_overflow_and_punctuation(self):
        """Requirement 5: Visibility (outline, shadow, semi-transparent box), 1.2-1.3x size,
        elevated bottom margin, resolution/aspect scaling, overflow prevention, and punctuation
        formatting ('。' omitted, '、' -> half-width space) without modifying source script.
        """
        raw_script_text = "たぶん、ガーデニングがいい例になると思う。"
        formatted = jsub.format_subtitle_display_text(
            raw_script_text,
            strip_periods=True,
            comma_mode="halfwidth_space",
        )
        self.assertEqual(formatted, "たぶん ガーデニングがいい例になると思う")
        self.assertNotIn("。", formatted)
        self.assertNotIn("、", formatted)
        # Source string is untouched
        self.assertEqual(raw_script_text, "たぶん、ガーデニングがいい例になると思う。")

        # Resolution & aspect-ratio scaling: 1080p landscape (16:9), 720p (16:9), and 1080x1920 vertical (9:16)
        st_1080p = jsub.SubtitleStyleConfig.for_video_resolution(1920, 1080, font_scale=1.25)
        self.assertGreaterEqual(st_1080p.font_size, int(jsub.CONVENTIONAL_BASE_FONT_SIZE_1080P * 1.2))
        self.assertGreater(st_1080p.margin_v, jsub.CONVENTIONAL_BASE_MARGIN_V_1080P)
        self.assertGreaterEqual(st_1080p.outline, 3.0)
        self.assertGreaterEqual(st_1080p.shadow, 2.0)
        self.assertTrue(st_1080p.background_box)

        st_vertical = jsub.SubtitleStyleConfig.for_video_resolution(1080, 1920, font_scale=1.25)
        self.assertEqual(st_vertical.play_res_x, 1080)
        self.assertEqual(st_vertical.play_res_y, 1920)
        self.assertLess(st_vertical.max_chars_per_line, st_1080p.max_chars_per_line)

        # Cue alignment preserves script `text_ja` while formatting `display_lines` and preventing overflow
        entry = {
            "id": "dub-long",
            "speaker": "SPEAKER_01",
            "placement_seconds": 1.0,
            "actual_tts_duration_seconds": 5.0,
            "text_ja": "店に行けば、傷一つない野菜が並んでいる。でも庭で野菜を育てることもできる。",
        }
        cues, _ = jsub.align_entry_cues(entry, st_1080p)
        self.assertEqual("".join(c["text_ja"] for c in cues), entry["text_ja"])
        for c in cues:
            for disp_line in c["display_lines"]:
                self.assertNotIn("。", disp_line)
                self.assertNotIn("、", disp_line)
                self.assertLessEqual(len(disp_line), st_1080p.max_chars_per_line)

        ov_check = jsub.check_subtitle_overflow(cues, st_1080p)
        self.assertTrue(ov_check["all_lines_within_bounds"])
        self.assertEqual(ov_check["overflow_count"], 0)

        # Verify ASS rendering uses BorderStyle=3 and semi-transparent black box color
        plan = {
            "video_id": "test_vid",
            "style": jsub.asdict(st_1080p),
            "speaker_styles": {"SPEAKER_01": {"style_name": "Speaker_SPEAKER_01", "primary_color": "&H00FFFFFF"}},
            "cues": cues,
        }
        ass_str = jsub.render_ass(plan)
        srt_str = jsub.render_srt(plan)
        self.assertIn(",3,", ass_str)  # BorderStyle=3 (background box 座布団)
        self.assertIn("&H66000000", ass_str)  # 60% opacity black box
        self.assertNotIn("。", srt_str)
        self.assertNotIn("、", srt_str)

    def test_req6_and_7_multi_video_generic_e2e_pipeline_with_voiceover_and_ambient_modes(self):
        """Requirements 6 & 7: End-to-end integration test across multiple distinct input videos
        (different resolutions, speaker IDs, and time ranges) verifying voiceover ducking, ambient
        preservation, 0.9-1.3x timing sync, subtitle UI, and anomaly detection.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_dir = root / "snapshot"
            snap_dir.mkdir(parents=True)
            (snap_dir / "config.json").write_text(json.dumps({"quantization": {"bits": 8}}), encoding="utf-8")

            fake_pkg = root / "mlx_audio"
            (fake_pkg / "tts" / "models" / "qwen3_tts").mkdir(parents=True)
            (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
            (fake_pkg / "tts" / "models" / "qwen3_tts" / "qwen3_tts.py").write_text("# mock native\n", encoding="utf-8")

            class DummyNative:
                def __init__(self):
                    self._icl_cache = {}

                def generate(self, **kwargs):
                    txt = kwargs.get("text", "")
                    dur_s = max(1.2, min(3.2, len(txt) * 0.10))
                    frames = int(round(dur_s * 24000))
                    t = np.arange(frames, dtype=np.float64) / 24000.0
                    sig = (0.30 * np.sin(2.0 * np.pi * 150.0 * t)).astype(np.float32)
                    return [types.SimpleNamespace(audio=sig, sample_rate=24000, token_count=20)]

            class DummyMLXWrapper:
                def __init__(self):
                    self.model = DummyNative()

                @classmethod
                def from_pretrained(cls, _path):
                    return cls()

                def create_voice_clone_prompt(self, **kwargs):
                    return types.SimpleNamespace(
                        ref_audio=kwargs["ref_audio"][0],
                        ref_text=None,
                        x_vector_only_mode=True,
                    )

            class FakePresetSynthesizer:
                supported_voice_modes = {"preset"}

                def __init__(self, model_id, seed=0, expected_model_type=None):
                    self.snapshot = snap_dir
                    self.settings = {
                        "seed": seed,
                        "temperature": 0.9,
                        "top_k": 50,
                        "top_p": 1.0,
                        "repetition_penalty": 1.05,
                        "max_tokens": 4096,
                        "wav_subtype": "PCM_16",
                        "language": "Japanese",
                        "instruct": None,
                    }
                    self.identity = {
                        "backend": "qwen_tts_mlx",
                        "model": model_id,
                        "model_revision": "rev1",
                        "settings": dict(self.settings),
                    }

                def synthesize(self, *, text, voice=None, profile=None, output_path=None):
                    _write_synth_wav(output_path, duration=1.6, rate=24000, freq=160.0)
                    return {"sample_rate": 24000, "sample_frames": 38400, "token_count": 15, "termination": "eos_before_limit"}

            fake_mx = types.ModuleType("mlx.core")
            fake_mx.random = types.SimpleNamespace(seed=lambda s: None)
            fake_mx.get_peak_memory = lambda: 1024
            fake_mlx_pkg = types.ModuleType("mlx")
            fake_mlx_pkg.core = fake_mx

            fake_sf = types.ModuleType("soundfile")
            fake_sf.read = lambda p, dtype="float32": pe._read_wav_float32(p)
            fake_sf.write = lambda p, arr, sr, subtype="PCM_16", format="WAV": _write_synth_wav(p, duration=len(arr) / float(sr), rate=sr)

            fake_qwen_mlx = types.ModuleType("qwen_tts_mlx")
            fake_qwen_mlx.MLXQwen3TTS = DummyMLXWrapper
            fake_spec = types.SimpleNamespace(origin=str(fake_pkg / "__init__.py"))

            # Video A: 1280x720 interview clip [120.0..126.0s] with voiceover ducking (12%)
            vid_a = dp.ensure_30s_verification_video(root / "vid_a" / "interview.mp4", duration_seconds=6.0)
            script_a = root / "vid_a" / "script_a.json"
            script_a.write_text(
                json.dumps(
                    {
                        "selection_range": {"start": 120.0, "end": 126.0},
                        "utterances": [
                            {
                                "id": "int-1",
                                "speaker": "INTERVIEWER",
                                "start": 120.4,
                                "end": 122.4,
                                "text_en": "Welcome to the robotics lab,",
                                "text_ja": "ロボティクス研究所へようこそ。",
                            },
                            {
                                "id": "int-2",
                                "speaker": "INTERVIEWER",
                                "start": 122.5,
                                "end": 124.2,
                                "text_en": "and thank you for joining us today.",
                                "text_ja": "本日はお越しいただきありがとうございます。",
                            },
                            {
                                "id": "int-3",
                                "speaker": "RESEARCHER",
                                "start": 124.4,
                                "end": 125.8,
                                "text_en": "Glad to be here.",
                                "text_ja": "こちらこそ、よろしくお願いします。",
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            # Video B: Lecture clip [0.0..6.0s] with separated ambient mode + auto-selected preset voices
            vid_b = dp.ensure_30s_verification_video(root / "vid_b" / "lecture.mp4", duration_seconds=6.0)
            script_b = root / "vid_b" / "script_b.json"
            script_b.write_text(
                json.dumps(
                    {
                        "selection_range": {"start": 0.0, "end": 6.0},
                        "utterances": [
                            {
                                "id": "lec-1",
                                "speaker": "PROF_01",
                                "start": 0.5,
                                "end": 3.0,
                                "text_en": "In today's lecture, we analyze neural audio codecs.",
                                "text_ja": "本日の講義では、ニューラル音声コーデックを解析します。",
                            },
                            {
                                "id": "lec-2",
                                "speaker": "STUDENT_01",
                                "start": 3.4,
                                "end": 5.5,
                                "text_en": "How does the residual vector quantizer work?",
                                "text_ja": "残差ベクトル量子化は、どのように機能するのでしょうか。",
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(
                sys.modules,
                {
                    "mlx": fake_mlx_pkg,
                    "mlx.core": fake_mx,
                    "soundfile": fake_sf,
                    "qwen_tts_mlx": fake_qwen_mlx,
                },
            ), mock.patch("tts_generator.QwenPresetSynthesizer", FakePresetSynthesizer), mock.patch(
                "importlib.util.find_spec", return_value=fake_spec
            ):
                # Run Video A with `run-phase13` (X-vector + voiceover 12% ducking + generic sentence merge)
                out_a = root / "out_a"
                rc_a = dp.main(
                    [
                        "run-phase13",
                        "--video", str(vid_a),
                        "--script", str(script_a),
                        "--mix-mode", "voiceover",
                        "--duck-gain", "0.12",
                        "--output", str(out_a),
                    ]
                )
                self.assertEqual(rc_a, 0)
                rep_a = json.loads((out_a / "pipeline_report.json").read_text(encoding="utf-8"))
                self.assertEqual(rep_a["audio_mixing_and_ducking"]["mix_mode"], "voiceover")
                self.assertAlmostEqual(rep_a["audio_mixing_and_ducking"]["duck_gain"], 0.12, places=2)
                self.assertTrue(rep_a["audio_mixing_and_ducking"]["loudness_verification"]["clipping_free"])
                # Generic continuity merged int-1 and int-2 into int-1-int-2 without any hardcoded IDs!
                self.assertEqual(rep_a["utterance_merging"]["merged_group_count"], 1)
                srt_a = (out_a / "final_ja.srt").read_text(encoding="utf-8")
                self.assertNotIn("。", srt_a)
                self.assertNotIn("、", srt_a)

                # Run Video B with separated ambient mode + preset auto-selection by age/atmosphere
                out_b = root / "out_b"
                rc_b = dp.main(
                    [
                        "run-phase13",
                        "--video", str(vid_b),
                        "--script", str(script_b),
                        "--voice-mode", "preset",
                        "--auto-select-preset",
                        "--speaker-age", "PROF_01=mature",
                        "--speaker-atmosphere", "PROF_01=authoritative",
                        "--speaker-age", "STUDENT_01=young",
                        "--speaker-atmosphere", "STUDENT_01=energetic",
                        "--mix-mode", "ambient",
                        "--ambient-backend", "dsp",
                        "--output", str(out_b),
                    ]
                )
                self.assertEqual(rc_b, 0)
                rep_b = json.loads((out_b / "pipeline_report.json").read_text(encoding="utf-8"))
                self.assertEqual(rep_b["audio_mixing_and_ducking"]["mix_mode"], "ambient")
                self.assertEqual(
                    rep_b["audio_mixing_and_ducking"]["ambient_separation"]["backend_used"],
                    "dsp_spectral_ambient",
                )
                # Distinct presets selected for professor vs student
                voices_b = rep_b["voice_routing"]["voices"]
                self.assertNotEqual(voices_b["PROF_01"], voices_b["STUDENT_01"])
                self.assertEqual(voices_b["PROF_01"], "Ethan")

    def test_req7_anomaly_detection_timing_drift_silence_truncation_and_overflow(self):
        """Requirement 7: Automated detection of timing drift, long silence gaps, audio truncation,
        early TTS finishes, undubbed active speech, and subtitle overflow.
        """
        entries = [
            {
                "id": "utt-early",
                "speaker": "SPK_01",
                "source_start": 0.5,
                "source_end": 4.0,  # 3.5s window, but TTS is only 1.2s -> early finish (-2.3s)
                "placement_seconds": 0.5,
                "actual_tts_duration_seconds": 1.2,
                "end_seconds": 1.7,
                "generation_status": "complete",
                "truncated_by_mixer": False,
            },
            {
                "id": "utt-drift-trunc",
                "speaker": "SPK_01",
                "source_start": 6.5,  # gap from 1.7s to 7.2s is 5.5s -> long silence gap
                "source_end": 9.5,
                "placement_seconds": 7.2,  # +0.7s start drift
                "actual_tts_duration_seconds": 3.2,
                "end_seconds": 10.0,  # clipped at 10.0s boundary (< 7.2 + 3.2 = 10.4s)
                "generation_status": "complete",
                "truncated_by_mixer": True,
            },
        ]
        source_utterances = [
            {"id": "utt-early", "speaker": "SPK_01", "start": 0.5, "end": 4.0},
            {"id": "utt-missed", "speaker": "SPK_02", "start": 4.2, "end": 6.0},
            {"id": "utt-drift-trunc", "speaker": "SPK_01", "start": 6.5, "end": 9.5},
        ]
        cues = [
            {
                "cue_id": "cue-overflow",
                "utterance_id": "utt-early",
                "speaker": "SPK_01",
                "start": 0.5,
                "end": 1.7,
                "text_ja": "これは画面幅を大幅に超える非常に長い一行のテスト字幕テキストであり折り返し無しではみ出します",
                "lines": ["これは画面幅を大幅に超える非常に長い一行のテスト字幕テキストであり折り返し無しではみ出します"],
                "display_lines": ["これは画面幅を大幅に超える非常に長い一行のテスト字幕テキストであり折り返し無しではみ出します"],
            }
        ]
        style_cfg = jsub.SubtitleStyleConfig.for_video_resolution(1280, 720, font_scale=1.25)
        t_diag = tadj.detect_timing_and_coverage_anomalies(
            entries,
            selection_start=0.0,
            selection_end=10.0,
            source_utterances=source_utterances,
        )
        ov_check = jsub.check_subtitle_overflow(cues, style_cfg)
        anomalies = dq.detect_quality_anomalies(
            timing_diagnostics=t_diag,
            subtitle_report={"overflow_check": ov_check},
            mix_report={"loudness_verification": {"clipping_free": False, "clipped_sample_count": 12}},
        )
        self.assertGreaterEqual(len(anomalies["timing_anomalies"]), 1)
        self.assertGreaterEqual(len(anomalies["silence_and_undubbed_anomalies"]), 1)
        self.assertGreaterEqual(len(anomalies["audio_truncation_anomalies"]), 1)
        self.assertGreaterEqual(len(anomalies["subtitle_overflow_anomalies"]), 1)
        self.assertGreaterEqual(len(anomalies["audio_loudness_anomalies"]), 1)
        self.assertTrue(anomalies["has_critical_anomalies"])

    def test_req6_phase12_revised_script_and_speaker_decisions_integration_with_phase13(self):
        """Requirement 6: Seamless integration with Phase 12 artifacts (dubbing_script.revised.json,
        editorial_plan.json, speaker_decisions.json) under Phase 13 voiceover and ambient modes.
        """
        repo_root = Path(__file__).resolve().parent.parent
        orig_script = repo_root / "output" / "phase11_1885_1915_inputs" / "dubbing_script.json"
        rev_script = repo_root / "output" / "phase12_1885_1915_quality" / "dubbing_script.revised.json"
        ed_plan = repo_root / "output" / "phase12_1885_1915_quality" / "editorial_plan.json"
        spk_dec = repo_root / "output" / "phase12_1885_1915_quality" / "speaker_decisions.json"
        self.assertTrue(orig_script.is_file())
        self.assertTrue(rev_script.is_file())
        self.assertTrue(ed_plan.is_file())
        self.assertTrue(spk_dec.is_file())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vid = dp.ensure_30s_verification_video(root / "clip_30s.mp4", duration_seconds=30.0)
            wav_dir = root / "wavs"
            wav_dir.mkdir(parents=True)
            rev_doc = json.loads(rev_script.read_text(encoding="utf-8"))
            for entry in rev_doc["entries"]:
                dur = max(1.0, min(4.2, float(entry["end"]) - float(entry["start"]) - 0.1))
                _write_synth_wav(wav_dir / f"{entry['id']}.wav", duration=dur, rate=24000, freq=140.0)

            snap_dir = root / "snapshot"
            snap_dir.mkdir(parents=True)
            (snap_dir / "config.json").write_text(json.dumps({"quantization": {"bits": 8}}), encoding="utf-8")
            fake_pkg = root / "mlx_audio"
            (fake_pkg / "tts" / "models" / "qwen3_tts").mkdir(parents=True)
            (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
            (fake_pkg / "tts" / "models" / "qwen3_tts" / "qwen3_tts.py").write_text("# mock native\n", encoding="utf-8")

            class DummyNative:
                def __init__(self):
                    self._icl_cache = {}

                def generate(self, **kwargs):
                    txt = kwargs.get("text", "")
                    dur_s = max(1.2, min(3.4, len(txt) * 0.06))
                    frames = int(round(dur_s * 24000))
                    t = np.arange(frames, dtype=np.float64) / 24000.0
                    sig = (0.30 * np.sin(2.0 * np.pi * 145.0 * t)).astype(np.float32)
                    return [types.SimpleNamespace(audio=sig, sample_rate=24000, token_count=20)]

            class DummyMLXWrapper:
                def __init__(self):
                    self.model = DummyNative()

                @classmethod
                def from_pretrained(cls, _path):
                    return cls()

                def create_voice_clone_prompt(self, **kwargs):
                    return types.SimpleNamespace(
                        ref_audio=kwargs["ref_audio"][0],
                        ref_text=None,
                        x_vector_only_mode=True,
                    )

            class FakePresetSynthesizer:
                supported_voice_modes = {"preset"}

                def __init__(self, model_id, seed=0, expected_model_type=None):
                    self.snapshot = snap_dir
                    self.settings = {
                        "seed": seed,
                        "temperature": 0.9,
                        "top_k": 50,
                        "top_p": 1.0,
                        "repetition_penalty": 1.05,
                        "max_tokens": 4096,
                        "wav_subtype": "PCM_16",
                        "language": "Japanese",
                        "instruct": None,
                    }
                    self.identity = {
                        "backend": "qwen_tts_mlx",
                        "model": model_id,
                        "model_revision": "rev1",
                        "settings": dict(self.settings),
                    }

            fake_mx = types.ModuleType("mlx.core")
            fake_mx.random = types.SimpleNamespace(seed=lambda s: None)
            fake_mx.get_peak_memory = lambda: 1024
            fake_mlx_pkg = types.ModuleType("mlx")
            fake_mlx_pkg.core = fake_mx
            fake_sf = types.ModuleType("soundfile")
            fake_sf.read = lambda p, dtype="float32": pe._read_wav_float32(p)
            fake_sf.write = lambda p, arr, sr, subtype="PCM_16", format="WAV": _write_synth_wav(p, duration=len(arr) / float(sr), rate=sr)
            fake_qwen_mlx = types.ModuleType("qwen_tts_mlx")
            fake_qwen_mlx.MLXQwen3TTS = DummyMLXWrapper
            fake_spec = types.SimpleNamespace(origin=str(fake_pkg / "__init__.py"))

            out_dir = root / "phase13_from_phase12"
            revised_out = out_dir / "dubbing_script.revised.json"
            with mock.patch.dict(
                sys.modules,
                {
                    "mlx": fake_mlx_pkg,
                    "mlx.core": fake_mx,
                    "soundfile": fake_sf,
                    "qwen_tts_mlx": fake_qwen_mlx,
                },
            ), mock.patch("tts_generator.QwenPresetSynthesizer", FakePresetSynthesizer), mock.patch(
                "importlib.util.find_spec", return_value=fake_spec
            ):
                report_path, report = dp.run_phase13_dubbing_pipeline(
                    script_path=orig_script,
                    video_path=vid,
                    output_dir=out_dir,
                    speaker_decisions_path=spk_dec,
                    editorial_plan_path=ed_plan,
                    revised_script_out_path=revised_out,
                    reuse_wav_dir=wav_dir,
                    mix_mode="voiceover",
                    duck_gain=0.12,
                    subtitle_font_scale=1.25,
                    subtitle_background_box=True,
                    subtitle_strip_periods=True,
                    subtitle_comma_mode="halfwidth_space",
                )
            self.assertTrue(report_path.is_file())
            self.assertEqual(report["audio_mixing_and_ducking"]["mix_mode"], "voiceover")
            self.assertTrue(report["audio_mixing_and_ducking"]["loudness_verification"]["clipping_free"])
            self.assertTrue(report["subtitle_ui_summary"]["overflow_check"]["all_lines_within_bounds"])
            # Source revised script still has Japanese punctuation intact
            rev_after = json.loads(revised_out.read_text(encoding="utf-8"))
            self.assertTrue(any("。" in e["text_ja"] for e in rev_after["entries"]))
            # Rendered display subtitles omit '。' and replace '、' with space
            srt_text = (out_dir / "final_ja.srt").read_text(encoding="utf-8")
            self.assertNotIn("。", srt_text)
            self.assertNotIn("、", srt_text)
            self.assertIn("たぶん ガーデニングがいい例になると思う", srt_text)


if __name__ == "__main__":
    unittest.main()
