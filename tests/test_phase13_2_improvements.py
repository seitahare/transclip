"""Regression and unit tests for TransClip Phase 13.2 improvements:
1. Same speaker's utterances all use the locked SpeakerVoiceProfile
2. Distinct speakers never confuse clone profiles
3. Speaker ID / voice profile mismatch is detected
4. Uncertain speaker assignments are never auto-confirmed by LLM/context guesses alone
5. Normal TTS prohibits automatic slowdown (`atempo < 1.0`)
6. Speedup exceeding 1.25x or excessive slowdown is rejected/clamped
7. Short TTS is not stretched and leaves a natural trailing pause
8. Audio truncation, click noise, and excessive resampling are detected
9. Long subtitles are split into multiple cues with at most 2 lines (<= 22 chars/line)
10. Japanese phrases like "ロボットとして" and "組み立て作業では" are never split mid-bunsetsu
11. Phase 13.1 continuous 18% ducking and word-level start synchronization are preserved
12. Preset TTS API (`QwenPresetSynthesizer.synthesize`) accepts both `voice=` and `profile=` without mismatch
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave

import numpy as np

import audio_mixer as am
import dubbing_pipeline as dp
import japanese_subtitles as js
import timing_adjustment as ta
import tts_generator as tg
import voice_consistency as vc


def _write_test_speech_wav(path, *, duration=1.5, rate=24000, f0_hz=120.0, add_click=False, cut_tail=False):
    frames = max(int(0.2 * rate), int(round(duration * rate)))
    t = np.arange(frames, dtype=np.float32) / float(rate)
    sig = (
        0.45 * np.sin(2.0 * np.pi * f0_hz * t)
        + 0.25 * np.sin(2.0 * np.pi * (2.0 * f0_hz) * t + 0.2)
        + 0.12 * np.sin(2.0 * np.pi * (3.0 * f0_hz) * t + 0.5)
    ).astype(np.float32)
    fade = min(int(0.03 * rate), frames // 6)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        sig[:fade] *= ramp
        if not cut_tail:
            sig[-fade:] *= ramp[::-1]
    if cut_tail and frames > 20:
        sig[-20:] = 0.42
    if add_click and frames > 200:
        mid = frames // 2
        sig[mid] = 0.95
        sig[mid + 1] = -0.95
    pcm = np.clip(np.round(sig * 32767.0), -32768, 32767).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return Path(path)


class TestPhase132Improvements(unittest.TestCase):
    def test_01_same_speaker_utterances_use_locked_speaker_voice_profile(self):
        """1. 同じ話者の全発話が固定された同一Speaker Voice Profileを使うこと"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref01 = _write_test_speech_wav(root / "spk01_ref.wav", duration=4.2, f0_hz=115.0)
            registry = vc.SpeakerVoiceRegistry()
            prof = registry.register_speaker_profile(
                speaker_id="SPEAKER_01",
                voice_mode="approved_clone",
                reference_audio_path=ref01,
                clone_method="xvector",
                model_id=dp.DEFAULT_BASE_MODEL,
                model_revision="rev1",
                generation_settings={"seed": 0, "temperature": 0.65, "top_k": 25, "top_p": 0.88},
            )
            tref = dp.TechnicalVoiceReference(speaker_id="SPEAKER_01", reference_audio=ref01)

            utterances = [
                {"id": 51, "speaker": "SPEAKER_01", "text_ja": "チェスではそれくらい強い。"},
                {"id": 54, "speaker": "SPEAKER_01", "text_ja": "それはあらゆることに当てはまる。"},
                {"id": 60, "speaker": "SPEAKER_01", "text_ja": "そう、だからデジタルAIと分けて言ったんだ。"},
                {"id": 61, "speaker": "SPEAKER_01", "text_ja": "ロボットはいわばエンドエフェクターだ。"},
            ]
            verified_ids = []
            for u in utterances:
                locked = registry.verify_utterance_routing(
                    u,
                    requested_voice=f"approved_clone:{tref.profile_fingerprint}",
                    requested_profile=tref,
                )
                verified_ids.append(locked.voice_profile_id)
                self.assertEqual(locked.reference_audio_sha256, prof.reference_audio_sha256)

            self.assertEqual(len(set(verified_ids)), 1)
            self.assertEqual(verified_ids[0], prof.voice_profile_id)

            # Re-registering the same speaker with a different reference WAV must raise SpeakerProfileMismatchError
            other_ref = _write_test_speech_wav(root / "other_ref.wav", duration=4.0, f0_hz=185.0)
            with self.assertRaises(vc.SpeakerProfileMismatchError):
                registry.register_speaker_profile(
                    speaker_id="SPEAKER_01",
                    voice_mode="approved_clone",
                    reference_audio_path=other_ref,
                )

    def test_02_distinct_speakers_never_confuse_clone_profiles(self):
        """2. 異なる話者間でクローンプロファイルが混同されないこと"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref00 = _write_test_speech_wav(root / "spk00_ref.wav", duration=4.0, f0_hz=205.0)
            ref01 = _write_test_speech_wav(root / "spk01_ref.wav", duration=4.5, f0_hz=114.0)

            registry = vc.SpeakerVoiceRegistry()
            p00 = registry.register_speaker_profile(
                speaker_id="SPEAKER_00",
                voice_mode="approved_clone",
                reference_audio_path=ref00,
                clone_method="xvector",
            )
            p01 = registry.register_speaker_profile(
                speaker_id="SPEAKER_01",
                voice_mode="approved_clone",
                reference_audio_path=ref01,
                clone_method="xvector",
            )
            self.assertNotEqual(p00.voice_profile_id, p01.voice_profile_id)
            self.assertNotEqual(p00.reference_audio_sha256, p01.reference_audio_sha256)

            tref00 = dp.TechnicalVoiceReference(speaker_id="SPEAKER_00", reference_audio=ref00)
            tref01 = dp.TechnicalVoiceReference(speaker_id="SPEAKER_01", reference_audio=ref01)

            # Passing SPEAKER_01's profile for a SPEAKER_00 utterance must raise SpeakerProfileMismatchError
            with self.assertRaises(vc.SpeakerProfileMismatchError):
                registry.verify_utterance_routing(
                    {"id": 57, "speaker": "SPEAKER_00"},
                    requested_voice=f"approved_clone:{tref01.profile_fingerprint}",
                    requested_profile=tref01,
                )
            # Passing SPEAKER_00's own profile succeeds
            ok_prof = registry.verify_utterance_routing(
                {"id": 57, "speaker": "SPEAKER_00"},
                requested_voice=f"approved_clone:{tref00.profile_fingerprint}",
                requested_profile=tref00,
            )
            self.assertEqual(ok_prof.speaker_id, "SPEAKER_00")

    def test_03_speaker_profile_mismatch_is_detected(self):
        """3. 話者IDと音声プロファイルの不整合が検出されること"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref00 = _write_test_speech_wav(root / "spk00.wav", duration=4.0, f0_hz=205.0)
            ref01 = _write_test_speech_wav(root / "spk01.wav", duration=4.0, f0_hz=114.0)
            registry = vc.SpeakerVoiceRegistry()
            p00 = registry.register_speaker_profile(
                speaker_id="SPEAKER_00",
                voice_mode="approved_clone",
                reference_audio_path=ref00,
            )
            p01 = registry.register_speaker_profile(
                speaker_id="SPEAKER_01",
                voice_mode="approved_clone",
                reference_audio_path=ref01,
            )

            # Simulate an entry where SPEAKER_00 was mistakenly synthesized with SPEAKER_01's voice_profile_id
            w00 = _write_test_speech_wav(root / "u53.wav", duration=1.2, f0_hz=205.0)
            w00_bad = _write_test_speech_wav(root / "u57_bad.wav", duration=2.5, f0_hz=114.0)
            entries = [
                {
                    "id": 53,
                    "speaker": "SPEAKER_00",
                    "voice_profile_id": p00.voice_profile_id,
                    "reference_audio_sha256": p00.reference_audio_sha256,
                    "audio_file": str(w00),
                },
                {
                    "id": 57,
                    "speaker": "SPEAKER_00",
                    "voice_profile_id": p01.voice_profile_id,
                    "reference_audio_sha256": p01.reference_audio_sha256,
                    "audio_file": str(w00_bad),
                },
            ]
            consistency = vc.evaluate_speaker_voice_consistency(entries, registry=registry)
            self.assertFalse(consistency["all_speakers_consistent"])
            spk00_rep = consistency["speakers"]["SPEAKER_00"]
            self.assertFalse(spk00_rep["profile_lock_verified"])
            self.assertIn("57", spk00_rep["outlier_utterance_ids"])

    def test_04_uncertain_speaker_assignment_is_not_auto_confirmed_by_llm_alone(self):
        """4. 不確実な話者推定が確定扱いされないこと"""
        utterances = [
            {"id": 1, "speaker": "SPEAKER_01", "start": 0.0, "end": 3.0, "text_en": "That is true for software."},
            {"id": 2, "speaker": None, "start": 4.2, "end": 5.5, "text_en": "What about hardware?"},
            {"id": 3, "speaker": "SPEAKER_01", "start": 5.8, "end": 9.0, "text_en": "Hardware takes longer."},
        ]
        records = vc.evaluate_contextual_speaker_assignment(utterances)
        rec_u2 = next(r for r in records if r["utterance_id"] == 2)
        self.assertIsNone(rec_u2["final_assigned_speaker"])
        self.assertTrue(rec_u2["speaker_verification_required"])
        self.assertEqual(rec_u2["assignment_status"], "speaker_verification_required")
        self.assertFalse(rec_u2["llm_only_override_forbidden"] is False)

    def test_05_normal_tts_prohibits_automatic_slowdown(self):
        """5. 通常のTTSで自動スロー再生が適用されないこと"""
        rate_plan = ta.compute_phase13_2_speech_rate_ratio(
            1.15,
            2.40,
            allow_slowdown=False,
            max_atempo=1.20,
        )
        self.assertEqual(rate_plan["applied_atempo"], 1.0)
        self.assertFalse(rate_plan["slowdown_applied"])
        self.assertAlmostEqual(rate_plan["natural_trailing_pause_seconds"], 1.25, places=2)

    def test_06_excessive_speedup_or_slowdown_is_rejected_and_clamped(self):
        """6. 1.25xを超える早回しや過度な引き伸ばしが拒否されること"""
        speed_plan = ta.compute_phase13_2_speech_rate_ratio(
            5.0,
            3.0,
            allow_slowdown=False,
            max_atempo=1.35,  # requests 1.35, but hard ceiling is 1.25
            hard_max_atempo=1.25,
        )
        self.assertLessEqual(speed_plan["applied_atempo"], 1.25)
        self.assertTrue(speed_plan["exceeded_max_atempo"])
        self.assertTrue(speed_plan["requires_structural_resolution"])

        # Even when exceptional slowdown is explicitly enabled, slowdown below 0.95x is clamped to >= 0.95x
        slow_plan = ta.compute_phase13_2_speech_rate_ratio(
            1.0,
            2.0,
            allow_slowdown=True,
            exceptional_min_atempo=0.95,
        )
        self.assertGreaterEqual(slow_plan["applied_atempo"], 0.95)

    def test_07_short_tts_is_not_stretched_and_leaves_natural_pause(self):
        """7. 短いTTSが引き伸ばされず、自然な余白として配置されること"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            short_wav = _write_test_speech_wav(root / "u56.wav", duration=1.05, f0_hz=205.0)
            entries = [
                {
                    "id": 56,
                    "speaker": "SPEAKER_00",
                    "text_en": "Everything?",
                    "text_ja": "あらゆることに？",
                    "mora_count": 8,
                    "start": 10.62,
                    "end": 12.10,
                    "source_start": 10.62,
                    "source_end": 12.10,
                    "audio_file": str(short_wav),
                }
            ]
            analysis = dp.analyze_timeline_collisions_and_overlaps(
                entries,
                selection_start=0.0,
                selection_end=38.65,
                auto_tempo_adjust=True,
                min_atempo=1.00,
                max_atempo=1.20,
                fit_utterance_slots=True,
                allow_slowdown=False,
                adjusted_wav_dir=root / "adjusted",
            )
            p56 = analysis["placements"]["56"]
            self.assertEqual(p56["applied_atempo"], 1.0)
            self.assertAlmostEqual(p56["actual_tts_duration_seconds"], 1.05, places=2)
            self.assertGreater(p56["natural_trailing_pause_seconds"], 0.35)
            self.assertEqual(len(analysis["duration_adjustments"]), 0)

    def test_08_truncation_click_noise_and_excessive_resampling_detected(self):
        """8. 音声途中切断、クリックノイズ、過度な再サンプリングが検出されること"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_wav = _write_test_speech_wav(root / "raw.wav", duration=2.0, f0_hz=120.0)
            bad_wav = _write_test_speech_wav(
                root / "bad.wav",
                duration=1.4,
                f0_hz=120.0,
                add_click=True,
                cut_tail=True,
            )
            entries = [
                {
                    "id": 99,
                    "speaker": "SPEAKER_01",
                    "text_ja": "テスト発話です。",
                    "raw_audio_file": str(raw_wav),
                    "audio_file": str(bad_wav),
                    "slot_duration_seconds": 2.0,
                    "raw_tts_duration_seconds": 2.0,
                    "actual_tts_duration_seconds": 1.4,
                    "applied_atempo": 1.0,
                    "resampling_count": 2,
                }
            ]
            report = ta.build_timing_quality_report(
                entries,
                placements_map={
                    "99": {
                        "slot_duration_seconds": 2.0,
                        "raw_tts_duration_seconds": 2.0,
                        "actual_tts_duration_seconds": 1.4,
                        "applied_atempo": 1.0,
                        "resampling_count": 2,
                        "raw_audio_file": str(raw_wav),
                        "active_audio_file": str(bad_wav),
                    }
                },
                allow_slowdown=False,
                max_atempo=1.20,
            )
            self.assertGreaterEqual(report["anomaly_count"], 1)
            u_rep = report["utterances"][0]
            self.assertIn("excessive_resampling:2", u_rep["issues"])
            self.assertTrue(
                any("click" in i or "truncat" in i for i in u_rep["issues"]),
                f"Expected click or truncation issue in {u_rep['issues']}",
            )

    def test_09_long_subtitles_split_into_multiple_cues_with_max_2_lines(self):
        """9. 長文字幕が最大2行・複数cueに分割されること"""
        style = js.SubtitleStyleConfig.for_video_resolution(1920, 1080)
        self.assertEqual(style.max_lines, 2)
        self.assertLessEqual(style.max_chars_per_line, 22)

        for uid, text_ja, dur in [
            (
                57,
                "でも肉体労働や組み立て作業では人型ロボットの能力も必要になりますが、まだそこまでは至っていませんよね？",
                8.4,
            ),
            (
                61,
                "ロボットはいわばエンドエフェクターで、局所的な知能を持ちつつ大規模AIモデルに管理される形になる。",
                6.8,
            ),
        ]:
            entry = {
                "id": uid,
                "speaker": "SPEAKER_00" if uid == 57 else "SPEAKER_01",
                "text_ja": text_ja,
                "placement_seconds": 15.0 if uid == 57 else 30.0,
                "actual_tts_duration_seconds": dur,
            }
            cues, _ = js.align_entry_cues(entry, style)
            self.assertGreaterEqual(len(cues), 2, f"Expected long utterance {uid} to split into >=2 cues")
            for c in cues:
                disp = c["display_lines"]
                self.assertLessEqual(len(disp), 2)
                for line in disp:
                    self.assertLessEqual(len(line), 22, f"Line exceeded 22 chars: {line!r}")

    def test_10_bunsetsu_phrases_never_split_mid_phrase(self):
        """10. 「ロボットとして」のような語が文節途中で分断されないこと"""
        style = js.SubtitleStyleConfig.for_video_resolution(1920, 1080)
        entry = {
            "id": 60,
            "speaker": "SPEAKER_01",
            "text_ja": "そう、だからデジタルAIと、ロボットとして物理的に具現化されたAIを分けて言ったんだ。",
            "placement_seconds": 25.34,
            "actual_tts_duration_seconds": 4.70,
        }
        cues, _ = js.align_entry_cues(entry, style)
        for c in cues:
            self.assertLessEqual(len(c["display_lines"]), 2)
            for l_idx in range(len(c["display_lines"]) - 1):
                left = c["display_lines"][l_idx]
                right = c["display_lines"][l_idx + 1]
                self.assertFalse(
                    left.endswith("ロボットと") and right.startswith("して"),
                    f"Prohibited split 'ロボットと / して' found in {c['display_lines']}",
                )
        unnatural = js.detect_unnatural_line_breaks(cues)
        self.assertTrue(unnatural["passed"], f"Unnatural line breaks detected: {unnatural}")

    def test_11_phase13_1_continuous_ducking_and_word_level_sync_preserved(self):
        """11. Phase 13.1のダッキングとword-level開始同期が壊れていないこと"""
        sr = 24000
        env = am.build_ducking_envelope(
            total_frames=sr * 10,
            sample_rate=sr,
            speech_intervals=[(1.0, 3.5), (5.0, 8.0)],
            duck_gain=0.18,
            unspoken_gain=0.18,
            ducking_style="continuous",
        )
        # During speech and pauses prior to the final fade-out, envelope stays at 0.18 (never jumping to 1.0)
        mid_slice = env[int(0.5 * sr) : int(8.5 * sr)]
        self.assertTrue(np.allclose(mid_slice, 0.18, atol=1e-3))

        # Word-level start sync skips leading fillers ("Right, so")
        words = [
            {"word": "Right,", "start": 24.42, "end": 24.74},
            {"word": "so", "start": 25.04, "end": 25.24},
            {"word": "that's", "start": 25.34, "end": 25.60},
            {"word": "why", "start": 25.60, "end": 25.80},
        ]
        cls_res = ta.classify_english_words_and_pauses(words, utterance_start=24.42, utterance_end=29.10)
        self.assertAlmostEqual(cls_res["effective_content_start"], 25.34, places=2)
        self.assertGreater(cls_res["leading_filler_skip_seconds"], 0.85)

    def test_12_preset_tts_api_accepts_both_voice_and_profile_without_mismatch(self):
        """12. Preset TTS APIの引数・実装不整合がないこと"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_dir = root / "snap"
            (snap_dir / "speech_tokenizer").mkdir(parents=True)
            (snap_dir / "config.json").write_text(
                json.dumps({"tts_model_type": "custom_voice"}),
                encoding="utf-8",
            )
            (snap_dir / "speech_tokenizer" / "config.json").write_text("{}", encoding="utf-8")
            (snap_dir / "model.safetensors").write_bytes(b"weights")
            (snap_dir / "speech_tokenizer" / "model.safetensors").write_bytes(b"tok_weights")

            fake_pkg = root / "qwen_tts_mlx"
            fake_pkg.mkdir(parents=True)
            init_py = fake_pkg / "__init__.py"
            init_py.write_text("# mock\n", encoding="utf-8")

            generate_calls = []

            class DummyNativeCustomVoice:
                def generate_custom_voice(self, **kwargs):
                    generate_calls.append(kwargs)
                    return [
                        types.SimpleNamespace(
                            audio=np.sin(np.linspace(0, 20, 4800)).astype(np.float32) * 0.25,
                            sample_rate=24000,
                            token_count=15,
                            processing_time_seconds=0.12,
                        )
                    ]

            class DummyMLXQwen3TTS:
                def __init__(self):
                    self.model = DummyNativeCustomVoice()

                @classmethod
                def from_pretrained(cls, _path):
                    return cls()

            fake_qwen_mod = types.ModuleType("qwen_tts_mlx")
            fake_qwen_mod.MLXQwen3TTS = DummyMLXQwen3TTS

            fake_mx = types.ModuleType("mlx.core")
            fake_mx.random = types.SimpleNamespace(seed=lambda s: None)
            fake_mx.get_peak_memory = lambda: 512
            fake_mlx_pkg = types.ModuleType("mlx")
            fake_mlx_pkg.core = fake_mx

            fake_sf = types.ModuleType("soundfile")
            fake_sf.write = lambda p, arr, sr, subtype="PCM_16", format="WAV": _write_test_speech_wav(
                p, duration=len(arr) / float(sr), rate=sr
            )

            fake_hf = types.ModuleType("huggingface_hub")
            fake_hf.snapshot_download = lambda *a, **kw: str(snap_dir)

            fake_spec = types.SimpleNamespace(origin=str(init_py))
            with mock.patch.dict(
                sys.modules,
                {
                    "qwen_tts_mlx": fake_qwen_mod,
                    "mlx": fake_mlx_pkg,
                    "mlx.core": fake_mx,
                    "soundfile": fake_sf,
                    "huggingface_hub": fake_hf,
                },
            ), mock.patch("importlib.util.find_spec", return_value=fake_spec), mock.patch(
                "importlib.metadata.version", return_value="0.1.0"
            ):
                from language.voice import VoiceProfile

                synth = tg.QwenPresetSynthesizer(seed=0)
                # Call 1: voice="Ono_Anna" (as called by dubbing_pipeline.py)
                out1 = root / "preset_voice_arg.wav"
                res1 = synth.synthesize(text="こんにちは。", voice="Ono_Anna", output_path=out1)
                self.assertTrue(out1.is_file())
                self.assertEqual(res1["preset_voice"], "Ono_Anna")
                self.assertEqual(generate_calls[-1]["speaker"], "Ono_Anna")

                # Call 2: profile=VoiceProfile(...) (legacy caller)
                prof_obj = VoiceProfile(speaker="SPEAKER_01", mode="preset", voice="Uncle_Fu")
                out2 = root / "preset_profile_arg.wav"
                res2 = synth.synthesize(text="こんにちは。", profile=prof_obj, output_path=out2)
                self.assertTrue(out2.is_file())
                self.assertEqual(res2["preset_voice"], "Uncle_Fu")
                self.assertEqual(generate_calls[-1]["speaker"], "Uncle_Fu")



    def test_post_tts_auto_extend_clip_handles_clone_and_preset_utterance_61(self):
        """Regression test for Utterance 61 clip boundary overflow in Clone (8.16s -> 1874.66) and Preset (8.24s -> 1874.74)."""
        import tempfile
        import wave
        import numpy as np
        from dubbing_pipeline import (
            _fallback_mix_audio_and_video,
            analyze_timeline_collisions_and_overlaps,
            ensure_30s_verification_video,
            resolve_post_tts_clip_extension,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            sr = 24000
            for label, wav_dur, expected_end in (
                ("clone", 8.16, 1874.66),
                ("preset", 8.24, 1874.74),
            ):
                wav_path = tmp_dir / f"utt61_{label}.wav"
                n_samples = int(round(wav_dur * sr))
                t = np.arange(n_samples, dtype=np.float32) / float(sr)
                sig = (0.25 * np.sin(2.0 * np.pi * 135.0 * t) * 32767.0).astype("<i2")
                with wave.open(str(wav_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(sig.tobytes())

                entries = [
                    {
                        "id": "61",
                        "speaker": "SPEAKER_01",
                        "start": 1866.50,
                        "end": 1874.30,
                        "source_start": 1866.50,
                        "source_end": 1874.30,
                        "text_ja": "ロボットはいわばエンドエフェクターで、局所的な知能を持ちつつ大規模AIモデルに管理される形になる。",
                        "audio_file": str(wav_path),
                        "generation_status": "complete",
                    }
                ]
                tl = analyze_timeline_collisions_and_overlaps(
                    entries,
                    selection_start=1836.45,
                    selection_end=1874.65,
                    auto_tempo_adjust=True,
                    min_atempo=1.0,
                    max_atempo=1.20,
                    allow_slowdown=False,
                    smooth_intra_speaker_rates=True,
                    adjusted_wav_dir=tmp_dir / f"adj_{label}",
                    auto_extend_clip=True,
                    max_clip_extension_seconds=12.0,
                    source_video_duration=1915.0,
                )
                ext = resolve_post_tts_clip_extension(
                    feedback_entries=entries,
                    placements_map=tl["placements"],
                    selection_start=1836.45,
                    selection_end=1874.65,
                    requested_selection_end=1871.0,
                    auto_extend_clip=True,
                    max_clip_extension_seconds=12.0,
                    source_video_duration=1915.0,
                )
                self.assertTrue(ext["extension_applied"])
                self.assertAlmostEqual(ext["effective_selection_end"], expected_end, places=2)
                self.assertAlmostEqual(tl["placements"]["61"]["applied_atempo"], 1.0, places=3)

                vid_path = ensure_30s_verification_video(
                    tmp_dir / f"src_{label}.mp4",
                    duration_seconds=ext["effective_clip_duration"],
                )
                _, mix_rep = _fallback_mix_audio_and_video(
                    video_path=vid_path,
                    feedback_doc={"entries": entries},
                    placements_map=tl["placements"],
                    selection_start=1836.45,
                    clip_duration=ext["effective_clip_duration"],
                    output_dir=tmp_dir / f"mix_{label}",
                )
                self.assertAlmostEqual(
                    mix_rep["clip_duration_seconds"],
                    round(expected_end - 1836.45, 4),
                    places=2,
                )

    def test_post_tts_auto_extend_clip_raises_when_exceeding_source_video_duration(self):
        """Extending beyond source_video_duration must raise a clear ValueError."""
        from dubbing_pipeline import resolve_post_tts_clip_extension

        entries = [{"id": "61", "generation_status": "complete"}]
        placements = {
            "61": {
                "placement_start_seconds": 1866.50,
                "placement_end_seconds": 1874.74,
                "clip_relative_start_seconds": 30.05,
                "clip_relative_end_seconds": 38.29,
            }
        }
        with self.assertRaisesRegex(ValueError, "exceeds source video duration"):
            resolve_post_tts_clip_extension(
                feedback_entries=entries,
                placements_map=placements,
                selection_start=1836.45,
                selection_end=1874.65,
                auto_extend_clip=True,
                source_video_duration=1874.70,
            )

    def test_post_tts_no_auto_extend_clip_refuses_truncation(self):
        """When auto_extend_clip=False, overflowing TTS WAV must refuse truncation with ValueError."""
        from dubbing_pipeline import resolve_post_tts_clip_extension

        entries = [{"id": "61", "generation_status": "complete"}]
        placements = {
            "61": {
                "placement_start_seconds": 1866.50,
                "placement_end_seconds": 1874.74,
                "clip_relative_start_seconds": 30.05,
                "clip_relative_end_seconds": 38.29,
            }
        }
        with self.assertRaisesRegex(ValueError, "Full WAV for utterance 61 exceeds the selected clip; truncation refused"):
            resolve_post_tts_clip_extension(
                feedback_entries=entries,
                placements_map=placements,
                selection_start=1836.45,
                selection_end=1874.65,
                auto_extend_clip=False,
                source_video_duration=1915.0,
            )

if __name__ == "__main__":
    unittest.main()
