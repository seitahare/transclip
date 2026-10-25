"""Unit and End-to-End integration tests for Phase 11 & Phase 12 Japanese Dubbing Pipeline (`dubbing_pipeline.py`)."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import dubbing_pipeline as dp
import prosody_experiments as pe
from tests.test_prosody_experiments import _build_approved_consent_and_review, _write_test_wav


class TestPhase11And12DubbingPipeline(unittest.TestCase):
    # =========================================================================
    # Phase 11 Backward-Compatibility Tests (5 tests)
    # =========================================================================
    def test_voice_plan_defaults_approved_profile_to_xvector_and_supports_icl_and_preset_cli(self):
        script_doc = {
            "selection_range": {"start": 0.0, "end": 30.0},
            "utterances": [
                {"id": "dub-001", "speaker": "SYNTH_EN_01", "text_ja": "テスト1", "preset_voice": "Ryan"},
            ],
        }
        fake_vc = types.ModuleType("voice_cloning")
        fake_vc.profile_snapshot = lambda prof: {
            "profile_fingerprint": "fp123456",
            "reference_audio_sha256": "audsha",
            "reference_text_sha256": "txtsha",
        }
        profiles = {"SYNTH_EN_01": types.SimpleNamespace(speaker_id="SYNTH_EN_01")}

        with mock.patch.dict(sys.modules, {"voice_cloning": fake_vc}):
            # 1. Default `auto` -> `approved_clone:fp123456` with `xvector`
            plan_auto = dp.resolve_speaker_voice_plan(
                script_doc,
                voice_profiles=profiles,
                voice_mode="auto",
                clone_method="xvector",
            )
            self.assertEqual(plan_auto["voices"], {"SYNTH_EN_01": "approved_clone:fp123456"})
            self.assertEqual(plan_auto["effective_clone_method"], "xvector")
            self.assertTrue(plan_auto["uses_clone"])
            self.assertFalse(plan_auto["uses_preset"])
            self.assertTrue(plan_auto["speaker_routing_details"][0]["x_vector_only_mode"])

            # 2. `voice_mode="icl"` -> `approved_clone:fp123456` with `icl`
            plan_icl = dp.resolve_speaker_voice_plan(
                script_doc,
                voice_profiles=profiles,
                voice_mode="icl",
            )
            self.assertEqual(plan_icl["voices"], {"SYNTH_EN_01": "approved_clone:fp123456"})
            self.assertEqual(plan_icl["effective_clone_method"], "icl")
            self.assertFalse(plan_icl["speaker_routing_details"][0]["x_vector_only_mode"])

            # 3. `voice_mode="preset"` -> Preset `"Ryan"`
            plan_preset = dp.resolve_speaker_voice_plan(
                script_doc,
                voice_profiles=profiles,
                voice_mode="preset",
                default_preset_voice="Ryan",
            )
            self.assertEqual(plan_preset["voices"], {"SYNTH_EN_01": "Ryan"})
            self.assertIsNone(plan_preset["voice_profiles"])
            self.assertFalse(plan_preset["uses_clone"])
            self.assertTrue(plan_preset["uses_preset"])

    def test_unapproved_real_speaker_is_never_cloned_and_requires_explicit_preset(self):
        script_doc = {
            "selection_range": {"start": 0.0, "end": 30.0},
            "utterances": [
                {"id": "dub-001", "speaker": "SYNTH_EN_01", "text_ja": "承認済み合成話者のセリフです。"},
                {"id": "dub-002", "speaker": "REAL_HOST_01", "text_ja": "未承認の実在話者のセリフです。"},
            ],
        }
        fake_vc = types.ModuleType("voice_cloning")
        fake_vc.profile_snapshot = lambda prof: {
            "profile_fingerprint": "fp_synth_01",
            "reference_audio_sha256": "audsha",
            "reference_text_sha256": "txtsha",
        }
        profiles = {"SYNTH_EN_01": types.SimpleNamespace(speaker_id="SYNTH_EN_01")}

        with mock.patch.dict(sys.modules, {"voice_cloning": fake_vc}):
            with self.assertRaises(dp.UnauthorizedSpeakerError):
                dp.resolve_speaker_voice_plan(
                    script_doc,
                    voice_profiles=profiles,
                    voice_mode="auto",
                )

            plan = dp.resolve_speaker_voice_plan(
                script_doc,
                voice_profiles=profiles,
                voice_mode="auto",
                clone_method="xvector",
                preset_speaker_map=dp.parse_preset_speaker_mappings(["REAL_HOST_01=Ryan"]),
            )
            self.assertEqual(
                plan["voices"],
                {
                    "SYNTH_EN_01": "approved_clone:fp_synth_01",
                    "REAL_HOST_01": "Ryan",
                },
            )
            self.assertTrue(plan["uses_clone"])
            self.assertTrue(plan["uses_preset"])
            self.assertEqual(list(plan["voice_profiles"].keys()), ["SYNTH_EN_01"])

            unknown_script = {
                "selection_range": {"start": 0.0, "end": 30.0},
                "utterances": [{"id": "dub-001", "speaker": "UNKNOWN", "text_ja": "不明話者"}],
            }
            with self.assertRaises(dp.UnauthorizedSpeakerError):
                dp.resolve_speaker_voice_plan(unknown_script, voice_profiles=profiles, default_preset_voice="Ryan")

    def test_same_speaker_collision_detection_and_natural_multi_speaker_overlap_preservation(self):
        entries = [
            {
                "id": "dub-001",
                "speaker": "SYNTH_EN_01",
                "text_ja": "最初のセリフです。",
                "mora_count": 9,
                "start": 2.0,
                "end": 5.0,
                "actual_tts_duration_seconds": 3.30,
            },
            {
                "id": "dub-002",
                "speaker": "SYNTH_EN_01",
                "text_ja": "二番目のセリフです。",
                "mora_count": 10,
                "start": 5.0,
                "end": 8.0,
                "actual_tts_duration_seconds": 2.50,
            },
            {
                "id": "dub-003",
                "speaker": "REAL_HOST_01",
                "text_ja": "相槌のコメントです。",
                "mora_count": 10,
                "start": 6.0,
                "end": 7.5,
                "actual_tts_duration_seconds": 1.40,
            },
        ]
        res = dp.analyze_timeline_collisions_and_overlaps(
            entries,
            selection_start=0.0,
            selection_end=30.0,
            max_timing_shift=0.5,
        )
        self.assertEqual(len(res["initial_same_speaker_collisions"]), 1)
        self.assertEqual(len(res["resolved_same_speaker_collisions"]), 1)
        self.assertEqual(len(res["unresolved_same_speaker_collisions"]), 0)
        self.assertAlmostEqual(res["placements"]["dub-002"]["placement_start_seconds"], 5.30, places=3)
        self.assertAlmostEqual(res["placements"]["dub-003"]["placement_start_seconds"], 6.00, places=3)
        self.assertEqual(len(res["preserved_other_speaker_overlaps"]), 1)
        self.assertEqual(res["preserved_other_speaker_overlaps"][0]["first_speaker"], "SYNTH_EN_01")
        self.assertEqual(res["preserved_other_speaker_overlaps"][0]["second_speaker"], "REAL_HOST_01")

        severe_entries = [
            {
                "id": "dub-001",
                "speaker": "SYNTH_EN_01",
                "text_ja": "非常に長い日本語の翻訳文がここに入ります。",
                "mora_count": 45,
                "start": 2.0,
                "end": 6.0,
                "actual_tts_duration_seconds": 9.30,
            },
            {
                "id": "dub-002",
                "speaker": "SYNTH_EN_01",
                "text_ja": "次の文です。",
                "mora_count": 6,
                "start": 6.0,
                "end": 9.0,
                "actual_tts_duration_seconds": 1.50,
            },
        ]
        severe_res = dp.analyze_timeline_collisions_and_overlaps(
            severe_entries,
            selection_start=0.0,
            selection_end=30.0,
            max_timing_shift=0.5,
        )
        self.assertEqual(len(severe_res["unresolved_same_speaker_collisions"]), 1)
        unres = severe_res["unresolved_same_speaker_collisions"][0]
        self.assertGreater(unres["required_reduction_seconds"], 4.5)
        self.assertLess(unres["target_mora_budget"], unres["current_mora_count"])

    def test_30s_e2e_cli_renders_mp4_subtitles_and_unifies_clip_relative_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_dir = root / "snapshot"
            snap_dir.mkdir(parents=True)
            (snap_dir / "config.json").write_text(json.dumps({"quantization": {"bits": 8}}), encoding="utf-8")

            fake_pkg = root / "mlx_audio"
            (fake_pkg / "tts" / "models" / "qwen3_tts").mkdir(parents=True)
            (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
            (fake_pkg / "tts" / "models" / "qwen3_tts" / "qwen3_tts.py").write_text("# mock native\n", encoding="utf-8")

            lang_dir = Path(pe.__file__).parent / "language"
            created_lang = False
            created_voice = False
            voice_file = lang_dir / "voice.py"
            if not lang_dir.exists():
                lang_dir.mkdir(parents=True)
                created_lang = True
            if not voice_file.exists():
                voice_file.write_text("# mock voice contract\n", encoding="utf-8")
                created_voice = True

            ref_wav = _write_test_wav(root / "inputs" / "reference.wav", duration=12.48, rate=24000, f0_start=132.0, f0_end=126.0)
            ref_txt = root / "inputs" / "reference.txt"
            ref_txt.write_text("English synthetic reference transcript for SYNTH_EN_01.\n", encoding="utf-8")
            consent_json, review_json, ref_sha, text_sha = _build_approved_consent_and_review(
                root / "inputs",
                speaker="SYNTH_EN_01",
                ref_wav=ref_wav,
                ref_txt=ref_txt,
            )
            expected_fp = hashlib.sha256(f"SYNTH_EN_01:{ref_sha}:{text_sha}".encode("utf-8")).hexdigest()

            script_path = root / "inputs" / "dubbing_script_1880_1910.json"
            script_doc = {
                "schema_version": 1,
                "selection_range": {"start": 1880.0, "end": 1910.0},
                "utterances": [
                    {
                        "id": "dub-001",
                        "speaker": "SYNTH_EN_01",
                        "text_en": pe.CORE_SAMPLES[0]["text_en"],
                        "text_ja": pe.CORE_SAMPLES[0]["text_ja"],
                        "mora_count": pe.CORE_SAMPLES[0]["mora_count"],
                        "start": 1882.0,
                        "end": 1887.5,
                    },
                    {
                        "id": "dub-002",
                        "speaker": "SYNTH_EN_01",
                        "text_en": pe.CORE_SAMPLES[1]["text_en"],
                        "text_ja": pe.CORE_SAMPLES[1]["text_ja"],
                        "mora_count": pe.CORE_SAMPLES[1]["mora_count"],
                        "start": 1890.0,
                        "end": 1896.5,
                    },
                    {
                        "id": "dub-003",
                        "speaker": "REAL_HOST_01",
                        "text_en": "Indeed, let us check the next experiment.",
                        "text_ja": "ええ、次の実験で詳しく確かめてみましょう。",
                        "mora_count": 21,
                        "start": 1891.0,
                        "end": 1895.0,
                    },
                    {
                        "id": "dub-004",
                        "speaker": "SYNTH_EN_01",
                        "text_en": pe.CORE_SAMPLES[2]["text_en"],
                        "text_ja": pe.CORE_SAMPLES[2]["text_ja"],
                        "mora_count": pe.CORE_SAMPLES[2]["mora_count"],
                        "start": 1901.0,
                        "end": 1907.5,
                    },
                ],
            }
            script_path.write_text(json.dumps(script_doc, ensure_ascii=False, indent=2), encoding="utf-8")

            class CloneBlockedForReviewError(RuntimeError):
                pass

            def strict_audit_english_reference(*, speaker, reference_audio, reference_text_path, consent_path, review_path, output_dir, generation_record_path=None):
                out_d = Path(output_dir)
                out_d.mkdir(parents=True, exist_ok=True)
                rep = {
                    "speaker": speaker,
                    "status": "verified_ready_for_clone",
                    "ready_for_clone_synthesis": True,
                    "mechanical_passed": True,
                    "pending_human_items": [],
                }
                rep_p = out_d / "audit_report.json"
                rep_p.write_text(json.dumps(rep), encoding="utf-8")
                return rep_p, rep

            def strict_register_profile(*, speaker, reference_audio, reference_text_path, consent_path, review_path, output_dir, preset_voice="Ryan", generation_record_path=None):
                reg_d = Path(output_dir)
                reg_d.mkdir(parents=True, exist_ok=True)
                reg_p = reg_d / "voice_profiles.json"
                reg_doc = {
                    "profiles": {
                        speaker: {
                            "speaker_id": speaker,
                            "reference_audio": str(reference_audio),
                            "reference_text": Path(reference_text_path).read_text(encoding="utf-8").strip(),
                            "preset_voice": preset_voice,
                            "profile_fingerprint": expected_fp,
                        }
                    }
                }
                reg_p.write_text(json.dumps(reg_doc), encoding="utf-8")
                return reg_p, reg_doc

            def strict_load_profiles(registry_path):
                raw = json.loads(Path(registry_path).read_text(encoding="utf-8"))["profiles"]
                return {
                    k: types.SimpleNamespace(
                        speaker_id=v["speaker_id"],
                        reference_audio=v["reference_audio"],
                        reference_text=v["reference_text"],
                        preset_voice=v["preset_voice"],
                        profile_fingerprint=v["profile_fingerprint"],
                    )
                    for k, v in raw.items()
                }

            def strict_profile_snapshot(prof):
                return {
                    "profile_fingerprint": expected_fp,
                    "reference_audio_sha256": ref_sha,
                    "reference_text_sha256": text_sha,
                }

            def strict_run_feedback(script_p, output_d, *, backend=None, voices=None, voice_profiles=None):
                self.assertIsNotNone(backend)
                self.assertEqual(
                    voices,
                    {
                        "SYNTH_EN_01": f"approved_clone:{expected_fp}",
                        "REAL_HOST_01": "Ryan",
                    },
                )
                self.assertIsNotNone(voice_profiles)
                self.assertIn("SYNTH_EN_01", voice_profiles)
                self.assertNotIn("REAL_HOST_01", voice_profiles)

                s_doc = json.loads(Path(script_p).read_text(encoding="utf-8"))
                out_d = Path(output_d)
                out_d.mkdir(parents=True, exist_ok=True)
                entries = []
                for u in s_doc["utterances"]:
                    spk = u["speaker"]
                    v_spec = voices[spk]
                    wav_out = out_d / "audio" / f"{u['id']}.wav"
                    wav_out.parent.mkdir(parents=True, exist_ok=True)
                    if v_spec.startswith("approved_clone:"):
                        res = backend.synthesize(text=u["text_ja"], profile=voice_profiles[spk], output_path=wav_out)
                    else:
                        res = backend.synthesize(text=u["text_ja"], voice=v_spec, output_path=wav_out)
                    samples, sr = pe._read_wav_float32(wav_out)
                    dur = len(samples) / float(sr)
                    entries.append(
                        {
                            "id": u["id"],
                            "speaker": spk,
                            "voice": v_spec,
                            "text_en": u["text_en"],
                            "text_ja": u["text_ja"],
                            "mora_count": u["mora_count"],
                            "start": u["start"],
                            "end": u["end"],
                            "placement_seconds": u["start"],
                            "actual_tts_duration_seconds": round(dur, 4),
                            "generation_status": "complete",
                            "audio_file": str(wav_out),
                            "cache_reused": False,
                            "backend_result": res,
                            "tts": dict(backend.identity),
                        }
                    )
                fb_doc = {
                    "selection_range": s_doc["selection_range"],
                    "tts": dict(backend.identity),
                    "entries": entries,
                }
                fb_p = out_d / "feedback.json"
                fb_p.write_text(json.dumps(fb_doc, ensure_ascii=False, indent=2), encoding="utf-8")
                return fb_p, fb_doc

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
                        "model_revision": "e7dd0585652209fa0d7783659aad4e8a324de11c",
                        "settings": dict(self.settings),
                    }

                def synthesize(self, *, text, voice, output_path):
                    _write_test_wav(output_path, duration=2.2, rate=24000, f0_start=125.0, f0_end=118.0)
                    return {
                        "sample_rate": 24000,
                        "sample_frames": 52800,
                        "token_count": 20,
                        "termination": "eos_before_limit",
                        "processing_seconds": 0.30,
                        "clone_method": "preset",
                        "x_vector_only_mode": False,
                    }

            class DummyNative:
                def __init__(self):
                    self._icl_cache = {}

                def generate(self, **kwargs):
                    if kwargs.get("ref_text") is not None:
                        raise AssertionError("X-vector mode must pass ref_text=None")
                    frames = 54000
                    t = np.arange(frames, dtype=np.float64) / 24000.0
                    sig = (0.25 * np.sin(2.0 * np.pi * 140.0 * t)).astype(np.float32)
                    return [types.SimpleNamespace(audio=sig, sample_rate=24000, token_count=24)]

            class DummyMLXWrapper:
                def __init__(self):
                    self.model = DummyNative()

                @classmethod
                def from_pretrained(cls, _path):
                    return cls()

                def create_voice_clone_prompt(self, **kwargs):
                    return types.SimpleNamespace(
                        ref_audio=kwargs["ref_audio"][0],
                        ref_text=None if kwargs.get("x_vector_only_mode") else kwargs.get("ref_text"),
                        x_vector_only_mode=bool(kwargs.get("x_vector_only_mode")),
                    )

            fake_clc = types.ModuleType("cross_lingual_clone")
            fake_clc.CloneBlockedForReviewError = CloneBlockedForReviewError
            fake_clc.audit_english_reference = strict_audit_english_reference
            fake_clc.build_phase8_benchmark_script = lambda output_path, **kw: (Path(output_path), script_doc)

            fake_vc = types.ModuleType("voice_cloning")
            fake_vc.register_profile = strict_register_profile
            fake_vc.load_profiles = strict_load_profiles
            fake_vc.digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            fake_vc.profile_snapshot = strict_profile_snapshot

            fake_fb = types.ModuleType("tts_feedback")
            fake_fb.run_feedback = strict_run_feedback

            fake_tts_gen = types.ModuleType("tts_generator")
            fake_tts_gen.QwenPresetSynthesizer = FakePresetSynthesizer

            fake_mx = types.ModuleType("mlx.core")
            fake_mx.random = types.SimpleNamespace(seed=lambda s: None)
            fake_mx.get_peak_memory = lambda: 2048
            fake_mlx_pkg = types.ModuleType("mlx")
            fake_mlx_pkg.core = fake_mx

            fake_sf = types.ModuleType("soundfile")
            fake_sf.read = lambda p, dtype="float32": pe._read_wav_float32(p)
            fake_sf.write = lambda p, arr, sr, subtype="PCM_16", format="WAV": _write_test_wav(p, duration=len(arr) / float(sr), rate=sr)

            fake_qwen_mlx = types.ModuleType("qwen_tts_mlx")
            fake_qwen_mlx.MLXQwen3TTS = DummyMLXWrapper

            fake_spec = types.SimpleNamespace(origin=str(fake_pkg / "__init__.py"))

            out_dir = root / "output" / "phase11_e2e"
            try:
                with mock.patch.dict(
                    sys.modules,
                    {
                        "cross_lingual_clone": fake_clc,
                        "voice_cloning": fake_vc,
                        "tts_feedback": fake_fb,
                        "tts_generator": fake_tts_gen,
                        "mlx": fake_mlx_pkg,
                        "mlx.core": fake_mx,
                        "soundfile": fake_sf,
                        "qwen_tts_mlx": fake_qwen_mlx,
                    },
                ), mock.patch("importlib.util.find_spec", return_value=fake_spec):
                    rc = dp.main(
                        [
                            "run",
                            "--script",
                            str(script_path),
                            "--speaker",
                            "SYNTH_EN_01",
                            "--reference-audio",
                            str(ref_wav),
                            "--reference-text",
                            str(ref_txt),
                            "--consent",
                            str(consent_json),
                            "--review",
                            str(review_json),
                            "--voice-mode",
                            "auto",
                            "--clone-method",
                            "xvector",
                            "--preset-speaker",
                            "REAL_HOST_01=Ryan",
                            "--output",
                            str(out_dir),
                        ]
                    )
                    self.assertEqual(rc, 0)

                    self.assertTrue((out_dir / "final_ja.mp4").is_file())
                    self.assertTrue((out_dir / "final_ja_no_subtitles.mp4").is_file())
                    self.assertTrue((out_dir / "final_ja.srt").is_file())
                    self.assertTrue((out_dir / "final_ja.ass").is_file())
                    self.assertTrue((out_dir / "dubbed_audio.wav").is_file())
                    self.assertTrue((out_dir / "pipeline_report.json").is_file())
                    self.assertTrue((out_dir / "pipeline_report.md").is_file())

                    report = json.loads((out_dir / "pipeline_report.json").read_text(encoding="utf-8"))
                    self.assertEqual(report["status"], "completed_diagnostic")
                    self.assertEqual(report["completion"]["video_generation"]["status"], "succeeded")
                    self.assertEqual(report["completion"]["quality_review"]["status"], "pending")
                    self.assertFalse(report["completion"]["completed_clean"])
                    self.assertTrue(report["media_verification"]["has_video"])
                    self.assertTrue(report["media_verification"]["has_audio"])
                    self.assertAlmostEqual(report["media_verification"]["duration_seconds"], 30.0, delta=0.25)
                    self.assertTrue(report["media_verification"]["relative_time_synchronized"])
                    self.assertAlmostEqual(report["media_verification"]["first_subtitle_cue_start_seconds"], 2.0, places=2)
                    srt_content = (out_dir / "final_ja.srt").read_text(encoding="utf-8")
                    self.assertIn("00:00:02,000 -->", srt_content)
                    self.assertNotIn("00:31:", srt_content)

                    overlaps = report["collision_and_overlap_analysis"]["preserved_other_speaker_overlaps"]
                    self.assertEqual(len(overlaps), 1)
            finally:
                if created_voice and voice_file.exists():
                    voice_file.unlink()
                if created_lang and lang_dir.exists():
                    lang_dir.rmdir()

    def test_live_repository_phase11_integration_when_present(self):
        try:
            import audio_mixer
            import cross_lingual_clone
            import japanese_subtitles
            import timing_adjustment
            import tts_feedback
            import voice_cloning
        except ImportError:
            return

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_dir = root / "snapshot"
            snap_dir.mkdir(parents=True)
            (snap_dir / "config.json").write_text(json.dumps({"quantization": {"bits": 8}}), encoding="utf-8")

            fake_pkg = root / "mlx_audio"
            (fake_pkg / "tts" / "models" / "qwen3_tts").mkdir(parents=True)
            (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
            (fake_pkg / "tts" / "models" / "qwen3_tts" / "qwen3_tts.py").write_text("# mock native\n", encoding="utf-8")

            ref_wav = _write_test_wav(root / "inputs" / "reference.wav", duration=12.48, rate=24000, f0_start=132.0, f0_end=126.0)
            ref_txt = root / "inputs" / "reference.txt"
            ref_txt.write_text("New technology is gradually changing the way we live, and we verify every step carefully.\n", encoding="utf-8")
            consent_json, review_json, _, _ = _build_approved_consent_and_review(
                root / "inputs",
                speaker="SYNTH_EN_01",
                ref_wav=ref_wav,
                ref_txt=ref_txt,
            )

            class DummyNative:
                def __init__(self):
                    self._icl_cache = {}

                def generate(self, **kwargs):
                    if kwargs.get("ref_text") is not None:
                        raise AssertionError("X-vector mode must pass ref_text=None")
                    frames = 48000
                    t = np.arange(frames, dtype=np.float64) / 24000.0
                    sig = (0.25 * np.sin(2.0 * np.pi * 138.0 * t)).astype(np.float32)
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
                        ref_text=None if kwargs.get("x_vector_only_mode") else kwargs.get("ref_text"),
                        x_vector_only_mode=bool(kwargs.get("x_vector_only_mode")),
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
                        "model_revision": "e7dd0585652209fa0d7783659aad4e8a324de11c",
                        "model_type": expected_model_type or "custom_voice",
                        "voice_mode": "preset",
                        "settings": dict(self.settings),
                    }

                def synthesize(self, *, text, voice, output_path):
                    _write_test_wav(output_path, duration=2.0, rate=24000, f0_start=135.0, f0_end=142.0)
                    return {
                        "sample_rate": 24000,
                        "sample_frames": 48000,
                        "token_count": 20,
                        "termination": "eos_before_limit",
                        "processing_seconds": 0.30,
                    }

            fake_mx = types.ModuleType("mlx.core")
            fake_mx.random = types.SimpleNamespace(seed=lambda s: None)
            fake_mx.get_peak_memory = lambda: 2048
            fake_mlx_pkg = types.ModuleType("mlx")
            fake_mlx_pkg.core = fake_mx

            fake_sf = types.ModuleType("soundfile")
            fake_sf.read = lambda p, dtype="float32": pe._read_wav_float32(p)
            fake_sf.write = lambda p, arr, sr, subtype="PCM_16", format="WAV": _write_test_wav(p, duration=len(arr) / float(sr), rate=sr)

            fake_qwen_mlx = types.ModuleType("qwen_tts_mlx")
            fake_qwen_mlx.MLXQwen3TTS = DummyMLXWrapper

            fake_spec = types.SimpleNamespace(origin=str(fake_pkg / "__init__.py"))

            out_dir = root / "live_phase11_out"
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
                rc = dp.main(
                    [
                        "run-synth-30s",
                        "--speaker",
                        "SYNTH_EN_01",
                        "--reference-audio",
                        str(ref_wav),
                        "--reference-text",
                        str(ref_txt),
                        "--consent",
                        str(consent_json),
                        "--review",
                        str(review_json),
                        "--voice-mode",
                        "xvector",
                        "--output",
                        str(out_dir),
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue((out_dir / "final_ja.mp4").is_file())
                self.assertTrue((out_dir / "final_ja.srt").is_file())
                self.assertTrue((out_dir / "dubbed_audio.wav").is_file())
                self.assertTrue((out_dir / "pipeline_report.json").is_file())

    # =========================================================================
    # Phase 12 Unit & E2E Integration Tests (6 tests)
    # =========================================================================
    def test_phase12_xvector_primary_never_hides_failure_with_preset_fallback(self):
        """Requirement 1: X-vector synthesis is the primary mode and never falls back to Preset on missing ref or failure."""
        script_doc = {
            "selection_range": {"start": 1885.0, "end": 1915.0},
            "utterances": [
                {
                    "id": 61,
                    "speaker": "SPEAKER_00",
                    "text_en": "Testing X-vector strict mode.",
                    "text_ja": "X-vector厳格モードのテストです。",
                    "preset_voice": "Ryan",
                }
            ],
        }
        # Even if `default_preset_voice="Ryan"` and `utterance["preset_voice"]="Ryan"` are present,
        # `voice_mode="xvector"` / `allow_preset_fallback=False` MUST raise `XVectorSynthesisError` when reference is missing!
        with self.assertRaises(dp.XVectorSynthesisError):
            dp.resolve_speaker_voice_plan(
                script_doc,
                voice_profiles={},
                technical_references={},
                voice_mode="xvector",
                default_preset_voice="Ryan",
                allow_preset_fallback=False,
            )

        # Also verify that runtime synthesis error inside X-vector backend raises `XVectorSynthesisError` instead of switching to Preset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_wav = _write_test_wav(root / "spk00.wav", duration=3.5, rate=24000)
            tref = dp.TechnicalVoiceReference(speaker_id="SPEAKER_00", reference_audio=ref_wav)
            plan = dp.resolve_speaker_voice_plan(
                script_doc,
                technical_references={"SPEAKER_00": tref},
                voice_mode="xvector",
                allow_preset_fallback=False,
            )

            class FailingXVectorBackend:
                identity = {"backend": "qwen_tts_mlx", "clone_method": "xvector", "x_vector_only_mode": True}

                def synthesize(self, **kwargs):
                    raise RuntimeError("Simulated X-vector model conditioning error")

            with self.assertRaises(dp.XVectorSynthesisError):
                dp.run_technical_xvector_feedback(
                    utterances=[
                        {
                            "id": "61",
                            "raw_id": 61,
                            "speaker": "SPEAKER_00",
                            "text_en": "Testing.",
                            "text_ja": "テストです。",
                            "mora_count": 6,
                            "start": 1886.0,
                            "end": 1888.5,
                            "source_start": 1886.0,
                            "source_end": 1888.5,
                        }
                    ],
                    selection_range={"start": 1885.0, "end": 1915.0},
                    output_dir=root / "fb_out",
                    backend=FailingXVectorBackend(),
                    voice_plan=plan,
                )

    def test_phase12_separation_of_technical_synthesis_and_rights_governance(self):
        """Requirement 2: Separate technical synthesis processing from rights verification & publication governance.
        Eliminate redundant registration steps for synthetic/permitted voices while blocking unpermitted real-person cloning.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_synth = _write_test_wav(root / "synth_en_01.wav", duration=4.0, rate=24000)
            ref_perm = _write_test_wav(root / "speaker_00.wav", duration=4.2, rate=24000)
            ref_unperm = _write_test_wav(root / "celebrity.wav", duration=4.0, rate=24000)

            # 1. TechnicalVoiceReference requires ONLY the valid WAV file (no consent.json, review.json, or reference.txt needed)
            tref_synth = dp.TechnicalVoiceReference(speaker_id="SYNTH_EN_01", reference_audio=ref_synth, source_origin="explicit_speaker_ref")
            tref_perm = dp.TechnicalVoiceReference(speaker_id="SPEAKER_00", reference_audio=ref_perm, source_origin="source_video_extracted")
            tref_unperm = dp.TechnicalVoiceReference(speaker_id="REAL_CELEB_01", reference_audio=ref_unperm)

            snap = tref_synth.technical_snapshot()
            self.assertEqual(snap["reference_audio_sha256"], hashlib.sha256(ref_synth.read_bytes()).hexdigest())
            self.assertIsNone(snap["reference_text_sha256"])

            # 2. RightsAndPublicationPolicy clears synthetic and permitted local speakers without fabricating consent/review files
            policy = dp.RightsAndPublicationPolicy(
                rights_mode="permitted_local",
                speaker_source_types={
                    "SYNTH_EN_01": "synthetic",
                    "SPEAKER_00": "permitted",
                    "REAL_CELEB_01": "unpermitted_real_person",
                },
            )
            gov_synth = policy.classify_and_verify_speaker("SYNTH_EN_01", has_registered_profile=False)
            self.assertEqual(gov_synth["source_classification"], "synthetic_speaker")
            self.assertTrue(gov_synth["external_publication_allowed"])
            self.assertTrue(gov_synth["registration_ceremony_eliminated"])
            self.assertFalse(gov_synth["authentication_bypass_used"])

            gov_perm = policy.classify_and_verify_speaker("SPEAKER_00", has_registered_profile=False, reference_origin="source_video_extracted")
            self.assertEqual(gov_perm["source_classification"], "permitted_audio_source")
            self.assertTrue(gov_perm["registration_ceremony_eliminated"])
            self.assertFalse(gov_perm["authentication_bypass_used"])

            # 3. Unpermitted real person is strictly blocked by RightsAndPublicationPolicy
            with self.assertRaises(dp.RightsPolicyViolationError):
                policy.classify_and_verify_speaker("REAL_CELEB_01", has_registered_profile=False)

            with self.assertRaises(dp.RightsPolicyViolationError):
                dp.resolve_speaker_voice_plan(
                    {"utterances": [{"id": "1", "speaker": "REAL_CELEB_01", "text_ja": "無断クローン禁止テスト"}]},
                    technical_references={"REAL_CELEB_01": tref_unperm},
                    rights_policy=policy,
                    voice_mode="xvector",
                )

    def test_phase12_detect_and_merge_split_utterances_ids_64_to_66_and_barriers(self):
        """Requirement 3: Detect and merge unnaturally split continuous sentences like utterance IDs 64–66,
        while respecting speaker boundaries, `speaker: null` barriers, and long gaps.
        """
        raw_script = {
            "selection_range": {"start": 1885.0, "end": 1915.0},
            "utterances": [
                {
                    "id": 63,
                    "speaker": "SPEAKER_01",
                    "start": 1885.5,
                    "end": 1888.0,
                    "text_en": "How does the new architecture handle latency?",
                    "text_ja": "新しいアーキテクチャは遅延をどう処理しますか？",
                },
                {
                    "id": 64,
                    "speaker": "SPEAKER_00",
                    "start": 1888.4,
                    "end": 1891.0,
                    "text_en": "Because our pipeline processes 144 streams in parallel,",
                    "text_ja": "私たちのパイプラインは144本のストリームを並列処理し。",
                },
                {
                    "id": 65,
                    "speaker": "SPEAKER_00",
                    "start": 1891.2,
                    "end": 1893.8,
                    "text_en": "and dynamically balances the compute load across nodes,",
                    "text_ja": "さらにノード間で計算負荷を動的に分散するため。",
                },
                {
                    "id": 66,
                    "speaker": "SPEAKER_00",
                    "start": 1894.0,
                    "end": 1897.2,
                    "text_en": "it keeps total latency under 10 milliseconds.",
                    "text_ja": "全体の遅延を10ミリ秒未満に抑えられます。",
                },
                # `speaker: null` barrier between 66 and 68
                {
                    "id": 67,
                    "speaker": None,
                    "start": 1897.5,
                    "end": 1899.0,
                    "text_en": "That is impressive.",
                    "text_ja": "それは素晴らしいですね。",
                },
                {
                    "id": 68,
                    "speaker": "SPEAKER_00",
                    "start": 1899.2,
                    "end": 1902.5,
                    "text_en": "We also verified stability under heavy load.",
                    "text_ja": "高負荷時の安定性も検証しました。",
                },
            ],
        }
        norm = dp.normalize_script_utterances(raw_script)
        merged_res = dp.detect_and_merge_split_utterances(norm["utterances"], enabled=True, max_merge_gap_seconds=1.5)

        self.assertEqual(len(merged_res["merged_groups"]), 1)
        mg = merged_res["merged_groups"][0]
        self.assertEqual(mg["merged_from_ids"], [64, 65, 66])
        self.assertEqual(mg["merged_id"], "64-66")
        self.assertEqual(mg["speaker"], "SPEAKER_00")
        self.assertAlmostEqual(mg["source_start_seconds"], 1888.4, places=2)
        self.assertAlmostEqual(mg["source_end_seconds"], 1897.2, places=2)
        # Numbers (144, 10) must be preserved and intermediate `。` converted to `、` for smooth continuous prosody
        self.assertIn("144本", mg["merged_text_ja"])
        self.assertIn("10ミリ秒", mg["merged_text_ja"])
        self.assertIn("並列処理し、", mg["merged_text_ja"])
        self.assertTrue(mg["merged_text_ja"].endswith("。"))

        # Utterance 67 (`speaker: null`) must NOT be merged into 64-66 or 68, and 68 must NOT bridge across 67
        out_ids = [u["id"] for u in merged_res["utterances"]]
        self.assertEqual(out_ids, ["63", "64-66", "67", "68"])

    def test_phase12_null_speaker_managed_as_verification_target_without_guessing(self):
        """Requirement 4: `speaker: null` utterances are managed as verification targets and never guessed."""
        raw_script = {
            "selection_range": {"start": 1885.0, "end": 1915.0},
            "utterances": [
                {"id": 64, "speaker": "SPEAKER_00", "start": 1886.0, "end": 1890.0, "text_en": "First line.", "text_ja": "最初のセリフです。"},
                {"id": 67, "speaker": None, "start": 1891.0, "end": 1893.0, "text_en": "Unverified speaker line.", "text_ja": "話者未確定のセリフです。"},
                {"id": 68, "speaker": "SPEAKER_00", "start": 1894.0, "end": 1898.0, "text_en": "Third line.", "text_ja": "三番目のセリフです。"},
            ],
        }
        norm = dp.normalize_script_utterances(raw_script)

        # 1. Default `hold_for_verification` holds ID 67 out of synthesis and records surrounding speakers as context only
        part = dp.partition_and_manage_null_speakers(norm["utterances"], null_speaker_policy="hold_for_verification")
        self.assertEqual([u["raw_id"] for u in part["assigned_utterances"]], [64, 68])
        self.assertEqual(len(part["speaker_verification_targets"]), 1)
        target = part["speaker_verification_targets"][0]
        self.assertEqual(target["id"], 67)
        self.assertIsNone(target["speaker"])
        self.assertFalse(target["inferred_identity_applied"])
        self.assertEqual(target["verification_status"], "pending_human_speaker_verification")
        self.assertEqual(target["context_prev_speaker"], "SPEAKER_00")
        self.assertEqual(target["context_next_speaker"], "SPEAKER_00")

        # 2. `null_speaker_policy="fail"` raises UnverifiedSpeakerError
        with self.assertRaises(dp.UnverifiedSpeakerError):
            dp.partition_and_manage_null_speakers(norm["utterances"], null_speaker_policy="fail")

        # 3. Explicit human verification override (`--assign-speaker 67=SPEAKER_01`) assigns only when explicitly specified
        part_override = dp.partition_and_manage_null_speakers(
            norm["utterances"],
            explicit_speaker_assignments={"67": "SPEAKER_01"},
            null_speaker_policy="fail",
        )
        self.assertEqual(len(part_override["speaker_verification_targets"]), 0)
        self.assertEqual(len(part_override["explicit_speaker_assignments"]), 1)
        self.assertEqual(part_override["assigned_utterances"][1]["speaker"], "SPEAKER_01")

    def test_phase12_measured_japanese_duration_overflow_and_inter_speaker_overlap_adjustment(self):
        """Requirement 5: Measure actual Japanese WAV duration, non-destructively adjust slot overflow &
        artificial TTS-expansion inter-speaker spill, and preserve genuine multi-speaker overlaps.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Utterance 1 (SPEAKER_00): slot is 1886.0..1889.0 (3.0s), but raw WAV is 4.10s (overflows into SPEAKER_01 at 1889.2s)
            wav1 = _write_test_wav(root / "orig" / "u1.wav", duration=4.10, rate=24000)
            wav1_sha_before = hashlib.sha256(wav1.read_bytes()).hexdigest()
            # Utterance 2 (SPEAKER_01): starts at 1889.2..1892.0 (did NOT overlap with u1 in English: 1889.0 <= 1889.2)
            wav2 = _write_test_wav(root / "orig" / "u2.wav", duration=2.40, rate=24000)
            # Utterance 3 (SPEAKER_00): genuinely overlaps with Utterance 2 in English (1890.5..1892.5)
            wav3 = _write_test_wav(root / "orig" / "u3.wav", duration=1.80, rate=24000)

            entries = [
                {
                    "id": 64,
                    "speaker": "SPEAKER_00",
                    "text_ja": "日本語の音声が元の英語枠より少し長くなったテスト文です。",
                    "mora_count": 28,
                    "start": 1886.0,
                    "end": 1889.0,
                    "source_start": 1886.0,
                    "source_end": 1889.0,
                    "audio_file": str(wav1),
                },
                {
                    "id": 65,
                    "speaker": "SPEAKER_01",
                    "text_ja": "次の話者の応答セリフです。",
                    "mora_count": 13,
                    "start": 1889.2,
                    "end": 1892.0,
                    "source_start": 1889.2,
                    "source_end": 1892.0,
                    "audio_file": str(wav2),
                },
                {
                    "id": 66,
                    "speaker": "SPEAKER_00",
                    "text_ja": "自然な相槌の重なりです。",
                    "mora_count": 12,
                    "start": 1890.5,
                    "end": 1892.5,
                    "source_start": 1890.5,
                    "source_end": 1892.5,
                    "audio_file": str(wav3),
                },
            ]

            res = dp.analyze_timeline_collisions_and_overlaps(
                entries,
                selection_start=1885.0,
                selection_end=1915.0,
                max_timing_shift=1.2,
                max_inter_speaker_shift=0.6,
                auto_tempo_adjust=True,
                max_atempo=1.25,
                adjusted_wav_dir=root / "adjusted",
            )
            # Original WAV1 must NOT be modified on disk
            self.assertEqual(hashlib.sha256(wav1.read_bytes()).hexdigest(), wav1_sha_before)
            # Duration adjustment (trim / bounded atempo) was applied to u1 into `root / "adjusted"`
            self.assertGreaterEqual(len(res["duration_adjustments"]), 1)
            self.assertLess(res["placements"]["64"]["actual_tts_duration_seconds"], 4.10)
            # Genuine overlap between 65 (SPEAKER_01) and 66 (SPEAKER_00) is preserved
            self.assertGreaterEqual(len(res["preserved_other_speaker_overlaps"]), 1)

    def test_phase12_real_video_1885_1915_e2e_cli_integration_and_non_destructive_verification(self):
        """Full Phase 12 E2E test with `output/phase11_1885_1915/source.mp4` and `output/phase11_1885_1915_inputs/dubbing_script.json`:
        - X-vector voice synthesis from video-extracted speaker reference WAVs (no silent Preset fallback)
        - Utterance IDs 64–66 merged into a single continuous utterance `64-66`
        - `speaker: null` utterance held as a verification target (`speaker_verification_targets.json`)
        - Measured Japanese WAV durations, overflow/overlap adjustment, original video preservation, subtitles, and MP4 output
        - Non-destructive SHA-256 verification of all existing input artifacts
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

            lang_dir = Path(pe.__file__).parent / "language"
            created_lang = False
            created_voice = False
            voice_file = lang_dir / "voice.py"
            if not lang_dir.exists():
                lang_dir.mkdir(parents=True)
                created_lang = True
            if not voice_file.exists():
                voice_file.write_text("# mock voice contract\n", encoding="utf-8")
                created_voice = True

            # Create `output/phase11_1885_1915/source.mp4` (30.0s video clip)
            video_path = root / "output" / "phase11_1885_1915" / "source.mp4"
            dp.ensure_30s_verification_video(video_path, duration_seconds=30.0)
            video_sha_before = hashlib.sha256(video_path.read_bytes()).hexdigest()

            # Create `output/phase11_1885_1915_inputs/dubbing_script.json`
            script_path = root / "output" / "phase11_1885_1915_inputs" / "dubbing_script.json"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_doc = {
                "schema_version": 3,
                "selection_range": {"start": 1885.0, "end": 1915.0},
                "entries": [
                    {
                        "id": 63,
                        "speaker": "SPEAKER_01",
                        "start": 1885.8,
                        "end": 1889.2,
                        "text_en": "Let us look at how the system behaves in real time.",
                        "text_ja": "システムがリアルタイムでどのように動作するか見てみましょう。",
                    },
                    {
                        "id": 64,
                        "speaker": "SPEAKER_00",
                        "start": 1889.6,
                        "end": 1892.4,
                        "text_en": "When we feed the live sensor stream into the model,",
                        "text_ja": "ライブセンサーのストリームをモデルに入力すると。",
                    },
                    {
                        "id": 65,
                        "speaker": None,
                        "start": 1892.6,
                        "end": 1895.5,
                        "text_en": "and extract the speaker representation across frames,",
                        "text_ja": "フレーム全体から話者表現を抽出して。",
                    },
                    {
                        "id": 66,
                        "speaker": "SPEAKER_00",
                        "start": 1895.7,
                        "end": 1899.2,
                        "text_en": "it reproduces the natural prosody without losing voice timbre.",
                        "text_ja": "元の声質を失わずに自然な韻律を再現できます。",
                    },
                    {
                        "id": 67,
                        "speaker": None,
                        "start": 1899.8,
                        "end": 1901.4,
                        "text_en": "Right, exactly.",
                        "text_ja": "ええ、その通りです。",
                    },
                    {
                        "id": 68,
                        "speaker": "SPEAKER_01",
                        "start": 1902.0,
                        "end": 1906.5,
                        "text_en": "And what happens when two speakers talk almost at once?",
                        "text_ja": "では、二人の話者がほぼ同時に話した場合はどうなりますか？",
                    },
                    {
                        "id": 69,
                        "speaker": "SPEAKER_00",
                        "start": 1905.8,
                        "end": 1913.5,
                        "text_en": "We preserve natural overlaps while resolving same-speaker collisions.",
                        "text_ja": "同一話者の衝突を解消しながら、異なる話者の自然な重なりはそのまま保持します。",
                    },
                ],
            }
            script_path.write_text(json.dumps(script_doc, ensure_ascii=False, indent=2), encoding="utf-8")
            script_sha_before = hashlib.sha256(script_path.read_bytes()).hexdigest()

            synthesized_texts = []

            class DummyNative:
                def __init__(self):
                    self._icl_cache = {}

                def generate(self, **kwargs):
                    if kwargs.get("ref_text") is not None:
                        raise AssertionError("Phase 12 X-vector mode must always pass ref_text=None")
                    synthesized_texts.append(kwargs.get("text"))
                    # Make duration proportional to text length to exercise duration measurement & adjustment
                    dur_s = max(1.8, min(6.2, len(kwargs.get("text", "")) * 0.11))
                    frames = int(round(dur_s * 24000))
                    t = np.arange(frames, dtype=np.float64) / 24000.0
                    sig = (0.24 * np.sin(2.0 * np.pi * 138.0 * t)).astype(np.float32)
                    return [types.SimpleNamespace(audio=sig, sample_rate=24000, token_count=25)]

            class DummyMLXWrapper:
                def __init__(self):
                    self.model = DummyNative()

                @classmethod
                def from_pretrained(cls, _path):
                    return cls()

                def create_voice_clone_prompt(self, **kwargs):
                    if not kwargs.get("x_vector_only_mode"):
                        raise AssertionError("Phase 12 must use x_vector_only_mode=True")
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
                        "model_revision": "e7dd0585652209fa0d7783659aad4e8a324de11c",
                        "settings": dict(self.settings),
                    }

                def synthesize(self, **kwargs):
                    raise AssertionError("Preset synthesizer must NEVER be called in Phase 12 X-vector run!")

            fake_vc = types.ModuleType("voice_cloning")
            fake_vc.digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            fake_vc.profile_snapshot = lambda prof: (_ for _ in ()).throw(
                AssertionError("Legacy voice_cloning.profile_snapshot ceremony must not be required for TechnicalVoiceReference!")
            )

            fake_tts_gen = types.ModuleType("tts_generator")
            fake_tts_gen.QwenPresetSynthesizer = FakePresetSynthesizer

            fake_mx = types.ModuleType("mlx.core")
            fake_mx.random = types.SimpleNamespace(seed=lambda s: None)
            fake_mx.get_peak_memory = lambda: 2048
            fake_mlx_pkg = types.ModuleType("mlx")
            fake_mlx_pkg.core = fake_mx

            fake_sf = types.ModuleType("soundfile")
            fake_sf.read = lambda p, dtype="float32": pe._read_wav_float32(p)
            fake_sf.write = lambda p, arr, sr, subtype="PCM_16", format="WAV": _write_test_wav(p, duration=len(arr) / float(sr), rate=sr)

            fake_qwen_mlx = types.ModuleType("qwen_tts_mlx")
            fake_qwen_mlx.MLXQwen3TTS = DummyMLXWrapper

            fake_spec = types.SimpleNamespace(origin=str(fake_pkg / "__init__.py"))

            out_dir = root / "output" / "phase12_1885_1915"
            try:
                with mock.patch.dict(
                    sys.modules,
                    {
                        "voice_cloning": fake_vc,
                        "tts_generator": fake_tts_gen,
                        "mlx": fake_mlx_pkg,
                        "mlx.core": fake_mx,
                        "soundfile": fake_sf,
                        "qwen_tts_mlx": fake_qwen_mlx,
                    },
                ), mock.patch("importlib.util.find_spec", return_value=fake_spec):
                    rc = dp.main(
                        [
                            "run-phase12",
                            "--assign-speaker", "65=SPEAKER_00",
                            "--video",
                            str(video_path),
                            "--script",
                            str(script_path),
                            "--output",
                            str(out_dir),
                        ]
                    )
                    self.assertEqual(rc, 0)

                    # 1. Verify all required deliverables exist
                    self.assertTrue((out_dir / "final_ja.mp4").is_file())
                    self.assertTrue((out_dir / "final_ja_no_subtitles.mp4").is_file())
                    self.assertTrue((out_dir / "final_ja.srt").is_file())
                    self.assertTrue((out_dir / "final_ja.ass").is_file())
                    self.assertTrue((out_dir / "dubbed_audio.wav").is_file())
                    self.assertTrue((out_dir / "pipeline_report.json").is_file())
                    self.assertTrue((out_dir / "pipeline_report.md").is_file())
                    self.assertTrue((out_dir / "speaker_verification_targets.json").is_file())

                    report = json.loads((out_dir / "pipeline_report.json").read_text(encoding="utf-8"))
                    self.assertEqual(report["kind"], "transclip_phase12_dubbing_report")
                    self.assertEqual(report["status"], "completed_diagnostic")
                    self.assertEqual(report["completion"]["video_generation"]["status"], "succeeded")
                    self.assertEqual(report["completion"]["quality_review"]["status"], "pending")
                    self.assertFalse(report["completion"]["completed_clean"])

                    # 2. Verify X-vector synthesis for all active speakers & no Preset usage
                    self.assertTrue(report["voice_routing"]["uses_clone"])
                    self.assertFalse(report["voice_routing"]["uses_preset"])
                    self.assertEqual(report["voice_routing"]["effective_clone_method"], "xvector")
                    for item in report["voice_routing"]["speaker_routing_details"]:
                        self.assertEqual(item["mode"], "approved_clone")
                        self.assertEqual(item["clone_method"], "xvector")
                        self.assertTrue(item["x_vector_only_mode"])
                        self.assertEqual(item["reference_origin"], "source_video_extracted")

                    # 3. Verify utterance IDs 64–66 were merged into `64-66` and synthesized as 1 combined sentence
                    self.assertEqual(report["utterance_merging"]["merged_group_count"], 1)
                    mg = report["utterance_merging"]["merged_groups"][0]
                    self.assertEqual(mg["merged_id"], "64-66")
                    self.assertEqual(mg["merged_from_ids"], [64, 65, 66])
                    self.assertEqual(len(synthesized_texts), 4)  # 63, 64-66, 68, 69 (67 held as speaker: null)

                    # 4. Verify `speaker: null` utterance 67 was held as a verification target and NOT guessed
                    svm = report["speaker_verification_management"]
                    self.assertEqual(svm["unassigned_speaker_target_count"], 1)
                    self.assertEqual(svm["auto_guessed_speaker_count"], 0)
                    self.assertEqual(svm["targets"][0]["id"], 67)
                    self.assertIsNone(svm["targets"][0]["speaker"])

                    # 5. Verify original video stream preservation and clip-relative timestamps [0.0, 30.0]
                    mv = report["media_verification"]
                    self.assertTrue(mv["has_video"])
                    self.assertTrue(mv["has_audio"])
                    self.assertTrue(mv["video_dimensions_preserved"])
                    self.assertAlmostEqual(mv["duration_seconds"], 30.0, delta=0.3)
                    self.assertTrue(mv["relative_time_synchronized"])

                    # 6. Verify input files were NEVER modified
                    self.assertTrue(report["input_integrity_verification"]["verified"])
                    self.assertEqual(hashlib.sha256(video_path.read_bytes()).hexdigest(), video_sha_before)
                    self.assertEqual(hashlib.sha256(script_path.read_bytes()).hexdigest(), script_sha_before)
            finally:
                if created_voice and voice_file.exists():
                    voice_file.unlink()
                if created_lang and lang_dir.exists():
                    lang_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
