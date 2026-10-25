"""Unit and integration tests for Cross-lingual Voice Cloning Japanese Prosody Improvement (prosody_experiments.py)."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave

import numpy as np

import prosody_experiments as pe


def _write_test_wav(path, *, duration=1.6, rate=24000, f0_start=135.0, f0_end=135.0, add_pause=True):
    frames = int(round(duration * rate))
    t = np.arange(frames, dtype=np.float64) / float(rate)
    f0 = np.linspace(f0_start, f0_end, frames)
    phase = 2.0 * np.pi * np.cumsum(f0) / float(rate)
    sig = 0.28 * np.sin(phase) + 0.09 * np.sin(2.0 * phase) + 0.04 * np.sin(3.0 * phase)
    fade = min(int(0.08 * rate), frames // 4)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        sig[:fade] *= ramp
        sig[-fade:] *= ramp[::-1]
    if add_pause and frames > int(0.6 * rate):
        p0 = int(0.45 * frames)
        p1 = min(frames - int(0.15 * rate), p0 + int(0.18 * rate))
        sig[p0:p1] *= 0.002
    pcm = np.clip(np.round(sig * 32767.0), -32768, 32767).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())
    return Path(path)


def _build_approved_consent_and_review(root, *, speaker, ref_wav, ref_txt):
    """Create synthetic unit-test fixtures for the real Voice Profile contract."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    audio_sha = hashlib.sha256(Path(ref_wav).read_bytes()).hexdigest()

    # Preserve whitespace and trailing newlines exactly.
    text_sha = hashlib.sha256(Path(ref_txt).read_bytes()).hexdigest()

    fixture_reviewer = "synthetic_unit_test_fixture"

    consent = {
        "authorized": True,
        "scope": "voice_cloning",
        "speaker_id": speaker,
        "reference_audio_sha256": audio_sha,
        "evidence": "Generated synthetic test waveform; unit-test fixture only.",
        "verified_by": fixture_reviewer,
    }

    review = {
        "transcript": {
            "status": "verified",
            "verified_by": fixture_reviewer,
            "reference_audio_sha256": audio_sha,
            "reference_text_sha256": text_sha,
        },
        "quality": {
            "reference_audio_sha256": audio_sha,
            "verified_by": fixture_reviewer,
            "speaker_identity_verified": True,
            "single_speaker_verified": True,
            "intelligibility_verified": True,
            "boundaries_verified": True,
            "known_overlap_seconds": 0,
            "unresolved_issues": [],
            "source_kind": "original",
        },
    }

    consent_path = root / "consent.json"
    review_path = root / "review.json"

    consent_path.write_text(
        json.dumps(consent, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    review_path.write_text(
        json.dumps(review, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return consent_path, review_path, audio_sha, text_sha


class TestProsodyExperiments(unittest.TestCase):
    def test_analyze_japanese_prosody_distinguishes_falling_vs_rising_sentence_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decl_wav = _write_test_wav(root / "decl.wav", duration=1.8, f0_start=150.0, f0_end=115.0)
            inter_wav = _write_test_wav(root / "inter.wav", duration=1.8, f0_start=130.0, f0_end=185.0)

            decl = pe.analyze_japanese_prosody(decl_wav, mora_count=26)
            inter = pe.analyze_japanese_prosody(inter_wav, mora_count=27)

            self.assertLess(decl["sentence_final_f0_delta_semitones"], -1.5)
            self.assertGreater(inter["sentence_final_f0_delta_semitones"], 1.5)
            self.assertGreater(decl["mean_mora_duration_ms"], 30.0)
            self.assertGreater(decl["spoken_mora_per_second"], 10.0)
            self.assertGreaterEqual(decl["pause_count"], 1)

    def test_build_experiment_c_specs_varies_single_parameter_at_a_time(self):
        base = pe.BASELINE_CLONE_SETTINGS
        for method in ("xvector", "icl"):
            specs = pe.build_experiment_c_specs(method)
            self.assertGreaterEqual(len(specs), 5)
            for spec in specs:
                diffs = [k for k, v in spec["settings"].items() if base[k] != v]
                self.assertEqual(
                    diffs,
                    [spec["changed_parameter"]],
                    f"Condition {spec['mode_name']} changed multiple parameters: {diffs}",
                )

    def test_comparison_bundle_creates_only_available_modes_and_never_fabricates_missing_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_wav = _write_test_wav(root / "src" / "ref.wav", duration=2.8, f0_start=132.0, f0_end=128.0)

            preset_dir = root / "runs" / "preset"
            icl_dir = root / "runs" / "icl"
            xvec_dir = root / "runs" / "xvector"
            for mode_dir, f0_end in [(preset_dir, 145.0), (icl_dir, 125.0), (xvec_dir, 138.0)]:
                entries = []
                for idx, sample in enumerate(pe.CORE_SAMPLES, start=1):
                    wav_path = _write_test_wav(
                        mode_dir / f"{sample['dub_id']}.wav",
                        duration=2.1 + 0.1 * idx,
                        f0_start=135.0,
                        f0_end=f0_end + (15.0 if sample["sentence_type"] == "interrogative" else -8.0),
                    )
                    entries.append(
                        {
                            "id": sample["dub_id"],
                            "generation_status": "complete",
                            "audio_file": str(wav_path),
                            "mora_count": sample["mora_count"],
                            "cache_reused": False,
                            "backend_result": {
                                "clone_method": mode_dir.name,
                                "x_vector_only_mode": mode_dir.name == "xvector",
                                "processing_seconds": 1.25,
                                "conditioning_reused": idx > 1,
                            },
                            "tts": {
                                "model": pe.DEFAULT_PRESET_MODEL if mode_dir.name == "preset" else pe.DEFAULT_BASE_MODEL,
                                "model_revision": "e7dd0585652209fa0d7783659aad4e8a324de11c",
                                "clone_method": mode_dir.name,
                                "settings": dict(pe.BASELINE_CLONE_SETTINGS),
                            },
                        }
                    )
                (mode_dir / "feedback.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")

            out_all = root / "output_all"
            comp_path, comp = pe.build_prosody_comparison_bundle(
                reference_wav=ref_wav,
                speaker_id="SYNTH_EN_01",
                mode_feedbacks={
                    "preset": preset_dir / "feedback.json",
                    "icl": icl_dir / "feedback.json",
                    "xvector": xvec_dir / "feedback.json",
                },
                output_dir=out_all,
            )
            self.assertTrue(comp_path.is_file())
            self.assertEqual(comp["available_modes"], ["preset", "icl", "xvector"])
            self.assertTrue((out_all / "reference" / "english_reference.wav").is_file())
            for mode in ("preset", "icl", "xvector"):
                for s in ("sample_01.wav", "sample_02.wav", "sample_03.wav"):
                    self.assertTrue((out_all / mode / s).is_file())
            self.assertTrue((out_all / "comparison.md").is_file())
            self.assertTrue((out_all / "review.html").is_file())
            html_text = (out_all / "review.html").read_text(encoding="utf-8")
            self.assertIn("SYNTH_EN_01", html_text)
            self.assertIn("xvector/sample_01.wav", html_text)

            out_partial = root / "output_partial"
            _, comp_partial = pe.build_prosody_comparison_bundle(
                reference_wav=ref_wav,
                speaker_id="SYNTH_EN_01",
                mode_feedbacks={
                    "preset": preset_dir / "feedback.json",
                    "icl": icl_dir / "feedback.json",
                    "xvector": None,
                },
                output_dir=out_partial,
            )
            self.assertEqual(comp_partial["available_modes"], ["preset", "icl"])
            self.assertFalse((out_partial / "xvector").exists())

    def test_synthesizer_routes_xvector_vs_icl_and_isolates_cache_identity(self):
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

            ref_wav = _write_test_wav(root / "ref.wav", duration=1.0, rate=24000)
            ref_sha = hashlib.sha256(ref_wav.read_bytes()).hexdigest()

            prompt_calls = []
            generate_calls = []

            class DummyNative:
                def __init__(self, leak_icl=False):
                    self._icl_cache = {}
                    self.leak_icl = leak_icl

                def generate(self, **kwargs):
                    generate_calls.append(kwargs)
                    if kwargs.get("ref_text") is not None or self.leak_icl:
                        self._icl_cache["k"] = "cached_icl"
                    return [
                        types.SimpleNamespace(
                            audio=np.sin(np.linspace(0, 20, 4800)).astype(np.float32) * 0.25,
                            sample_rate=24000,
                            token_count=18,
                        )
                    ]

            class DummyMLXWrapper:
                leak_icl = False

                def __init__(self):
                    self.model = DummyNative(leak_icl=self.leak_icl)

                @classmethod
                def from_pretrained(cls, _path):
                    return cls()

                def create_voice_clone_prompt(self, **kwargs):
                    prompt_calls.append(kwargs)
                    return types.SimpleNamespace(
                        ref_audio=kwargs["ref_audio"][0],
                        ref_text=None if kwargs.get("x_vector_only_mode") else kwargs.get("ref_text"),
                        x_vector_only_mode=bool(kwargs.get("x_vector_only_mode")),
                    )

            class FakePresetSynthesizer:
                def __init__(self, model_id, seed=0, expected_model_type="base"):
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
                    self.identity = {"model": model_id, "model_revision": "rev123"}

            fake_tts_gen = types.ModuleType("tts_generator")
            fake_tts_gen.QwenPresetSynthesizer = FakePresetSynthesizer

            fake_vc = types.ModuleType("voice_cloning")
            fake_vc.digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            fake_vc.profile_snapshot = lambda prof: {
                "profile_fingerprint": f"fp-{prof.speaker_id}",
                "reference_audio_sha256": ref_sha,
                "reference_text_sha256": "txtsha",
            }

            fake_mx = types.ModuleType("mlx.core")
            fake_mx.random = types.SimpleNamespace(seed=lambda s: None)
            fake_mx.get_peak_memory = lambda: 1024

            fake_mlx_pkg = types.ModuleType("mlx")
            fake_mlx_pkg.core = fake_mx

            fake_sf = types.ModuleType("soundfile")
            fake_sf.read = lambda p, dtype="float32": pe._read_wav_float32(p)
            fake_sf.write = lambda p, arr, sr, subtype="PCM_16", format="WAV": _write_test_wav(p, duration=len(arr) / float(sr), rate=sr)

            fake_qwen_mlx = types.ModuleType("qwen_tts_mlx")
            fake_qwen_mlx.MLXQwen3TTS = DummyMLXWrapper

            fake_spec = types.SimpleNamespace(origin=str(fake_pkg / "__init__.py"))

            try:
                with mock.patch.dict(
                    sys.modules,
                    {
                        "tts_generator": fake_tts_gen,
                        "voice_cloning": fake_vc,
                        "mlx": fake_mlx_pkg,
                        "mlx.core": fake_mx,
                        "soundfile": fake_sf,
                        "qwen_tts_mlx": fake_qwen_mlx,
                    },
                ), mock.patch("importlib.util.find_spec", return_value=fake_spec):
                    xvec_synth = pe.QwenProsodyCloneSynthesizer(clone_method="xvector", temperature=0.7)
                    icl_synth = pe.QwenProsodyCloneSynthesizer(clone_method="icl", temperature=0.9, ref_trailing_silence_seconds=0.25)

                    self.assertNotEqual(xvec_synth.identity["clone_method"], icl_synth.identity["clone_method"])
                    self.assertTrue(xvec_synth.identity["x_vector_only_mode"])
                    self.assertFalse(icl_synth.identity["x_vector_only_mode"])

                    prof = types.SimpleNamespace(
                        speaker_id="SYNTH_EN_01",
                        reference_audio=str(ref_wav),
                        reference_text="English reference transcript.",
                    )

                    out_xvec = root / "out_xvec.wav"
                    res_xvec = xvec_synth.synthesize(text="新しい技術が、私たちの暮らしを少しずつ変えています。", profile=prof, output_path=out_xvec)
                    self.assertTrue(res_xvec["x_vector_only_mode"])
                    self.assertIsNone(res_xvec["reference_text_sha256"])
                    self.assertTrue(prompt_calls[-1]["x_vector_only_mode"])
                    self.assertIsNone(prompt_calls[-1]["ref_text"])
                    self.assertIsNone(generate_calls[-1]["ref_text"])
                    self.assertEqual(generate_calls[-1]["lang_code"], "Japanese")
                    self.assertEqual(generate_calls[-1]["temperature"], 0.7)

                    out_icl = root / "out_icl.wav"
                    res_icl = icl_synth.synthesize(text="新しい技術が、私たちの暮らしを少しずつ変えています。", profile=prof, output_path=out_icl)
                    self.assertFalse(res_icl["x_vector_only_mode"])
                    self.assertEqual(res_icl["reference_text_sha256"], "txtsha")
                    self.assertFalse(prompt_calls[-1]["x_vector_only_mode"])
                    self.assertEqual(prompt_calls[-1]["ref_text"], "English reference transcript.")
                    self.assertEqual(generate_calls[-1]["ref_text"], "English reference transcript.")
                    self.assertEqual(len(prompt_calls[-1]["ref_audio"][0]), 30000)
                    self.assertEqual(hashlib.sha256(ref_wav.read_bytes()).hexdigest(), ref_sha)

                    DummyMLXWrapper.leak_icl = True
                    bad_xvec = pe.QwenProsodyCloneSynthesizer(clone_method="xvector")
                    with self.assertRaisesRegex(RuntimeError, "ICL path leaked"):
                        bad_xvec.synthesize(text="テストです。", profile=prof, output_path=root / "bad.wav")
            finally:
                if created_voice and voice_file.exists():
                    voice_file.unlink()
                if created_lang and lang_dir.exists():
                    lang_dir.rmdir()

    def test_run_all_cli_integration_exercises_exact_main_branch_contracts(self):
        """End-to-end integration test for `run-all` and `compare` CLI verifying:
        1. `cross_lingual_clone.build_phase8_benchmark_script`
        2. `cross_lingual_clone.audit_english_reference(speaker=..., reference_text_path=...) -> (report_path, report)`
        3. `voice_cloning.register_profile`, `load_profiles`, and `profile_snapshot`
        4. `tts_feedback.run_feedback(script_path, output_dir, *, backend=..., voices=..., voice_profiles=...)`:
           - Preset passes `voices={'SYNTH_EN_01': 'Ryan'}`
           - ICL & X-vector & Exp C pass `voices={'SYNTH_EN_01': 'approved_clone:<profile_fingerprint>'}` AND `voice_profiles={'SYNTH_EN_01': profile}`
        5. `voice_quality` / `voice_profiler` existing embedding implementation (`PyannoteBackend`) without `PyannoteEmbedder`
        6. Speaker ID `SYNTH_EN_01`
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

            call_log = {
                "audit_english_reference": [],
                "build_phase8_benchmark_script": [],
                "register_profile": [],
                "load_profiles": [],
                "run_feedback": [],
                "embed": [],
            }

            class CloneBlockedForReviewError(RuntimeError):
                pass

            def strict_audit_english_reference(*, speaker, reference_audio, reference_text_path, consent_path, review_path, output_dir, generation_record_path=None):
                self.assertEqual(speaker, "SYNTH_EN_01")
                self.assertTrue(Path(reference_audio).is_file())
                self.assertTrue(Path(reference_text_path).is_file())
                self.assertTrue(Path(consent_path).is_file())
                self.assertTrue(Path(review_path).is_file())
                call_log["audit_english_reference"].append(
                    {
                        "speaker": speaker,
                        "reference_audio": str(reference_audio),
                        "reference_text_path": str(reference_text_path),
                    }
                )
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                rep_path = out_dir / "audit_report.json"
                rep = {
                    "speaker": speaker,
                    "status": "verified_ready_for_clone",
                    "ready_for_clone_synthesis": True,
                    "mechanical_passed": True,
                    "pending_human_items": [],
                }
                rep_path.write_text(json.dumps(rep), encoding="utf-8")
                return rep_path, rep

            def strict_build_phase8_benchmark_script(output_path, *, speaker="SYNTH_EN_01", voice_mode="approved_clone", preset_voice="Ryan"):
                self.assertEqual(speaker, "SYNTH_EN_01")
                self.assertIn(voice_mode, ("preset", "approved_clone"))
                call_log["build_phase8_benchmark_script"].append({"voice_mode": voice_mode, "speaker": speaker})
                out_p = Path(output_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                doc = {
                    "schema_version": 1,
                    "speaker": speaker,
                    "voice_mode": voice_mode,
                    "utterances": [
                        {
                            "id": s["dub_id"],
                            "speaker": speaker,
                            "voice_mode": voice_mode,
                            "preset_voice": preset_voice,
                            "text_en": s["text_en"],
                            "text_ja": s["text_ja"],
                            "mora_count": s["mora_count"],
                        }
                        for s in pe.CORE_SAMPLES
                    ],
                }
                out_p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
                return out_p, doc

            def strict_register_profile(*, speaker, reference_audio, reference_text_path, consent_path, review_path, output_dir, preset_voice="Ryan", generation_record_path=None):
                self.assertEqual(speaker, "SYNTH_EN_01")
                self.assertTrue(Path(reference_text_path).is_file())
                call_log["register_profile"].append({"speaker": speaker, "output_dir": str(output_dir)})
                reg_dir = Path(output_dir)
                reg_dir.mkdir(parents=True, exist_ok=True)
                reg_path = reg_dir / "voice_profiles.json"
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
                reg_path.write_text(json.dumps(reg_doc), encoding="utf-8")
                return reg_path, reg_doc

            def strict_load_profiles(registry_path):
                self.assertTrue(Path(registry_path).is_file())
                call_log["load_profiles"].append(str(registry_path))
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

            def strict_run_feedback(script_path, output_dir, *, backend=None, voices=None, voice_profiles=None):
                self.assertIsNotNone(backend)
                self.assertIsInstance(voices, dict)
                self.assertIn("SYNTH_EN_01", voices)
                script_doc = json.loads(Path(script_path).read_text(encoding="utf-8"))
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                entries = []
                for idx, u in enumerate(script_doc["utterances"], start=1):
                    spk = u["speaker"]
                    voice_spec = voices[spk]
                    wav_out = out_dir / "audio" / f"{u['id']}.wav"
                    wav_out.parent.mkdir(parents=True, exist_ok=True)
                    if "approved_clone" in getattr(backend, "supported_voice_modes", set()):
                        # Enforce exact main branch contract:
                        # 1. `voices[speaker]` MUST be `approved_clone:<profile_fingerprint>`
                        # 2. `voice_profiles` MUST be provided and contain the matching `VoiceProfile`
                        self.assertEqual(voice_spec, f"approved_clone:{expected_fp}")
                        self.assertIsNotNone(voice_profiles, "Approved Voice Clone backend requires voice_profiles")
                        self.assertIn(spk, voice_profiles)
                        prof = voice_profiles[spk]
                        self.assertEqual(strict_profile_snapshot(prof)["profile_fingerprint"], expected_fp)
                        backend_res = backend.synthesize(
                            text=u["text_ja"],
                            profile=prof,
                            output_path=wav_out,
                        )
                    else:
                        self.assertEqual(voice_spec, "Ryan")
                        self.assertIsNone(voice_profiles)
                        _write_test_wav(wav_out, duration=2.0 + 0.1 * idx, rate=24000, f0_start=135.0, f0_end=145.0)
                        backend_res = {"clone_method": "preset", "x_vector_only_mode": False, "processing_seconds": 0.45}
                    entries.append(
                        {
                            "id": u["id"],
                            "speaker": spk,
                            "voice": voice_spec,
                            "generation_status": "complete",
                            "audio_file": str(wav_out),
                            "mora_count": u["mora_count"],
                            "cache_reused": False,
                            "backend_result": backend_res,
                            "tts": dict(backend.identity),
                        }
                    )
                fb_doc = {"tts": dict(backend.identity), "entries": entries}
                fb_path = out_dir / "feedback.json"
                fb_path.write_text(json.dumps(fb_doc, ensure_ascii=False), encoding="utf-8")
                call_log["run_feedback"].append(
                    {
                        "output_dir": str(out_dir),
                        "voices": dict(voices),
                        "has_voice_profiles": voice_profiles is not None,
                        "clone_method": backend.identity.get("clone_method", "preset"),
                        "x_vector_only_mode": backend.identity.get("x_vector_only_mode", False),
                    }
                )
                return fb_path, fb_doc

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
                        "model": model_id,
                        "model_revision": "e7dd0585652209fa0d7783659aad4e8a324de11c",
                        "settings": dict(self.settings),
                    }

            class DummyNative:
                def __init__(self):
                    self._icl_cache = {}

                def generate(self, **kwargs):
                    if kwargs.get("ref_text") is not None:
                        self._icl_cache["k"] = "icl_state"
                    f0_end = 158.0 if "でしょうか" in kwargs.get("text", "") else 122.0
                    frames = 48000
                    t = np.arange(frames, dtype=np.float64) / 24000.0
                    f0 = np.linspace(136.0, f0_end, frames)
                    phase = 2.0 * np.pi * np.cumsum(f0) / 24000.0
                    sig = (0.25 * np.sin(phase) + 0.08 * np.sin(2.0 * phase)).astype(np.float32)
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

            class FakePyannoteBackend:
                """Matches voice_profiler.PyannoteBackend (no PyannoteEmbedder in voice_quality)."""

                def embed(self, samples, sample_rate):
                    call_log["embed"].append((len(samples), int(sample_rate)))
                    arr = np.asarray(samples, dtype=np.float64)
                    return np.array([float(np.sqrt(np.mean(arr ** 2))), float(np.mean(np.abs(arr))), 1.0], dtype=np.float64)

            fake_clc = types.ModuleType("cross_lingual_clone")
            fake_clc.CloneBlockedForReviewError = CloneBlockedForReviewError
            fake_clc.audit_english_reference = strict_audit_english_reference
            fake_clc.build_phase8_benchmark_script = strict_build_phase8_benchmark_script

            fake_vc = types.ModuleType("voice_cloning")
            fake_vc.register_profile = strict_register_profile
            fake_vc.load_profiles = strict_load_profiles
            fake_vc.digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            fake_vc.profile_snapshot = strict_profile_snapshot

            fake_fb = types.ModuleType("tts_feedback")
            fake_fb.run_feedback = strict_run_feedback

            fake_tts_gen = types.ModuleType("tts_generator")
            fake_tts_gen.QwenPresetSynthesizer = FakePresetSynthesizer

            fake_vq = types.ModuleType("voice_quality")
            fake_vq.PyannoteBackend = FakePyannoteBackend

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

            out_dir = root / "output" / "prosody_experiments"
            try:
                with mock.patch.dict(
                    sys.modules,
                    {
                        "cross_lingual_clone": fake_clc,
                        "voice_cloning": fake_vc,
                        "tts_feedback": fake_fb,
                        "tts_generator": fake_tts_gen,
                        "voice_quality": fake_vq,
                        "mlx": fake_mlx_pkg,
                        "mlx.core": fake_mx,
                        "soundfile": fake_sf,
                        "qwen_tts_mlx": fake_qwen_mlx,
                    },
                ), mock.patch("importlib.util.find_spec", return_value=fake_spec):
                    rc = pe.main(
                        [
                            "run-all",
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
                            "--preset-voice",
                            "Ryan",
                            "--include-experiment-c",
                            "--embeddings",
                            "--output",
                            str(out_dir),
                        ]
                    )
                    self.assertEqual(rc, 0)

                    self.assertEqual(len(call_log["audit_english_reference"]), 1)
                    self.assertEqual(call_log["audit_english_reference"][0]["speaker"], "SYNTH_EN_01")
                    self.assertEqual(len(call_log["register_profile"]), 1)
                    self.assertEqual(len(call_log["load_profiles"]), 1)
                    self.assertEqual(
                        [c["voice_mode"] for c in call_log["build_phase8_benchmark_script"]],
                        ["preset", "approved_clone"],
                    )
                    self.assertEqual(len(call_log["run_feedback"]), 9)
                    # First call is preset (`voices={"SYNTH_EN_01": "Ryan"}`, `has_voice_profiles=False`)
                    self.assertEqual(call_log["run_feedback"][0]["voices"], {"SYNTH_EN_01": "Ryan"})
                    self.assertFalse(call_log["run_feedback"][0]["has_voice_profiles"])
                    # Remaining 8 calls (ICL, X-vector, and 6 Exp C modes) pass `approved_clone:<fp>` and `voice_profiles`
                    for fb_call in call_log["run_feedback"][1:]:
                        self.assertEqual(fb_call["voices"], {"SYNTH_EN_01": f"approved_clone:{expected_fp}"})
                        self.assertTrue(fb_call["has_voice_profiles"])

                    comp = json.loads((out_dir / "comparison.json").read_text(encoding="utf-8"))
                    self.assertEqual(comp["speaker_id"], "SYNTH_EN_01")
                    for expected_mode in (
                        "preset",
                        "icl",
                        "xvector",
                        "xvector_temp07",
                        "xvector_topp09",
                        "xvector_topk25",
                        "xvector_rep102",
                        "xvector_seed42",
                        "icl_pad040",
                    ):
                        self.assertIn(expected_mode, comp["available_modes"])
                        for s_id in ("sample_01", "sample_02", "sample_03"):
                            self.assertTrue((out_dir / expected_mode / f"{s_id}.wav").is_file())
                            mode_entry = comp["samples"][s_id]["modes"][expected_mode]
                            self.assertIsNotNone(mode_entry["reference_embedding_cosine"])
                            if expected_mode != "preset":
                                self.assertEqual(mode_entry["voice"], f"approved_clone:{expected_fp}")
                                self.assertEqual(mode_entry["profile_fingerprint"], expected_fp)

                    cmp_out = root / "output" / "prosody_compare_only"
                    rc_cmp = pe.main(
                        [
                            "compare",
                            "--speaker",
                            "SYNTH_EN_01",
                            "--reference-audio",
                            str(ref_wav),
                            "--preset-feedback",
                            str(out_dir / "_runs" / "preset_run" / "feedback.json"),
                            "--icl-feedback",
                            str(out_dir / "_runs" / "icl_run" / "feedback.json"),
                            "--xvector-feedback",
                            str(out_dir / "_runs" / "xvector_run" / "feedback.json"),
                            "--embeddings",
                            "--output",
                            str(cmp_out),
                        ]
                    )
                    self.assertEqual(rc_cmp, 0)
                    self.assertTrue((cmp_out / "comparison.json").is_file())
                    self.assertTrue((cmp_out / "review.html").is_file())

                    def blocked_audit(**kwargs):
                        return root / "blocked.json", {"ready_for_clone_synthesis": False, "pending_human_items": ["transcript_matches_speech"]}

                    fake_clc.audit_english_reference = blocked_audit
                    with self.assertRaises(CloneBlockedForReviewError):
                        pe.main(
                            [
                                "run-all",
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
                                "--output",
                                str(root / "blocked_out"),
                            ]
                        )
            finally:
                if created_voice and voice_file.exists():
                    voice_file.unlink()
                if created_lang and lang_dir.exists():
                    lang_dir.rmdir()

    def test_real_tts_feedback_integration_with_icl_and_xvector(self):
        """Integration test executing the ACTUAL `tts_feedback.py`, `voice_cloning.py`, and `cross_lingual_clone.py`
        from the repository with both ICL (`clone_method='icl'`) and X-vector (`clone_method='xvector'`) backends,
        verifying `voice_profiles` and `approved_clone:<profile_fingerprint>` propagation and cache isolation.
        """
        try:
            import cross_lingual_clone
            import tts_feedback
            import voice_cloning
            import voice_quality
        except ImportError:
            # In isolated environment where repo modules are not present, skip live repo execution
            return

        self.assertTrue(hasattr(cross_lingual_clone, "build_phase8_benchmark_script"))
        self.assertTrue(hasattr(cross_lingual_clone, "audit_english_reference"))
        audit_params = inspect.signature(cross_lingual_clone.audit_english_reference).parameters
        self.assertIn("speaker", audit_params)
        self.assertIn("reference_text_path", audit_params)

        self.assertTrue(hasattr(voice_cloning, "register_profile"))
        self.assertTrue(hasattr(voice_cloning, "load_profiles"))
        self.assertTrue(hasattr(voice_cloning, "profile_snapshot"))

        fb_params = inspect.signature(tts_feedback.run_feedback).parameters
        self.assertIn("backend", fb_params)
        self.assertIn("voices", fb_params)
        self.assertIn("voice_profiles", fb_params)

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
            consent_json, review_json, ref_sha, _ = _build_approved_consent_and_review(
                root / "inputs",
                speaker="SYNTH_EN_01",
                ref_wav=ref_wav,
                ref_txt=ref_txt,
            )

            # Register real VoiceProfile via repository's voice_cloning.register_profile & load_profiles
            reg_dir = root / "registry"
            reg_dir.mkdir(parents=True, exist_ok=True)
            reg_path = pe._invoke_register_profile(
                voice_cloning.register_profile,
                speaker="SYNTH_EN_01",
                reference_audio=ref_wav,
                reference_text_path=ref_txt,
                consent_path=consent_json,
                review_path=review_json,
                output_dir=reg_dir,
                preset_voice="Ryan",
            )
            voice_profiles = voice_cloning.load_profiles(reg_path)
            self.assertIn("SYNTH_EN_01", voice_profiles)
            profile = voice_profiles["SYNTH_EN_01"]
            snap = voice_cloning.profile_snapshot(profile)
            fp = snap["profile_fingerprint"]
            clone_voice_id = f"approved_clone:{fp}"

            # Build real benchmark dubbing_script.json via cross_lingual_clone.build_phase8_benchmark_script
            preset_script = pe._invoke_benchmark_script_builder(
                cross_lingual_clone.build_phase8_benchmark_script,
                output_path=root / "scripts" / "preset_script.json",
                speaker="SYNTH_EN_01",
                voice_mode="preset",
                preset_voice="Ryan",
                voice="Ryan",
            )
            clone_script = pe._invoke_benchmark_script_builder(
                cross_lingual_clone.build_phase8_benchmark_script,
                output_path=root / "scripts" / "clone_script.json",
                speaker="SYNTH_EN_01",
                voice_mode="approved_clone",
                preset_voice="Ryan",
                voice=clone_voice_id,
                profile=profile,
                profile_fingerprint=fp,
                voice_profiles=voice_profiles,
            )
            self.assertTrue(preset_script.is_file())
            self.assertTrue(clone_script.is_file())

            prompt_calls = []
            generate_calls = []

            class DummyNative:
                def __init__(self):
                    self._icl_cache = {}

                def generate(self, **kwargs):
                    generate_calls.append(kwargs)
                    if kwargs.get("ref_text") is not None:
                        self._icl_cache["k"] = "icl_state"
                    f0_end = 160.0 if "でしょうか" in kwargs.get("text", "") else 120.0
                    frames = 48000
                    t = np.arange(frames, dtype=np.float64) / 24000.0
                    f0 = np.linspace(135.0, f0_end, frames)
                    phase = 2.0 * np.pi * np.cumsum(f0) / 24000.0
                    sig = (0.25 * np.sin(phase) + 0.08 * np.sin(2.0 * phase)).astype(np.float32)
                    return [types.SimpleNamespace(audio=sig, sample_rate=24000, token_count=22)]

            class DummyMLXWrapper:
                def __init__(self):
                    self.model = DummyNative()

                @classmethod
                def from_pretrained(cls, _path):
                    return cls()

                def create_voice_clone_prompt(self, **kwargs):
                    prompt_calls.append(kwargs)
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

                def synthesize(self, *, text, profile, output_path):
                    _write_test_wav(output_path, duration=2.1, rate=24000, f0_start=135.0, f0_end=142.0)
                    return {
                        "sample_rate": 24000,
                        "sample_frames": 50400,
                        "token_count": 20,
                        "termination": "eos_before_limit",
                        "processing_seconds": 0.35,
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

            with mock.patch.dict(
                sys.modules,
                {
                    "mlx": fake_mlx_pkg,
                    "mlx.core": fake_mx,
                    "soundfile": fake_sf,
                    "qwen_tts_mlx": fake_qwen_mlx,
                },
            ), mock.patch("importlib.util.find_spec", return_value=fake_spec), mock.patch(
                "tts_generator.QwenPresetSynthesizer", FakePresetSynthesizer
            ):
                # 1. Verify REAL tts_feedback.run_feedback rejects missing voice_profiles for approved_clone
                icl_synth = pe.QwenProsodyCloneSynthesizer(clone_method="icl", seed=0)
                with self.assertRaises((ValueError, RuntimeError, KeyError)):
                    tts_feedback.run_feedback(
                        clone_script,
                        root / "missing_profiles_run",
                        backend=icl_synth,
                        voices={"SYNTH_EN_01": clone_voice_id},
                        voice_profiles=None,
                    )

                # 2. Execute full `run_prosody_experiments` through REAL `tts_feedback.run_feedback`
                out_dir = root / "prosody_out"
                comp_path, comp = pe.run_prosody_experiments(
                    speaker_id="SYNTH_EN_01",
                    reference_audio=ref_wav,
                    reference_text=ref_txt,
                    consent_path=consent_json,
                    review_path=review_json,
                    expected_sha256=ref_sha,
                    output_dir=out_dir,
                    preset_voice="Ryan",
                    include_experiment_c=False,
                    embeddings=False,
                )
                self.assertTrue(comp_path.is_file())
                self.assertEqual(comp["available_modes"], ["preset", "icl", "xvector"])

                # Verify both ICL and X-vector passed `approved_clone:<profile_fingerprint>` and `voice_profiles`
                for mode_name, expected_xvec in [("icl", False), ("xvector", True)]:
                    run_root = out_dir / "_runs" / f"{mode_name}_run"
                    feedback_paths = list(run_root.glob("runs/*/feedback.json"))
                    self.assertEqual(len(feedback_paths), 1)
                    fb_doc = json.loads(feedback_paths[0].read_text(encoding="utf-8"))
                    self.assertEqual(len(fb_doc["entries"]), 3)
                    for entry in fb_doc["entries"]:
                        self.assertEqual(entry["generation_status"], "complete")
                        actual_fp = entry["voice_profile"]["profile_fingerprint"]
                        self.assertEqual(entry["voice"], f"approved_clone:{actual_fp}")
                        self.assertEqual(
                            entry["voice_profile"]["profile"]["reference_audio_sha256"],
                            ref_sha,
                        )
                        self.assertEqual(
                        entry["backend_result"]["profile_fingerprint"],
                        entry["voice_profile"]["profile_fingerprint"],
                    )
                        self.assertEqual(entry["backend_result"]["clone_method"], mode_name)
                        self.assertEqual(entry["backend_result"]["x_vector_only_mode"], expected_xvec)


if __name__ == "__main__":
    unittest.main()
